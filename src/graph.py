"""
LangGraph agent graph - the central orchestrator for Track 1.

Graph flow:
  START -> classify -> local_infer
                          ├-- PASS  -> END          (free, no Fireworks tokens)
                          └-- FAIL  -> compress_prompt -> fireworks_call -> END

Local inference (Qwen2.5-3B) is attempted first for categories where a 3B model
is reliable (factual, sentiment, summarization, NER). Hard categories (code, logic,
math) and high-difficulty tasks skip straight to Fireworks.

State is fully typed (TypedDict) and passed between nodes as a dict.
"""

from __future__ import annotations

import logging
import threading
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import classifier, compressor, router
from src.prompts import get_output_guidance, get_system_prompt
from src.token_counter import GLOBAL_TRACKER, TokenBudget, count_tokens

logger = logging.getLogger(__name__)

# -- State definition ----------------------------------------------------------

class AgentState(TypedDict):
    # Input
    task_id: str
    original_prompt: str

    # Classification
    category: str
    difficulty: float

    # Processing
    compressed_prompt: str
    system_prompt: str
    current_tier: int           # 1=local (free), 3=Fireworks
    fireworks_models: list[str]
    answer: str
    confidence: float
    cached: bool

    # Token tracking
    fireworks_input_tokens: int
    fireworks_output_tokens: int
    local_tokens: int

    # Routing
    fireworks_model: str

    # Budget
    budget: dict                # Serialized TokenBudget state

    # Error
    error: str


# -- Categories where local 3B is reliable enough to skip Fireworks ------------
# Code / logic / math require stronger reasoning -> always go to Fireworks.
_LOCAL_CAPABLE_CATEGORIES = {"factual", "sentiment", "summarization", "ner"}

# Difficulty threshold above which we skip local and go straight to Fireworks.
# Keep this LOW (0.5) so only very easy tasks try local inference.
# On 2 vCPU, local inference is ~3-5 tok/sec - only worth it for simple tasks.
_LOCAL_DIFFICULTY_CUTOFF = 0.5

# Max tokens for local inference - keeps each call under ~30s on 2 vCPU.
# Short answers (sentiment, factual, NER) don't need long outputs anyway.
_LOCAL_MAX_TOKENS = 96


# -- Node implementations ------------------------------------------------------

def classify_node(state: AgentState) -> AgentState:
    """Classify task category and difficulty. Zero API cost."""
    prompt = state["original_prompt"]
    cat, diff = classifier.classify(prompt)
    sys_p = get_system_prompt(cat)
    guidance = get_output_guidance(cat)

    logger.info("[%s] classify -> %s (diff=%.2f)", state["task_id"], cat, diff)

    return {
        **state,
        "category": cat,
        "difficulty": diff,
        "system_prompt": sys_p,
        # Append output guidance to prompt (not system) to save system tokens
        "compressed_prompt": f"{prompt}\n\n{guidance}" if guidance else prompt,
        "answer": "",
        "confidence": 0.0,
        "cached": False,
        "current_tier": 1,
        "fireworks_models": [],
        "fireworks_input_tokens": 0,
        "fireworks_output_tokens": 0,
        "local_tokens": 0,
        "fireworks_model": "",
        "error": "",
    }


def local_infer_node(state: AgentState) -> AgentState:
    """
    Run local Qwen2.5-3B inference (zero token cost).

    For categories where the 3B model is unlikely to be accurate (code, logic,
    hard math) we skip straight to Fireworks. For easy categories we try local
    first and only escalate if the validator rejects the answer.
    """
    from src import local_model, validator

    category   = state["category"]
    difficulty = state["difficulty"]

    # Hard categories or high difficulty: skip local, go straight to Fireworks
    skip_local = (
        category not in _LOCAL_CAPABLE_CATEGORIES
        or difficulty >= _LOCAL_DIFFICULTY_CUTOFF
        or not local_model.is_available()
    )

    if skip_local:
        logger.info(
            "[%s] Skipping local inference (cat=%s diff=%.2f) -> Fireworks",
            state["task_id"], category, difficulty,
        )
        return {**state, "current_tier": 3, "confidence": 0.0}

    logger.info("[%s] Running local inference (tier 1, max_tokens=%d)...", state["task_id"], _LOCAL_MAX_TOKENS)

    answer = local_model.infer(
        system_prompt=state["system_prompt"],
        user_prompt=state["compressed_prompt"],
        max_tokens=_LOCAL_MAX_TOKENS,
        temperature=0.05,
    )

    if not answer:
        logger.warning("[%s] Local model returned empty answer", state["task_id"])
        return {**state, "current_tier": 3, "confidence": 0.0}

    confidence, passed = validator.validate(answer, category)
    logger.info(
        "[%s] Local answer: conf=%.2f pass=%s",
        state["task_id"], confidence, passed,
    )

    return {
        **state,
        "answer": answer,
        "confidence": confidence,
        "current_tier": 1 if passed else 3,
    }


def route_after_local(state: AgentState) -> str:
    """
    Conditional edge after local_infer_node.
      - Local passed validation -> END  (free answer, zero Fireworks tokens)
      - Otherwise              -> compress_prompt -> fireworks_call
    """
    if state["current_tier"] == 1 and state.get("answer", ""):
        logger.info("[%s] Local answer accepted - skipping Fireworks", state["task_id"])
        return "end"
    return "compress_prompt"


def compress_prompt_node(state: AgentState) -> AgentState:
    """
    Compress prompt before Fireworks call and select the Fireworks model.
    """
    category   = state["category"]
    difficulty = state["difficulty"]

    # Select model(s)
    fw_models = router.get_capable_models(category, difficulty)
    if not fw_models:
        logger.error("[%s] No Fireworks model available!", state["task_id"])
        return {**state, "fireworks_models": [], "error": "no_fw_model"}

    # Determine token budget for this model (using first as representative)
    budget = TokenBudget(model_key=fw_models[0])
    budget.set_system_prompt(state["system_prompt"])

    # Compress prompt (non-aggressive to preserve accuracy)
    compressed = compressor.compress_prompt(
        prompt=state["compressed_prompt"],
        category=category,
        available_tokens=budget.available,
        aggressive=False,
    )

    saved = count_tokens(state["compressed_prompt"]) - count_tokens(compressed)
    if saved > 0:
        logger.info("[%s] Compressed prompt: saved %d tokens", state["task_id"], saved)

    return {
        **state,
        "compressed_prompt": compressed,
        "fireworks_models": fw_models,
        "current_tier": 3,
    }


def fireworks_call_node(state: AgentState) -> AgentState:
    """Call Fireworks API with compressed prompt, using dynamic model fallback."""
    model_ids = state.get("fireworks_models", [])
    if not model_ids:
        return {**state, "answer": state.get("answer", ""), "error": "no_fw_models"}

    # Tune max output tokens per category - avoids paying for unused token budget
    category = state["category"]
    if category in ("code_gen", "code_debug"):
        max_out = 2048
    elif category == "logic":
        max_out = 1024
    elif category == "math":
        max_out = 1024
    elif category in ("sentiment", "ner"):
        max_out = 256
    else:
        max_out = 1024

    answer, in_tok, out_tok = router.call_fireworks(
        model_ids=model_ids,
        system_prompt=state["system_prompt"],
        user_prompt=state["compressed_prompt"],
        max_tokens=max_out,
        temperature=0.05,
    )

    # Track scored tokens
    GLOBAL_TRACKER.record_fireworks(in_tok, out_tok)
    logger.info(
        "[%s] Fireworks done: in=%d out=%d total=%d",
        state["task_id"], in_tok, out_tok, in_tok + out_tok,
    )

    return {
        **state,
        "answer": answer or state.get("answer", ""),
        "fireworks_input_tokens": state["fireworks_input_tokens"] + in_tok,
        "fireworks_output_tokens": state["fireworks_output_tokens"] + out_tok,
    }


# -- Build the graph -----------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph agent.

    Flow:
      START -> classify -> local_infer
                              ├-- pass  -> END          (free, no Fireworks)
                              └-- fail  -> compress_prompt -> fireworks_call -> END
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("classify",        classify_node)
    builder.add_node("local_infer",     local_infer_node)
    builder.add_node("compress_prompt", compress_prompt_node)
    builder.add_node("fireworks_call",  fireworks_call_node)

    # Edges
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "local_infer")

    # Conditional: local passed -> END, else -> compress+fireworks
    builder.add_conditional_edges(
        "local_infer",
        route_after_local,
        {"end": END, "compress_prompt": "compress_prompt"},
    )
    builder.add_edge("compress_prompt", "fireworks_call")
    builder.add_edge("fireworks_call",  END)

    return builder.compile()


# -- Compiled graph singleton --------------------------------------------------
_graph = None
_graph_lock = threading.Lock()

def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:   # double-checked locking
                _graph = build_graph()
                logger.info("LangGraph compiled successfully")
    return _graph


def run_task(task_id: str, prompt: str) -> dict:
    """
    Run a single task through the full agent pipeline.

    Returns
    -------
    dict with keys: task_id, answer, _meta (token stats)
    """
    graph = get_graph()

    initial_state: AgentState = {
        "task_id": task_id,
        "original_prompt": prompt,
        "category": "",
        "difficulty": 0.0,
        "compressed_prompt": prompt,
        "system_prompt": "",
        "current_tier": 0,
        "answer": "",
        "confidence": 0.0,
        "cached": False,
        "fireworks_models": [],
        "fireworks_input_tokens": 0,
        "fireworks_output_tokens": 0,
        "local_tokens": 0,
        "fireworks_model": "",
        "budget": {},
        "error": "",
    }

    try:
        final_state = graph.invoke(initial_state)
        GLOBAL_TRACKER.task_count += 1

        answer = final_state.get("answer", "").strip()
        if not answer:
            logger.warning("[%s] Empty answer after full pipeline!", task_id)
            answer = "Unable to process this task."

        return {
            "task_id": task_id,
            "answer": answer,
            "_meta": {
                "category": final_state.get("category"),
                "difficulty": final_state.get("difficulty"),
                "tier": final_state.get("current_tier"),
                "cached": final_state.get("cached"),
                "fw_in": final_state.get("fireworks_input_tokens", 0),
                "fw_out": final_state.get("fireworks_output_tokens", 0),
                "fw_model": final_state.get("fireworks_model"),
                "confidence": final_state.get("confidence"),
            },
        }

    except Exception as e:
        logger.error("[%s] Pipeline error: %s", task_id, e, exc_info=True)
        GLOBAL_TRACKER.task_count += 1
        return {
            "task_id": task_id,
            "answer": "Error processing task.",
            "_meta": {"error": str(e)},
        }
