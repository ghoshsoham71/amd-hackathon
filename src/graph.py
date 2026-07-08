"""
LangGraph agent graph — the central orchestrator for Track 1.

Graph flow:
  START
    │
    ▼
  classify ──────────────────────────────────────────────────────────────┐
    │                                                                     │
    ▼                                                                     │
  check_cache ──HIT──────────────────────────────────────────────────► END
    │ MISS
    ▼
  local_l1 (Qwen2.5-1.5B)
    │
    ▼
  validate_l1
    ├── PASS ──► cache_store ──────────────────────────────────────────► END
    └── FAIL ──► local_l2 (Qwen2.5-3B)
                   │
                   ▼
                 validate_l2
                   ├── PASS ──► cache_store ───────────────────────────► END
                   └── FAIL ──► compress_prompt ──► fireworks_call
                                                       │
                                                       ▼
                                                     cache_store ──────► END

State is fully typed (TypedDict) and passed between nodes as a dict.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src import classifier, compressor, router
from src.prompts import get_output_guidance, get_system_prompt
from src.token_counter import GLOBAL_TRACKER, TokenBudget, count_tokens

logger = logging.getLogger(__name__)

# ── State definition ──────────────────────────────────────────────────────────

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
    current_tier: int           # 1=L1 local, 2=L2 local, 3=Fireworks
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


# ── Node implementations ──────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    """Classify task category and difficulty. Zero API cost."""
    prompt = state["original_prompt"]
    cat, diff = classifier.classify(prompt)
    sys_p = get_system_prompt(cat)
    guidance = get_output_guidance(cat)

    logger.info("[%s] classify → %s (diff=%.2f)", state["task_id"], cat, diff)

    return {
        **state,
        "category": cat,
        "difficulty": diff,
        "system_prompt": sys_p,
        # Append output guidance to prompt (not system) to save sys tokens
        "compressed_prompt": f"{prompt}\n\n{guidance}" if guidance else prompt,
        "answer": "",
        "confidence": 0.0,
        "cached": False,
        "current_tier": 3,
        "fireworks_models": [],
        "fireworks_input_tokens": 0,
        "fireworks_output_tokens": 0,
        "local_tokens": 0,
        "fireworks_model": "",
        "error": "",
    }


def compress_prompt_node(state: AgentState) -> AgentState:
    """
    Compress prompt before Fireworks call.
    Also select the Fireworks model.
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
    """Call Fireworks API with compressed prompt, using dynamic fallback."""
    model_ids = state.get("fireworks_models", [])
    if not model_ids:
        return {**state, "answer": state.get("answer", ""), "error": "no_fw_models"}

    # Give models enough room to finish their thoughts
    max_out = 512
    if state["category"] in ("code_gen", "code_debug"):
        max_out = 1024
    elif state["category"] == "logic":
        max_out = 800

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




# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph agent."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("classify",        classify_node)
    builder.add_node("compress_prompt", compress_prompt_node)
    builder.add_node("fireworks_call",  fireworks_call_node)

    # Entry
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "compress_prompt")
    builder.add_edge("compress_prompt", "fireworks_call")
    builder.add_edge("fireworks_call",  END)

    return builder.compile()


# ── Compiled graph singleton ──────────────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph compiled successfully")
    return _graph


def run_task(task_id: str, prompt: str) -> dict:
    """
    Run a single task through the full agent pipeline.

    Returns
    -------
    dict with keys: task_id, answer, meta (token stats)
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
