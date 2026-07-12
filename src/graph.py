"""
LangGraph agent graph — central orchestrator for Track 1.

Graph flow:
  START → classify → local_infer
                        ├── PASS  → END          (free, zero Fireworks tokens)
                        └── FAIL  → compress_prompt → fireworks_call → END

Local inference (HF Qwen2.5-0.5B) is attempted first for categories where a
small model is reliable (factual, sentiment, summarization, NER).
Hard categories (code, logic, math) skip straight to Fireworks.

Token budget is SCORED: every Fireworks input + output token costs rank points.
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

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

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
    output_guidance: str       # stored for compressor dedup pass
    current_tier: int          # 1=local (free), 3=Fireworks
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
    budget: dict

    # Error
    error: str


# ---------------------------------------------------------------------------
# Routing config
# ---------------------------------------------------------------------------

# Categories where small local model is reliable (Disabled for maximum accuracy)
_LOCAL_CAPABLE_CATEGORIES = {"factual", "sentiment", "summarization", "ner", "logic", "math", "code_debug"}

# Skip local inference for anything harder than this
_LOCAL_DIFFICULTY_CUTOFF = 0.7

# Max new tokens for local model (keeps latency under ~20s on 2 vCPU)
_LOCAL_MAX_TOKENS = 512

# Per-category Fireworks max_tokens — brutally capped to achieve <1000 token limit
_CATEGORY_MAX_OUTPUT: dict[str, int] = {
    "factual":       50,
    "math":          150,
    "sentiment":     20,
    "summarization": 150,
    "ner":           100,
    "code_debug":   250,
    "logic":         150,
    "code_gen":     300,
}


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> AgentState:
    """Classify task category and difficulty. Zero API cost."""
    prompt = state["original_prompt"]
    cat, diff = classifier.classify(prompt)
    sys_p = get_system_prompt(cat)
    guidance = get_output_guidance(cat)

    logger.info("[%s] classify → %s (diff=%.2f)", state["task_id"], cat, diff)

    # Append output guidance to prompt ONLY if non-empty
    if guidance:
        user_prompt = f"{prompt}\n{guidance}"
    else:
        user_prompt = prompt

    return {
        **state,
        "category":               cat,
        "difficulty":             diff,
        "system_prompt":          sys_p,
        "output_guidance":        guidance,
        "compressed_prompt":      user_prompt,
        "answer":                 "",
        "confidence":             0.0,
        "cached":                 False,
        "current_tier":           1,
        "fireworks_models":       [],
        "fireworks_input_tokens": 0,
        "fireworks_output_tokens": 0,
        "local_tokens":           0,
        "fireworks_model":        "",
        "error":                  "",
    }


def local_infer_node(state: AgentState) -> AgentState:
    """
    Attempt local Qwen2.5-0.5B inference (zero token cost).

    Skips for: code, logic, math (need stronger reasoning).
    Skips for: high-difficulty tasks (small model unreliable).
    """
    from src import local_model, validator

    category   = state["category"]
    difficulty = state["difficulty"]

    # We only send basic tasks to the local 0.5B model.
    # We lowered the difficulty cutoff to 0.4 so we don't risk accuracy on medium-hard tasks.
    _SAFE_LOCAL_CATEGORIES = {"factual", "sentiment", "summarization", "ner"}
    skip_local = (
        category not in _SAFE_LOCAL_CATEGORIES
        or difficulty >= 0.4
        or not local_model.is_available()
    )

    if skip_local:
        logger.info(
            "[%s] Skipping local (cat=%s diff=%.2f avail=%s) → Fireworks",
            state["task_id"], category, difficulty, local_model.is_available(),
        )
        return {**state, "current_tier": 3, "confidence": 0.0}

    logger.info("[%s] Local inference (tier 1, max=%d tok)…", state["task_id"], _LOCAL_MAX_TOKENS)

    answer = local_model.infer(
        system_prompt=state["system_prompt"],
        user_prompt=state["compressed_prompt"],
        max_tokens=_LOCAL_MAX_TOKENS,
        temperature=0.05,
    )

    if not answer:
        logger.warning("[%s] Local model returned empty answer", state["task_id"])
        return {**state, "current_tier": 3, "confidence": 0.0}

    # Record local tokens (free, but good for metrics)
    in_tok = count_tokens(state["compressed_prompt"])
    out_tok = count_tokens(answer)
    GLOBAL_TRACKER.record_local(in_tok + out_tok)

    confidence, passed = validator.validate(answer, category)
    logger.info("[%s] Local: conf=%.2f pass=%s", state["task_id"], confidence, passed)

    return {
        **state,
        "answer":       answer,
        "confidence":   confidence,
        "current_tier": 1 if passed else 3,
    }


def route_after_local(state: AgentState) -> str:
    """
    Conditional edge after local_infer_node.
      - Local passed → END  (free answer, zero Fireworks tokens)
      - Otherwise    → compress_prompt → fireworks_call
    """
    if state["current_tier"] == 1 and state.get("answer", ""):
        logger.info("[%s] Local answer accepted — skipping Fireworks", state["task_id"])
        return "end"
    return "compress_prompt"


def compress_prompt_node(state: AgentState) -> AgentState:
    """Compress prompt and select Fireworks model before API call."""
    category   = state["category"]
    difficulty = state["difficulty"]

    # Select capable models (ordered: smallest first for efficiency)
    fw_models = router.get_capable_models(category, difficulty)
    if not fw_models:
        logger.error("[%s] No Fireworks model available!", state["task_id"])
        return {**state, "fireworks_models": [], "error": "no_fw_model"}

    # Determine token budget using first (smallest) model as representative
    budget = TokenBudget(model_key=fw_models[0])
    budget.set_system_prompt(state["system_prompt"])

    before_tokens = count_tokens(state["compressed_prompt"])

    # Compress — pass output_guidance for deduplication
    compressed = compressor.compress_prompt(
        prompt=state["compressed_prompt"],
        category=category,
        available_tokens=budget.available,
        aggressive=True,
        guidance_text=state.get("output_guidance", ""),
    )

    after_tokens = count_tokens(compressed)
    saved = before_tokens - after_tokens
    if saved > 0:
        logger.info(
            "[%s] Compressed: %d → %d tokens (saved %d, %.0f%%)",
            state["task_id"], before_tokens, after_tokens, saved,
            100 * saved / max(before_tokens, 1),
        )

    return {
        **state,
        "compressed_prompt": compressed,
        "fireworks_models":  fw_models,
        "current_tier":      3,
    }


def fireworks_call_node(state: AgentState) -> AgentState:
    """Call Fireworks API with compressed prompt, using dynamic model fallback."""
    model_ids = state.get("fireworks_models", [])
    if not model_ids:
        logger.error(
            "[%s] fireworks_call_node reached with no models — no_fw_models error",
            state["task_id"],
        )
        return {**state, "error": "no_fw_models"}

    category = state["category"]
    max_out  = _CATEGORY_MAX_OUTPUT.get(category, 512)

    logger.info(
        "[%s] Fireworks call: cat=%s model=%s max_out=%d in_tokens=%d",
        state["task_id"], category,
        model_ids[0].split("/")[-1], max_out,
        count_tokens(state["compressed_prompt"]),
    )

    answer, in_tok, out_tok = router.call_fireworks(
        model_ids=model_ids,
        system_prompt=state["system_prompt"],
        user_prompt=state["compressed_prompt"],
        max_tokens=max_out,
        temperature=0.05,
    )

    # Track scored tokens
    GLOBAL_TRACKER.record_fireworks(in_tok, out_tok)

    if not answer:
        logger.critical(
            "[%s] Fireworks returned EMPTY/None answer! in_tok=%d out_tok=%d — "
            "check FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and ALLOWED_MODELS.",
            state["task_id"], in_tok, out_tok,
        )
        # Preserve any existing local answer rather than overwriting with ""
        existing = state.get("answer", "")
        return {
            **state,
            "answer":                 existing,
            "fireworks_input_tokens":  state["fireworks_input_tokens"] + in_tok,
            "fireworks_output_tokens": state["fireworks_output_tokens"] + out_tok,
            "error":                  "fireworks_empty_response",
        }

    logger.info(
        "[%s] Fireworks OK: in=%d out=%d total=%d",
        state["task_id"], in_tok, out_tok, in_tok + out_tok,
    )

    return {
        **state,
        "answer":                 answer.strip(),
        "fireworks_input_tokens":  state["fireworks_input_tokens"] + in_tok,
        "fireworks_output_tokens": state["fireworks_output_tokens"] + out_tok,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Compile the LangGraph agent.

    Flow:
      START → classify → local_infer
                              ├── pass  → END
                              └── fail  → compress_prompt → fireworks_call → END
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify",        classify_node)
    builder.add_node("local_infer",     local_infer_node)
    builder.add_node("compress_prompt", compress_prompt_node)
    builder.add_node("fireworks_call",  fireworks_call_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "local_infer")

    builder.add_conditional_edges(
        "local_infer",
        route_after_local,
        {"end": END, "compress_prompt": "compress_prompt"},
    )
    builder.add_edge("compress_prompt", "fireworks_call")
    builder.add_edge("fireworks_call",  END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Compiled graph singleton
# ---------------------------------------------------------------------------
_graph      = None
_graph_lock = threading.Lock()
_IN_MEMORY_CACHE: dict[str, dict] = {}


def get_graph():
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
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
    if prompt in _IN_MEMORY_CACHE:
        logger.info("[%s] 🚀 EXACT MATCH CACHE HIT! 0 tokens used.", task_id)
        cached_meta = _IN_MEMORY_CACHE[prompt]["_meta"].copy()
        cached_meta["cached"] = True
        return {
            "task_id": task_id,
            "answer":  _IN_MEMORY_CACHE[prompt]["answer"],
            "_meta":   cached_meta,
        }

    graph = get_graph()

    initial_state: AgentState = {
        "task_id":                task_id,
        "original_prompt":        prompt,
        "category":               "",
        "difficulty":             0.0,
        "compressed_prompt":      prompt,
        "system_prompt":          "",
        "output_guidance":        "",
        "current_tier":           0,
        "answer":                 "",
        "confidence":             0.0,
        "cached":                 False,
        "fireworks_models":       [],
        "fireworks_input_tokens": 0,
        "fireworks_output_tokens": 0,
        "local_tokens":           0,
        "fireworks_model":        "",
        "budget":                 {},
        "error":                  "",
    }

    try:
        final_state = graph.invoke(initial_state)
        GLOBAL_TRACKER.task_count += 1

        answer = final_state.get("answer", "").strip()
        error  = final_state.get("error", "")

        if not answer:
            # Loud warning so this shows up in harness logs
            logger.error(
                "[%s] EMPTY ANSWER after full pipeline! error=%s tier=%s — "
                "Fireworks call likely failed. Check env vars.",
                task_id,
                error or "none",
                final_state.get("current_tier"),
            )
            # Return a descriptive fallback (grading will mark wrong, but at least
            # results.json is valid and non-empty)
            answer = f"[pipeline error: {error or 'empty_response'}]"

        result = {
            "task_id": task_id,
            "answer":  answer,
            "_meta": {
                "category":   final_state.get("category"),
                "difficulty": final_state.get("difficulty"),
                "tier":       final_state.get("current_tier"),
                "cached":     final_state.get("cached"),
                "fw_in":      final_state.get("fireworks_input_tokens", 0),
                "fw_out":     final_state.get("fireworks_output_tokens", 0),
                "fw_model":   final_state.get("fireworks_model"),
                "confidence": final_state.get("confidence"),
                "error":      error,
            },
        }
        
        # Save to cache if successful
        if answer and not error:
            _IN_MEMORY_CACHE[prompt] = result.copy()

        return result

    except Exception as exc:
        logger.error("[%s] Pipeline exception: %s", task_id, exc, exc_info=True)
        GLOBAL_TRACKER.task_count += 1
        return {
            "task_id": task_id,
            "answer":  f"[pipeline exception: {type(exc).__name__}]",
            "_meta":   {"error": str(exc)},
        }
