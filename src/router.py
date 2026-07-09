"""
Fireworks AI router — selects the smallest capable model for each task category.

CRITICAL RULES (from Track 1 spec):
  - ALLOWED_MODELS is read STRICTLY via os.environ["ALLOWED_MODELS"] — no default,
    no fallback. If it's missing the container should fail loudly.
  - Only models in ALLOWED_MODELS are permitted. Calls to any other model ID
    INVALIDATE the submission (MODEL_VIOLATION).
  - All calls MUST go through FIREWORKS_BASE_URL — read from os.environ["FIREWORKS_BASE_URL"].

Strategy:
  1. At first call, parse os.environ["ALLOWED_MODELS"].split(",") — exact model IDs
  2. Rank them by estimated size (from name heuristics) to pick smallest capable
  3. Hard guard in call_fireworks() rejects any model_id not in ALLOWED_MODELS
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── OpenAI client (Fireworks uses OpenAI-compatible API) ─────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("openai package not installed — Fireworks calls disabled")

# ── Non-model config (safe to read with defaults) ─────────────────────────────
MAX_RETRIES     = int(os.environ.get("FW_MAX_RETRIES", "2"))
RETRY_DELAY     = float(os.environ.get("FW_RETRY_DELAY", "1.0"))
REQUEST_TIMEOUT = int(os.environ.get("FW_TIMEOUT", "20"))  # must stay under 30s limit

# ── ALLOWED_MODELS — parsed lazily on first use, never at import time ─────────
# Parsed lazily so the module can be imported in tests without the env var set.
_allowed_models_set: Optional[set[str]] = None
_sorted_models: Optional[List[Tuple[float, str]]] = None


def _parse_allowed_models() -> List[str]:
    """
    Read and parse ALLOWED_MODELS from the environment.

    Uses os.environ["ALLOWED_MODELS"] (strict — raises KeyError if not set,
    exactly as specified in the Track 1 participant guide).

    Returns a list of exact model ID strings.
    """
    raw = os.environ["ALLOWED_MODELS"]          # KeyError if not injected by harness
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("ALLOWED_MODELS is set but contains no valid model IDs")
    return models


def _init_model_registry() -> None:
    """
    Build the allowed model set and sorted ranking on first call.
    Subsequent calls are no-ops (cached).
    """
    global _allowed_models_set, _sorted_models
    if _allowed_models_set is not None:
        return

    try:
        models = _parse_allowed_models()
        _allowed_models_set = set(models)

        ranked: List[Tuple[float, str]] = [
            (_parse_model_size(m), m) for m in models
        ]
        ranked.sort(key=lambda x: x[0])
        _sorted_models = ranked

        logger.info(
            "ALLOWED_MODELS registry: %d models loaded",
            len(models),
        )
        for size, mid in ranked:
            logger.info("  %.0fB  %s", size, mid)
    except (KeyError, ValueError) as e:
        logger.error("Failed to initialize ALLOWED_MODELS registry: %s", e)
        _allowed_models_set = set()
        _sorted_models = []


def is_model_allowed(model_id: str) -> bool:
    """Return True iff model_id is in the ALLOWED_MODELS list."""
    _init_model_registry()
    return model_id in _allowed_models_set   # type: ignore[operator]


# ── Model size estimation ─────────────────────────────────────────────────────

def _parse_model_size(model_id: str) -> float:
    """
    Estimate parameter count (billions) from model name.
    Used only for routing priority — does NOT affect which models are permitted.
    Fallback to 7.0B if no size can be inferred.
    """
    lower = model_id.lower()
    short = lower.split("/")[-1]   # strip accounts/fireworks/models/ prefix

    # ── Generic heuristics ────────────────────────────────────────────────────
    
    # 1. MoE: "8x7b" → treat as ~30B effective
    moe = re.search(r"(\d+)x(\d+)[bB]", short)
    if moe:
        return float(moe.group(1)) * float(moe.group(2)) * 0.6

    # 2. Dense parameters explicitly stated: "7b", "13b", "70b", "0.5b"
    dense = re.search(r"(\d+\.?\d*)[bB](?:[-_]|$)", short)
    if dense:
        return float(dense.group(1))

    # 3. Known generic size aliases
    if "nano"   in short: return 1.0
    if "mini"   in short: return 3.8
    if "small"  in short: return 7.0
    if "medium" in short: return 13.0
    if "large"  in short: return 30.0
    if "xl"     in short: return 70.0

    # 4. Fallback if completely opaque (treat as a mid-size capable model)
    return 7.0


# ── Category → minimum model size (billions)
# Routing preference — always constrained to ALLOWED_MODELS.
# 
# Generalized mapping:
#   < 7B : Factual, text processing
#   ~ 7B : Reasoning, logic
#  >=13B : Code generation
CATEGORY_MIN_PARAMS: Dict[str, float] = {
    "factual":       1.0,
    "sentiment":     1.0,
    "summarization": 1.0,
    "ner":           1.0,
    "math":          7.0,
    "logic":         7.0,
    "code_debug":    7.0,
    "code_gen":      13.0,
}

DIFFICULTY_UPSIZE_THRESHOLD = 0.75   # bump to next tier if difficulty >= this


def get_capable_models(category: str, difficulty: float) -> List[str]:
    """
    Return a list of capable models from ALLOWED_MODELS for this task,
    ordered from smallest (most efficient) to largest (fallback).

    Returns an empty list if the registry is empty.
    """
    _init_model_registry()
    if not _sorted_models:
        return []

    min_params = CATEGORY_MIN_PARAMS.get(category, 7.0)
    if difficulty >= DIFFICULTY_UPSIZE_THRESHOLD:
        min_params *= 1.5

    # Filter and collect all models that meet the minimum size requirement
    capable = [
        model_id for size, model_id in _sorted_models
        if size >= min_params
    ]

    if capable:
        logger.info(
            "router[%s diff=%.2f]: found %d capable models, preferring %s",
            category, difficulty, len(capable), capable[0].split("/")[-1],
        )
        return capable

    # All models are smaller than ideal — fallback to just the largest available
    _, fallback = _sorted_models[-1]
    logger.warning(
        "router[%s]: no model >= %.0fB in ALLOWED_MODELS; falling back to largest: %s",
        category, min_params, fallback.split("/")[-1],
    )
    return [fallback]


# ── Fireworks client ──────────────────────────────────────────────────────────

def _get_client() -> Optional["OpenAI"]:
    """
    Return an OpenAI-compatible client pointed at FIREWORKS_BASE_URL.

    Both API key and base URL are read strictly from the environment —
    the harness injects them; using your own key or URL bypasses token tracking.
    """
    if not _OPENAI_AVAILABLE:
        return None

    api_key  = os.environ["FIREWORKS_API_KEY"]   # KeyError if not injected
    base_url = os.environ["FIREWORKS_BASE_URL"]  # KeyError if not injected

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT,
    )


def call_fireworks(
    model_ids: List[str],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Tuple[Optional[str], int, int]:
    """
    Call Fireworks via the OpenAI-compatible API, falling back dynamically.

    Attempts each model in `model_ids` in order. If a model fails due to 
    rate limits or server errors, it seamlessly switches to the next model.
    """
    if not model_ids:
        return None, 0, 0

    # ── Hard guard: reject any model not in ALLOWED_MODELS ───────────────────
    for m in model_ids:
        if not is_model_allowed(m):
            logger.error(
                "MODEL_VIOLATION PREVENTED: '%s' is not in ALLOWED_MODELS. "
                "Call aborted — would have invalidated submission.", m
            )
            return None, 0, 0

    client = _get_client()
    if client is None:
        return None, 0, 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    for model_id in model_ids:
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                )
                text = response.choices[0].message.content.strip()
                usage = response.usage
                input_tok  = usage.prompt_tokens      if usage else 0
                output_tok = usage.completion_tokens  if usage else 0

                logger.info(
                    "Fireworks[%s]: in=%d out=%d total=%d",
                    model_id.split("/")[-1], input_tok, output_tok, input_tok + output_tok,
                )
                return text, input_tok, output_tok

            except Exception as e:
                err_str = str(e).lower()
                # Don't retry on permanent errors (auth, not found, bad request)
                is_permanent = any(x in err_str for x in (
                    "401", "403", "404", "400", "invalid_api_key",
                    "model_not_found", "not found",
                ))
                if is_permanent:
                    logger.error(
                        "Fireworks %s permanent error (not retrying): %s",
                        model_id.split("/")[-1], e,
                    )
                    break  # skip remaining attempts for this model

                logger.warning(
                    "Fireworks %s transient error (attempt %d/%d): %s. Retrying...",
                    model_id.split("/")[-1], attempt + 1, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

    logger.error("Fireworks call failed across all capable models")
    return None, 0, 0


# ── Utility ───────────────────────────────────────────────────────────────────

def get_available_models() -> List[str]:
    """Return the exact list from ALLOWED_MODELS (in size order)."""
    _init_model_registry()
    return [m for _, m in (_sorted_models or [])]


def log_model_registry() -> None:
    """Log the full ALLOWED_MODELS registry at startup."""
    _init_model_registry()
    logger.info("=== ALLOWED_MODELS (%d) ===", len(_sorted_models or []))
    for size, mid in (_sorted_models or []):
        logger.info("  %.0fB  %s", size, mid)
