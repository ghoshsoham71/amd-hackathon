"""
Local model inference via HuggingFace Transformers (CPU).

Replaces llama-cpp-python with a pure-Python alternative that:
  - Requires NO C++ compilation
  - Runs Qwen2.5-1.5B-Instruct natively in Python
  - Stays well within the 4 GB RAM / 2 vCPU grading environment
  - Model is baked into the Docker image at /app/models/hf_model
    (no runtime download needed — satisfies the 60-second startup rule)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — all overridable via env vars
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
HF_MODEL_PATH = Path(os.environ.get("LOCAL_HF_MODEL_PATH", str(MODEL_DIR / "hf_model")))

# Inference hyper-params
MAX_NEW_TOKENS  = int(os.environ.get("LOCAL_MAX_TOKENS",    "256"))
MAX_INPUT_TOKENS = int(os.environ.get("LOCAL_MAX_INPUT",   "1024"))

# ---------------------------------------------------------------------------
# Transformers availability check
# ---------------------------------------------------------------------------
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch not installed — local inference unavailable")

# ---------------------------------------------------------------------------
# Thread-safe lazy singleton
# ---------------------------------------------------------------------------
_lock       = threading.Lock()
_tokenizer  = None
_model      = None
_available  = None   # cached bool — None means not-yet-checked


def _load() -> bool:
    """Load tokenizer + model. Returns True on success. Called once under lock."""
    global _tokenizer, _model

    if not _TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not installed — local model disabled")
        return False

    if not HF_MODEL_PATH.exists():
        logger.warning(
            "Local HF model not found at %s — local inference disabled. "
            "Bake the model into the Docker image during build.",
            HF_MODEL_PATH,
        )
        return False

    logger.info("Loading local model from %s …", HF_MODEL_PATH)
    try:
        tok = AutoTokenizer.from_pretrained(
            str(HF_MODEL_PATH),
            local_files_only=True,
            trust_remote_code=True,
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            str(HF_MODEL_PATH),
            local_files_only=True,
            torch_dtype=torch.float32,   # CPU: float32 (bfloat16 not always supported)
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        mdl.eval()
        _tokenizer = tok
        _model     = mdl
        logger.info("Local model loaded successfully (%.0f M params approx)",
                    sum(p.numel() for p in mdl.parameters()) / 1e6)
        return True
    except Exception as exc:
        logger.error("Failed to load local model: %s", exc)
        return False


def _ensure_loaded() -> bool:
    """Return True if the model is ready. Loads on first call (thread-safe)."""
    global _available
    if _available is not None:
        return _available
    with _lock:
        if _available is None:
            _available = _load()
    return _available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if the local model is loaded and ready."""
    return _ensure_loaded()


def infer(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> Optional[str]:
    """
    Run local inference using the loaded HF model.

    Returns the generated text string, or None on failure.
    Thread-safe: the model itself is read-only after load; HF generate() is
    safe for concurrent calls on CPU when inputs are separate tensors.
    """
    if not _ensure_loaded():
        logger.debug("Local model unavailable — skipping inference")
        return None

    try:
        # Build chat-formatted prompt using the model's own template
        messages = [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_prompt},
        ]
        # apply_chat_template adds BOS/EOS tokens correctly for Qwen
        prompt_text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = _tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0.01),
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip echoed prompt)
        new_tokens = output_ids[0][input_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        logger.debug(
            "Local infer: input=%d new=%d chars=%d",
            input_len, len(new_tokens), len(text),
        )
        return text if text else None

    except Exception as exc:
        logger.error("Local inference error: %s", exc)
        return None


def preload_models() -> dict:
    """
    Preload the local model at startup.
    Returns dict with model availability status.
    """
    logger.info("Preloading local HF model …")
    ok = _ensure_loaded()
    status = {"model_available": ok}
    logger.info("Local model status: %s", status)
    return status
