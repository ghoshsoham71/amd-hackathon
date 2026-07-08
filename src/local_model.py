"""
Local GGUF model inference via llama-cpp-python.

Only ONE model (Qwen2.5-3B-Instruct Q4_K_M) is loaded to stay safely 
within the 4GB RAM budget.

The model uses:
  - n_gpu_layers=0  (no GPU in grading env)
  - use_mmap=True   (memory-map the model)
  - n_ctx=4096      (context window)
  - n_threads=2     (matches 2 vCPU grading env)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))

MODEL_FILENAME = os.environ.get(
    "LOCAL_MODEL_FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf"
)

# ── Inference config ──────────────────────────────────────────────────────────
N_CTX      = int(os.environ.get("LOCAL_N_CTX",     "4096"))
N_THREADS  = int(os.environ.get("LOCAL_N_THREADS", "2"))
MAX_TOKENS = int(os.environ.get("LOCAL_MAX_TOKENS","512"))

# ── Thread locks for lazy loading ─────────────────────────────────────────────
_lock = threading.Lock()
_model = None

# ── Llama.cpp availability ────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    logger.warning("llama-cpp-python not installed — local inference unavailable")


def _load_model(path: Path, tier: str) -> Optional["Llama"]:
    """Load a GGUF model. Returns None if file missing or llama-cpp unavailable."""
    if not _LLAMA_AVAILABLE:
        return None

    if not path.exists():
        logger.warning("%s model not found at %s — local tier disabled", tier, path)
        return None

    logger.info("Loading %s model: %s", tier, path.name)
    try:
        model = Llama(
            model_path=str(path),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=0,       # CPU-only (grading env has no GPU)
            use_mmap=True,        # Memory-map — only loads active pages
            use_mlock=False,      # Don't lock pages (limited RAM)
            verbose=False,
        )
        logger.info("%s model loaded successfully", tier)
        return model
    except Exception as e:
        logger.error("Failed to load %s model: %s", tier, e)
        return None


def _get_model() -> Optional["Llama"]:
    """Return (lazily loaded) model."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            _model = _load_model(MODEL_DIR / MODEL_FILENAME, "Local 3B")
    return _model


def _build_chat_prompt(system: str, user: str, model: "Llama") -> str:
    """
    Build a chat-formatted prompt using the model's built-in tokenizer.
    Falls back to a generic ChatML format if the model doesn't support it.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        # llama-cpp-python ≥0.2 supports apply_chat_template
        return model.tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: generic ChatML
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def infer(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> Optional[str]:
    """
    Run local inference.

    Parameters
    ----------
    system_prompt : str
    user_prompt : str
    max_tokens : int
        Max output tokens.
    temperature : float
        Sampling temperature. Low = more deterministic.

    Returns
    -------
    str | None
        Generated text, or None if the model is unavailable.
    """
    model = _get_model()
    if model is None:
        logger.warning("Local model unavailable — skipping inference")
        return None

    try:
        prompt_text = _build_chat_prompt(system_prompt, user_prompt, model)

        output = model(
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
            echo=False,
        )

        text = output["choices"][0]["text"].strip()
        tokens_used = output["usage"]["total_tokens"]
        logger.debug("Local inference: %d tokens → %d chars", tokens_used, len(text))
        return text

    except Exception as e:
        logger.error("Local inference error: %s", e)
        return None


def preload_models() -> dict:
    """
    Preload the local model at startup.
    Returns dict with model availability status.
    """
    logger.info("Preloading local model...")
    m = _get_model()
    status = {
        "model_available": m is not None,
    }
    logger.info("Model status: %s", status)
    return status


def is_available() -> bool:
    """Check if the local model is available without loading it."""
    path = MODEL_DIR / MODEL_FILENAME
    return _LLAMA_AVAILABLE and path.exists()
