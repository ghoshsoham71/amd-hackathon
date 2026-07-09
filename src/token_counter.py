"""
Runtime token budget manager.

Uses tiktoken (cl100k_base) for token counting — close enough to Fireworks
model tokenizers for budgeting purposes. The 5-10% variance is accounted for
by safety buffers.

Key contract: every node that builds a prompt must call `check_budget()` before
sending. If the prompt exceeds the budget, it must call `compress_to_budget()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENCODER = None  # type: ignore[assignment]


def count_tokens(text: str) -> int:
    """Return approximate token count for `text`."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    # Fallback: ~4 chars per token (rough but safe estimate)
    return max(1, len(text) // 4)


def count_tokens_multi(*texts: str) -> int:
    """Sum token counts for multiple text segments."""
    return sum(count_tokens(t) for t in texts if t)


# ── Model context limits ───────────────────────────────────────────────────────
# Conservative limits (actual may be higher, but we stay safe)
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Local models
    "local_l1": 4096,
    "local_l2": 4096,
    # Fireworks — will be overridden by runtime detection
    "default": 4096,
    "8b":  8192,
    "13b": 8192,
    "70b": 8192,
    "mixtral": 32768,
    "mistral": 8192,
}

# Safety buffer — never fill the context completely
SAFETY_BUFFER = 256
# Reserve for output generation
OUTPUT_RESERVE = 512


@dataclass
class TokenBudget:
    """
    Tracks token usage for a single task and enforces per-call budgets.

    Attributes
    ----------
    model_key : str
        Key into MODEL_CONTEXT_LIMITS to determine the context window.
    system_tokens : int
        Tokens consumed by the system prompt (set once).
    used_tokens : int
        Running total of tokens sent in all calls so far.
    fireworks_tokens : int
        Subset of used_tokens that went to Fireworks API (scored).
    """
    model_key: str = "default"
    system_tokens: int = 0
    used_tokens: int = 0
    fireworks_tokens: int = 0
    _limit: int = field(init=False, repr=False)

    def __post_init__(self):
        self._limit = self._resolve_limit()

    def _resolve_limit(self) -> int:
        for key, limit in MODEL_CONTEXT_LIMITS.items():
            if key in self.model_key.lower():
                return limit
        return MODEL_CONTEXT_LIMITS["default"]

    @property
    def available(self) -> int:
        """Tokens available for the user prompt in the next call."""
        return self._limit - self.system_tokens - OUTPUT_RESERVE - SAFETY_BUFFER

    def set_system_prompt(self, system: str) -> int:
        """Record system prompt tokens. Returns token count."""
        self.system_tokens = count_tokens(system)
        return self.system_tokens

    def can_fit(self, prompt: str) -> bool:
        """Return True if `prompt` fits within the available token budget."""
        return count_tokens(prompt) <= self.available

    def record_call(self, prompt: str, response: str, is_fireworks: bool = False):
        """Record tokens from a completed API/local call."""
        call_tokens = count_tokens(prompt) + count_tokens(response)
        self.used_tokens += call_tokens
        if is_fireworks:
            self.fireworks_tokens += call_tokens
        logger.debug(
            "Token record: prompt=%d resp=%d fireworks=%s total_fw=%d",
            count_tokens(prompt), count_tokens(response),
            is_fireworks, self.fireworks_tokens,
        )

    def __repr__(self):
        return (
            f"TokenBudget(limit={self._limit}, available={self.available}, "
            f"used={self.used_tokens}, fireworks={self.fireworks_tokens})"
        )


class GlobalTokenTracker:
    """
    Session-wide token tracker across all tasks.
    Thread-safe via simple int increments (GIL-protected for CPython).
    """

    def __init__(self):
        self.total_fireworks_input: int = 0
        self.total_fireworks_output: int = 0
        self.total_local: int = 0
        self.task_count: int = 0
        self.fireworks_calls: int = 0
        self.local_calls: int = 0

    def record_fireworks(self, input_tokens: int, output_tokens: int):
        self.total_fireworks_input += input_tokens
        self.total_fireworks_output += output_tokens
        self.fireworks_calls += 1

    def record_local(self, tokens: int):
        self.total_local += tokens
        self.local_calls += 1

    @property
    def total_fireworks(self) -> int:
        return self.total_fireworks_input + self.total_fireworks_output

    def summary(self) -> dict:
        return {
            "tasks": self.task_count,
            "fireworks_calls": self.fireworks_calls,
            "fireworks_total_tokens": self.total_fireworks,
            "fireworks_input_tokens": self.total_fireworks_input,
            "fireworks_output_tokens": self.total_fireworks_output,
            "local_calls": self.local_calls,
            "local_tokens": self.total_local,
        }

    def log_summary(self):
        s = self.summary()
        logger.info("=" * 60)
        logger.info("SESSION TOKEN SUMMARY")
        logger.info("  Tasks processed     : %d", s["tasks"])
        logger.info("  Fireworks calls     : %d", s["fireworks_calls"])
        logger.info("  Fireworks tokens    : %d (SCORED)", s["fireworks_total_tokens"])
        logger.info("    ↳ Input tokens    : %d", s["fireworks_input_tokens"])
        logger.info("    ↳ Output tokens   : %d", s["fireworks_output_tokens"])
        logger.info("  Local calls         : %d", s["local_calls"])
        logger.info("  Local tokens        : %d (not scored)", s["local_tokens"])
        logger.info("=" * 60)


# Module-level singleton
GLOBAL_TRACKER = GlobalTokenTracker()
