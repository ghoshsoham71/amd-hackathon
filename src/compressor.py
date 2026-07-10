"""
Prompt compressor - strips token fat before any Fireworks API call.

Techniques applied (in order):
1. Instruction normalization - replace verbose instruction phrases with compact ones
2. Passage pruning - for summarization/NER tasks, remove low-information sentences
   using TF-IDF scoring
3. Stopword-light trimming - remove filler from context (not from task content)
4. Token budget enforcement - truncate to fit within model context window

Design goal: Compress aggressively enough to save tokens, conservatively enough
to preserve all task-relevant information. Accuracy > token savings.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from src.token_counter import count_tokens
from src import local_model

logger = logging.getLogger(__name__)

# -- Verbose -> compact instruction phrase replacements -------------------------
_INSTRUCTION_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Preambles
    (re.compile(r"please\s+(can\s+you\s+)?", re.I), ""),
    (re.compile(r"i\s+(would|'d)\s+like\s+(you\s+)?to\s+", re.I), ""),
    (re.compile(r"can\s+you\s+please\s+", re.I), ""),
    (re.compile(r"could\s+you\s+(please\s+)?", re.I), ""),
    (re.compile(r"i\s+need\s+you\s+to\s+", re.I), ""),
    # Verbose summarize instructions
    (re.compile(r"summarize\s+the\s+following\s+(text|passage|paragraph)\s+(in|into)", re.I),
     "Summarize"),
    (re.compile(r"summarise\s+the\s+following\s+(text|passage|paragraph)\s+(in|into)", re.I),
     "Summarize"),
    # Verbose NER instructions
    (re.compile(r"extract\s+all\s+named\s+entities\s+(and\s+their\s+types\s+)?from(\s+the\s+following)?:?", re.I),
     "Extract entities from:"),
    # Verbose sentiment instructions
    (re.compile(r"classify\s+the\s+sentiment\s+of\s+(the\s+following\s+)?(text|review|statement|sentence):?", re.I),
     "Classify sentiment:"),
    # Trailing filler
    (re.compile(r"\.\s*please\s+be\s+concise\.?\s*$", re.I), "."),
    (re.compile(r"\.\s*keep\s+(it\s+)?brief\.?\s*$", re.I), "."),
    (re.compile(r"\.\s*do\s+not\s+include\s+unnecessary\s+details?\.?\s*$", re.I), "."),
]


def _normalize_instructions(text: str) -> str:
    """Apply instruction normalization replacements."""
    for pattern, replacement in _INSTRUCTION_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    # Collapse multiple spaces/newlines
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()




# -- Main compress function -----------------------------------------------------

def compress_prompt(
    prompt: str,
    category: str,
    available_tokens: int,
    aggressive: bool = False,
) -> str:
    """
    Compress a prompt to fit within `available_tokens`.

    Parameters
    ----------
    prompt : str
        The original user prompt.
    category : str
        Task category (guides compression strategy).
    available_tokens : int
        Maximum tokens the compressed prompt can occupy.
    aggressive : bool
        If True, apply more aggressive pruning (used for final Fireworks tier).

    Returns
    -------
    str
        Compressed prompt (may equal original if already within budget).
    """
    original_tokens = count_tokens(prompt)

    # Already fits -> no-op
    if original_tokens <= available_tokens:
        return prompt

    logger.info(
        "compress[%s]: %d -> target %d tokens",
        category, original_tokens, available_tokens,
    )

    compressed = prompt

    # -- Step 1: Instruction normalization (always safe) -----------------------
    compressed = _normalize_instructions(compressed)
    if count_tokens(compressed) <= available_tokens:
        return compressed

    # -- Step 2: Use Local LLM for intelligent semantic compression --
    # The local model has 0 token cost, so we use it to rewrite the prompt.
    if local_model.is_available():
        system_msg = (
            "You are an expert prompt compressor. Your job is to rewrite the user's "
            "text to be as concise as possible while keeping all core instructions, "
            "facts, constraints, and data. Do NOT answer the prompt, ONLY compress it. "
            "Output ONLY the compressed text."
        )
        answer = local_model.infer(
            system_prompt=system_msg,
            user_prompt=compressed,
            max_tokens=available_tokens,
            temperature=0.05,
        )
        if answer and len(answer) > 10:
            logger.info("Local LLM compression successful for %s", category)
            # If the LLM successfully shrank it within budget, return it!
            if count_tokens(answer) <= available_tokens:
                return answer
            
            # If it's still slightly too big, we at least start with the compressed version
            # before applying hard truncation
            if count_tokens(answer) < count_tokens(compressed):
                compressed = answer

    # -- Step 3: Hard truncation fallback (last resort) ------------------------

    words = compressed.split()
    truncated: list[str] = []
    current = 0
    for word in words:
        w_tokens = count_tokens(word) + 1
        if current + w_tokens > available_tokens - 5:
            truncated.append("[truncated]")
            break
        truncated.append(word)
        current += w_tokens

    compressed = " ".join(truncated)
    logger.warning(
        "compress[%s]: hard truncated to %d tokens", category, count_tokens(compressed)
    )
    return compressed


def estimate_compression_ratio(prompt: str, category: str) -> float:
    """Estimate how much this prompt can be compressed (0=none, 1=fully compressible)."""
    original = count_tokens(prompt)
    compressed = _normalize_instructions(prompt)
    normalized = count_tokens(compressed)
    ratio = 1.0 - (normalized / max(original, 1))

    # Long passages in content-heavy categories compress well
    if category in ("summarization", "ner", "factual") and original > 300:
        ratio = min(ratio + 0.3, 0.8)

    return round(ratio, 3)
