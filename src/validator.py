"""
Answer validator - checks whether a local model's answer meets quality bar.

For each category, apply targeted heuristics:
- Non-empty, non-trivially-short response
- Category-specific structural checks (math: number present, code: parses, etc.)
- Confidence scoring: 0.0 (garbage) -> 1.0 (confident)

PASS threshold: confidence >= 0.65 (configurable via env VALIDATOR_THRESHOLD)
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Tuple

logger = logging.getLogger(__name__)

PASS_THRESHOLD = float(os.environ.get("VALIDATOR_THRESHOLD", "0.65"))

# Answers shorter than this are probably incomplete
MIN_ANSWER_CHARS = {
    "factual":       20,
    "math":          5,
    "sentiment":     15,
    "summarization": 30,
    "ner":           5,
    "code_debug":    20,
    "logic":         10,
    "code_gen":      30,
}

# -- Refusal / failure patterns ------------------------------------------------
_REFUSAL_PATTERNS = re.compile(
    r"\b(i (cannot|can't|am unable|don't know|do not know)|"
    r"i'm not sure|i (am|'m) sorry|as an ai|i apologize)\b",
    re.I,
)

_UNCERTAIN_PATTERNS = re.compile(
    r"\b(i think|i believe|i'm not certain|i'm not 100%|"
    r"i'm not entirely|probably|might be|may be|could be)\b",
    re.I,
)


def _base_confidence(answer: str, category: str) -> float:
    """Compute a base confidence score from general heuristics."""
    if not answer or not answer.strip():
        return 0.0

    answer = answer.strip()

    # Refusal -> fail immediately
    if _REFUSAL_PATTERNS.search(answer):
        return 0.1

    # Length check
    min_len = MIN_ANSWER_CHARS.get(category, 10)
    if len(answer) < min_len:
        return 0.2

    # Repetition detection (model looping)
    words = answer.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # very repetitive
            return 0.15

    # Uncertainty penalty
    conf = 1.0
    if _UNCERTAIN_PATTERNS.search(answer):
        conf -= 0.15

    return conf


def _validate_math(answer: str) -> float:
    """Math: must contain a numeric result."""
    # Look for ANSWER: <number> pattern first
    answer_match = re.search(r"ANSWER:\s*([\d,.\-]+)", answer, re.I)
    if answer_match:
        return 0.95

    # Any number is a good sign
    numbers = re.findall(r"\b\d[\d,.\-]*\b", answer)
    if numbers:
        return 0.80

    return 0.3


def _validate_sentiment(answer: str) -> float:
    """Sentiment: must contain Positive/Negative/Mixed."""
    labels = ["positive", "negative", "mixed", "neutral"]
    lower = answer.lower()
    if any(lbl in lower for lbl in labels):
        return 0.90
    return 0.3


def _validate_ner(answer: str) -> float:
    """NER: should be valid JSON or contain entity brackets."""
    stripped = answer.strip()

    # Try JSON parse
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
            if isinstance(data, list) and all("text" in d for d in data):
                return 0.95
        except json.JSONDecodeError:
            pass

    # Look for entity-like patterns
    if re.search(r"\b(PERSON|ORG|LOC|DATE|ORGANIZATION|LOCATION)\b", answer, re.I):
        return 0.75

    return 0.4


def _validate_code(answer: str, category: str) -> float:
    """Code: extract and attempt to parse Python."""
    # Extract code blocks
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", answer, re.S)
    code = "\n".join(code_blocks) if code_blocks else answer

    # Remove non-code lines for parsing attempt
    code_lines = [ln for ln in code.splitlines()
                  if not ln.strip().startswith("#") or "def " in ln]
    code_str = "\n".join(code_lines)

    try:
        ast.parse(code_str)
        # Valid Python AST
        has_def = "def " in code_str
        if category == "code_gen" and has_def:
            return 0.92
        if category == "code_debug":
            return 0.88
        return 0.80
    except SyntaxError:
        # Might still be useful if it has function-like structure
        if "def " in answer or "return " in answer:
            return 0.55
        return 0.3


def _validate_logic(answer: str) -> float:
    """Logic: should contain a clear conclusion."""
    if re.search(r"\bANSWER:\s*\w+", answer, re.I):
        return 0.92
    # Name-like words, capitalized answers
    if re.search(r"\b(therefore|thus|so |hence|conclusion|answer is)\b", answer, re.I):
        return 0.82
    # At least has a conclusion of some sort
    lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last) < 80 and re.search(r"\b[A-Z][a-z]+\b", last):
            return 0.75
    return 0.5


def _validate_summarization(answer: str) -> float:
    """Summarization: must be non-trivially long but not longer than source."""
    length = len(answer.strip())
    if length < 20:
        return 0.2
    if length > 2000:  # probably not a summary
        return 0.5
    return 0.85


def validate(answer: str, category: str) -> Tuple[float, bool]:
    """
    Validate an answer for a given category.

    Returns
    -------
    confidence : float
        0.0-1.0 quality score.
    passed : bool
        True if confidence >= PASS_THRESHOLD.
    """
    base = _base_confidence(answer, category)
    if base < 0.3:
        return base, False

    # Category-specific checks
    specific: float
    if category == "math":
        specific = _validate_math(answer)
    elif category == "sentiment":
        specific = _validate_sentiment(answer)
    elif category == "ner":
        specific = _validate_ner(answer)
    elif category in ("code_gen", "code_debug"):
        specific = _validate_code(answer, category)
    elif category == "logic":
        specific = _validate_logic(answer)
    elif category == "summarization":
        specific = _validate_summarization(answer)
    else:  # factual
        specific = 0.80 if len(answer.strip()) >= 30 else 0.55

    # Weighted combination
    confidence = round(0.4 * base + 0.6 * specific, 3)
    passed = confidence >= PASS_THRESHOLD

    logger.debug(
        "validate[%s] base=%.2f specific=%.2f -> conf=%.2f pass=%s",
        category, base, specific, confidence, passed,
    )
    return confidence, passed
