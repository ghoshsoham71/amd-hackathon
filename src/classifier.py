"""
Task classifier - identifies category and difficulty score for each task.

Approach (zero Fireworks cost):
1. Fast regex/keyword pass (handles ~85% of tasks correctly)
2. For ambiguous cases: run local L1 model with a 1-shot classification prompt
3. Return: (category, difficulty_score 0.0-1.0)

Categories (matching Track 1 spec):
  1 factual        - explaining concepts, definitions
  2 math           - arithmetic, word problems, projections
  3 sentiment      - labelling + justification
  4 summarization  - condensing passages
  5 ner            - extracting named entities
  6 code_debug     - finding/fixing bugs
  7 logic          - constraint-based puzzles
  8 code_gen       - writing functions from spec
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

from src import local_model
from src.prompts import (
    CATEGORY_KEYWORDS,
    CATEGORY_ORDER,
    DIFFICULTY_SIGNALS,
)

logger = logging.getLogger(__name__)


# -- Valid category keys - defined at module level so classify() can reference it
CATEGORY_NAMES: dict[str, str] = {
    "factual":       "1-Factual Knowledge",
    "math":          "2-Mathematical Reasoning",
    "sentiment":     "3-Sentiment Classification",
    "summarization": "4-Text Summarisation",
    "ner":           "5-Named Entity Recognition",
    "code_debug":    "6-Code Debugging",
    "logic":         "7-Logical/Deductive Reasoning",
    "code_gen":      "8-Code Generation",
}


# -- Regex patterns for high-confidence classification -------------------------
_PATTERNS: list[Tuple[str, re.Pattern]] = [
    ("math", re.compile(
        r"\b(calculat|comput|how many|total|sum of|percentage|multiply|divide|"
        r"arith|equat|remain|profit|loss|interest|averag|mean\b|median|"
        r"probab|fraction|\d+\s*[\+\-\*\/]\s*\d+|\d+%)\b",
        re.I,
    )),
    ("sentiment", re.compile(
        r"\b(sentiment|classify the sentiment|positive|negative|emotion|"
        r"opinion of|feeling|tone of|attitude|review of)\b",
        re.I,
    )),
    ("summarization", re.compile(
        r"\b(summari[zs]e|summary|condense|tldr|brief(ly)?|"
        r"in (one|1|two|2) sentence|key points|main idea|shorten)\b",
        re.I,
    )),
    ("ner", re.compile(
        r"\b(named entit|extract entit|identif.{0,10}entit|"
        r"person.{0,20}org|entity recogn|ner\b)\b",
        re.I,
    )),
    ("code_debug", re.compile(
        r"\b(bug|debug|fix (this|the)|find the error|has a bug|"
        r"broken|incorrect output|this function should|exception|traceback)\b",
        re.I,
    )),
    ("logic", re.compile(
        r"\b(puzzle|constraint|who owns|which person|logical|deductive|"
        r"neither|but not|exactly one|each (own|have)|different pet|"
        r"three friends|five people)\b",
        re.I,
    )),
    ("code_gen", re.compile(
        r"\b(write (a |the )?function|implement (a |the )?|create (a |the )?|"
        r"code that|python function|write code|generate code|"
        r"program that|function that returns)\b",
        re.I,
    )),
]


def _keyword_score(prompt_lower: str, category: str) -> float:
    """Count keyword matches (0–1 normalized)."""
    keywords = CATEGORY_KEYWORDS.get(category, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in prompt_lower)
    return min(hits / max(len(keywords) * 0.2, 1), 1.0)


def _difficulty_score(prompt: str, category: str) -> float:
    """
    Estimate difficulty 0-1 based on:
    - Signal keywords (hard/easy)
    - Prompt length (longer -> harder)
    - Code block presence
    - Number of constraints
    """
    lower = prompt.lower()
    signals = DIFFICULTY_SIGNALS.get(category, {})

    high_hits = sum(1 for s in signals.get("high", []) if s in lower)
    low_hits  = sum(1 for s in signals.get("low",  []) if s in lower)

    # Keyword signal [-1, 1]
    signal = (high_hits - low_hits) / max(high_hits + low_hits + 1, 1)

    # Length signal [0, 1]
    length_signal = min(len(prompt) / 1200, 1.0)

    # Code block presence
    has_code = 1.0 if re.search(r"```|def |class |import ", prompt) else 0.0

    # Weighted composite
    difficulty = (0.4 * ((signal + 1) / 2)) + (0.3 * length_signal) + (0.3 * has_code)
    return round(min(max(difficulty, 0.0), 1.0), 3)


def classify(prompt: str) -> Tuple[str, float]:
    """
    Classify a task prompt.

    Returns
    -------
    category : str
        One of the 8 category keys.
    difficulty : float
        0.0 (trivial) -> 1.0 (very hard).
    """
    lower = prompt.lower()

    # -- Phase 1: Regex fast-path ----------------------------------------------
    best_cat: str | None = None
    best_matches = 0

    for cat, pattern in _PATTERNS:
        matches = len(pattern.findall(prompt))
        if matches > best_matches:
            best_matches = matches
            best_cat = cat

    if best_cat and best_matches >= 1:
        difficulty = _difficulty_score(prompt, best_cat)
        logger.debug("classify[regex] %s difficulty=%.2f", best_cat, difficulty)
        return best_cat, difficulty

    # -- Phase 2: Keyword scoring (ordered by specificity) --------------------
    scores: dict[str, float] = {}
    for cat in CATEGORY_ORDER:
        scores[cat] = _keyword_score(lower, cat)

    # Pick highest, break ties with CATEGORY_ORDER priority
    best_cat = max(CATEGORY_ORDER, key=lambda c: scores[c])
    difficulty = _difficulty_score(prompt, best_cat)

    # -- Phase 3: Local LLM Fallback (if low keyword confidence) --------------
    if scores[best_cat] < 0.3 and local_model.is_available():
        system_msg = (
            "You are a text classifier. Classify the user's prompt into exactly ONE "
            "of these categories: factual, math, sentiment, summarization, ner, "
            "code_debug, logic, code_gen. Output ONLY the category name."
        )
        answer = local_model.infer(
            system_prompt=system_msg,
            user_prompt=prompt,
            max_tokens=10,
            temperature=0.0,
        )
        if answer:
            clean_answer = answer.strip().lower()
            # Check against valid category KEYS (e.g. "factual", not "1-Factual Knowledge")
            if clean_answer in CATEGORY_NAMES:
                best_cat = clean_answer
                logger.info("classify[llm] override to %s", best_cat)

    logger.debug(
        "classify[final] %s (score=%.2f) difficulty=%.2f",
        best_cat, scores.get(best_cat, 1.0), difficulty,
    )
    return best_cat, difficulty


def classify_batch(prompts: list[str]) -> list[Tuple[str, float]]:
    """Classify a list of prompts."""
    return [classify(p) for p in prompts]
