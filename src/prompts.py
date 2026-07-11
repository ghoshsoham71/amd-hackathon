"""
Category-specific system prompts — AMD Hackathon Track 1.

Token budget is SCORED by the judging proxy.
Every token in the system prompt costs real score-points.

Design rules:
1. System prompt ≤ 15 tokens each — just enough to set role + output format.
2. ALL detailed output-format instructions go into OUTPUT_GUIDANCE (appended
   to the user message, but kept ultra-short here too).
3. For reasoning tasks (math, logic): enforce "end with ANSWER: <val>" so the
   validator has a reliable extraction hook with the fewest extra tokens.
4. For extractive tasks (NER, sentiment): one-line label enforces format.
5. No conversational filler ("Certainly!", "As an AI...", "I'd be happy to...").
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Core system prompts — ≤ 15 tokens each (verified with tiktoken cl100k_base)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS: Dict[str, str] = {
    # ~10 tokens
    "factual": "Answer directly and concisely. No intro or filler.",

    # ~9 tokens
    "math": "Calculate. Output ONLY the final number. ANSWER: <number>",

    # ~10 tokens
    "sentiment": "End with: SENTIMENT: Positive/Negative/Neutral/Mixed. No other text.",

    # ~10 tokens
    "summarization": "Summarize the text directly and concisely. No intro.",

    # ~12 tokens
    "ner": 'Output ONLY JSON: [{"text":"...","type":"PERSON|ORG|LOC|DATE|OTHER"}]',

    # ~13 tokens
    "code_debug": "Fix bug. Output ONLY python code in ```python block. No text.",

    # ~12 tokens
    "logic": "Reason minimally. End with: ANSWER: <value>",

    # ~11 tokens
    "code_gen": "Output ONLY Python code in ```python block. No tests or comments.",
}

# ---------------------------------------------------------------------------
# Output guidance — appended to USER message (not system).
# Ultra-short reminders only — do NOT repeat the system prompt.
# These are compressed away by compressor.py if they duplicate what's in prompt.
# ---------------------------------------------------------------------------
OUTPUT_GUIDANCE: Dict[str, str] = {
    "factual":       "",                        # No extra guidance needed
    "math":          "End: ANSWER: <number>",
    "sentiment":     "End: SENTIMENT: <Label>",
    "summarization": "",
    "ner":           'JSON only: [{"text":"...","type":"..."}]',
    "code_debug":    "Fix + corrected code.",
    "logic":         "End: ANSWER: <value>",
    "code_gen":      "Python code only.",
}

# ---------------------------------------------------------------------------
# Difficulty estimation keywords per category
# ---------------------------------------------------------------------------
DIFFICULTY_SIGNALS: Dict[str, Dict[str, list]] = {
    "factual": {
        "high": ["compare", "contrast", "explain why", "mechanism", "history of",
                 "difference between", "relationship between"],
        "low":  ["what is", "define", "capital of", "who is", "when was", "what are"],
    },
    "math": {
        "high": ["calculus", "integral", "derivative", "matrix", "probability",
                 "compound interest", "geometric series", "optimization", "differential"],
        "low":  ["add", "subtract", "multiply", "divide", "percent", "total",
                 "how many", "remaining", "simple interest"],
    },
    "sentiment": {
        "high": ["nuanced", "mixed feelings", "conflicting", "ambivalent", "complex"],
        "low":  ["review", "classify", "label", "sentiment of", "rate"],
    },
    "summarization": {
        "high": ["bullet points", "key themes", "critical analysis", "compare",
                 "multiple documents", "long passage"],
        "low":  ["one sentence", "briefly", "tldr", "summary", "short"],
    },
    "ner": {
        "high": ["nested", "all entities", "co-reference", "multiple paragraphs",
                 "complex document"],
        "low":  ["extract", "identify", "find entities", "named entities", "short text"],
    },
    "code_debug": {
        "high": ["concurrency", "async", "memory leak", "race condition",
                 "O(n)", "performance", "threading"],
        "low":  ["bug", "error", "fix", "incorrect", "off-by-one", "simple function"],
    },
    "logic": {
        "high": ["four people", "five constraints", "grid puzzle", "tournament",
                 "seven variables", "complex"],
        "low":  ["three friends", "two conditions", "who owns", "which is", "simple"],
    },
    "code_gen": {
        "high": ["class", "decorator", "generator", "async", "thread-safe",
                 "design pattern", "algorithm", "data structure", "recursion"],
        "low":  ["function", "returns", "simple", "basic", "write a function",
                 "one-liner"],
    },
}

# ---------------------------------------------------------------------------
# Category detection keywords
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: Dict[str, list] = {
    "math": [
        "calculate", "compute", "how many", "total", "sum", "percentage", "%",
        "multiply", "divide", "arithmetic", "equation", "solve", "math",
        "items remain", "remaining", "profit", "loss", "interest", "average",
        "mean", "median", "probability", "fraction", "algebra", "geometry",
    ],
    "sentiment": [
        "sentiment", "classify the sentiment", "positive", "negative", "opinion",
        "review", "feeling", "emotion", "tone of", "attitude", "rate this",
    ],
    "summarization": [
        "summarize", "summarise", "summary", "condense", "tldr", "brief",
        "in one sentence", "key points", "main idea", "shorten", "overview",
    ],
    "ner": [
        "named entity", "extract entities", "identify entities", "person",
        "organization", "location", "date", "ner", "entity recognition",
        "extract all named",
    ],
    "code_debug": [
        "bug", "debug", "fix this", "find the error", "incorrect output",
        "this function should", "has a bug", "broken", "wrong result",
        "fails", "exception", "traceback", "syntax error",
    ],
    "logic": [
        "puzzle", "constraint", "who owns", "which person", "logical",
        "deductive", "neither", "but not", "exactly one", "three friends",
        "five people", "each own", "different", "who is the",
    ],
    "code_gen": [
        "write a function", "implement", "create a", "code that", "python function",
        "write code", "generate code", "program that", "script that", "function that",
        "write python",
    ],
    # factual is the catch-all / default
    "factual": [
        "what is", "explain", "describe", "how does", "why does", "define",
        "what are", "tell me about", "capital of", "who is", "when was",
        "history of", "difference between",
    ],
}

CATEGORY_ORDER = [
    "math", "sentiment", "summarization", "ner",
    "code_debug", "logic", "code_gen", "factual",
]


def get_system_prompt(category: str) -> str:
    """Return the minimal system prompt for a given category."""
    return SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual"])


def get_output_guidance(category: str) -> str:
    """Return the output guidance hint to append to user prompt (may be empty)."""
    return OUTPUT_GUIDANCE.get(category, "")
