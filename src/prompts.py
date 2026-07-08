"""
Category-specific minimized system prompts.

Design principle: Every token in the system prompt is COUNTED against our score
when using the Fireworks API. These prompts are deliberately minimal — they give
the model just enough context to produce accurate, well-formatted answers without
any filler words.

Each prompt is tuned to:
1. Suppress unnecessary preamble ("Of course!", "Certainly!", "Great question!")
2. Enforce a specific output format for easy validation
3. Stay under 30 tokens
"""

from typing import Dict

# ── Core minimal system prompts ──────────────────────────────────────────────
SYSTEM_PROMPTS: Dict[str, str] = {
    "factual": (
        "Answer accurately and concisely. DO NOT output reasoning or thoughts. State facts directly."
    ),
    "math": (
        "DO NOT show working or reasoning. End with: ANSWER: <number>"
    ),
    "sentiment": (
        "Classify sentiment as Positive, Negative, or Mixed. DO NOT output reasoning. "
        "Format: SENTIMENT: <label>"
    ),
    "summarization": (
        "Summarize as instructed. Be concise. DO NOT output reasoning. Provide ONLY the final summary."
    ),
    "ner": (
        "Extract named entities. Output JSON array only. NO REASONING. "
        '[{"text":"...","type":"PERSON|ORG|LOC|DATE|OTHER"}]'
    ),
    "code_debug": (
        "Find the bug and return ONLY the corrected code. DO NOT output reasoning or thought blocks."
    ),
    "logic": (
        "DO NOT output reasoning. Provide ONLY the final answer: "
        "ANSWER: <answer>"
    ),
    "code_gen": (
        "Write correct, idiomatic Python. Code only. DO NOT output reasoning, thought process, or explanations."
    ),
}

# ── Output length guidance appended to user prompt (not system) ───────────────
OUTPUT_GUIDANCE: Dict[str, str] = {
    "factual":       "Be concise. DO NOT output reasoning or <think> blocks.",
    "math":          "DO NOT show working. Provide ONLY the final answer: ANSWER: <value>",
    "sentiment":     "One line max. DO NOT output reasoning.",
    "summarization": "Match the requested length exactly. DO NOT output your reasoning or thought process. Provide ONLY the final summary.",
    "ner":           "JSON array only, no markdown wrapping. NO REASONING.",
    "code_debug":    "Return only corrected code block. NO REASONING or explanations.",
    "logic":         "DO NOT output your reasoning. Provide ONLY the final answer: ANSWER: <value>",
    "code_gen":      "Code only, no prose. DO NOT output reasoning.",
}

# ── Difficulty estimation keywords per category ────────────────────────────────
DIFFICULTY_SIGNALS: Dict[str, Dict[str, float]] = {
    "factual": {
        "high": ["compare", "contrast", "explain why", "mechanism", "history of"],
        "low":  ["what is", "define", "capital of", "who is", "when was"],
    },
    "math": {
        "high": ["calculus", "integral", "derivative", "matrix", "probability",
                 "compound interest", "geometric series", "optimization"],
        "low":  ["add", "subtract", "multiply", "divide", "percent", "total",
                 "how many", "remaining"],
    },
    "sentiment": {
        "high": ["nuanced", "mixed feelings", "conflicting", "ambivalent"],
        "low":  ["review", "classify", "label", "sentiment of"],
    },
    "summarization": {
        "high": ["bullet points", "key themes", "critical analysis", "compare"],
        "low":  ["one sentence", "briefly", "tldr", "summary"],
    },
    "ner": {
        "high": ["nested", "all entities", "co-reference", "multiple paragraphs"],
        "low":  ["extract", "identify", "find entities", "named entities"],
    },
    "code_debug": {
        "high": ["concurrency", "async", "memory leak", "race condition", "O(n)"],
        "low":  ["bug", "error", "fix", "incorrect", "off-by-one"],
    },
    "logic": {
        "high": ["four people", "five constraints", "grid puzzle", "tournament"],
        "low":  ["three friends", "two conditions", "who owns", "which is"],
    },
    "code_gen": {
        "high": ["class", "decorator", "generator", "async", "thread-safe",
                 "design pattern", "algorithm", "data structure"],
        "low":  ["function", "returns", "simple", "basic", "write a function"],
    },
}

# ── Category detection keywords ────────────────────────────────────────────────
CATEGORY_KEYWORDS: Dict[str, list] = {
    "math": [
        "calculate", "compute", "how many", "total", "sum", "percentage", "%",
        "multiply", "divide", "arithmetic", "equation", "solve", "math",
        "items remain", "remaining", "profit", "loss", "interest", "average",
        "mean", "median", "probability", "fraction",
    ],
    "sentiment": [
        "sentiment", "classify the sentiment", "positive", "negative", "opinion",
        "review", "feeling", "emotion", "tone of", "attitude",
    ],
    "summarization": [
        "summarize", "summarise", "summary", "condense", "tldr", "brief",
        "in one sentence", "key points", "main idea", "shorten",
    ],
    "ner": [
        "named entity", "extract entities", "identify entities", "person",
        "organization", "location", "date", "ner", "entity recognition",
        "extract all named",
    ],
    "code_debug": [
        "bug", "debug", "fix this", "find the error", "incorrect output",
        "this function should", "has a bug", "broken", "wrong result",
        "fails", "exception",
    ],
    "logic": [
        "puzzle", "constraint", "who owns", "which person", "logical",
        "deductive", "neither", "but not", "exactly one", "three friends",
        "five people", "each own", "different",
    ],
    "code_gen": [
        "write a function", "implement", "create a", "code that", "python function",
        "write code", "generate code", "program that", "script that", "function that",
    ],
    # factual is the catch-all / default
    "factual": [
        "what is", "explain", "describe", "how does", "why does", "define",
        "what are", "tell me about", "capital of", "who is", "when was",
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
    """Return the output guidance hint to append to user prompt."""
    return OUTPUT_GUIDANCE.get(category, "")
