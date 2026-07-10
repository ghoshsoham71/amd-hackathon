"""
Category-specific system prompts - AMD Hackathon Track 1.

Design principles:
1. Every token in the system prompt is SCORED - keep them minimal.
2. Output guidance is appended to the USER message instead of duplicated here.
3. For tasks requiring reasoning (logic, math), allow internal reasoning but
   enforce a strict final-answer format so validation works and output is short.
4. For purely extractive tasks (sentiment, NER), no reasoning needed.
"""

from typing import Dict

# -- Core system prompts --------------------------------------------------------
# Rule: system prompt only. Output guidance is in OUTPUT_GUIDANCE (appended
# to user message). Do NOT duplicate these here - it wastes scored tokens.
SYSTEM_PROMPTS: Dict[str, str] = {
    # Factual: direct answer, no filler, no caveats
    "factual": (
        "You are a factual assistant. Answer directly and concisely. "
        "No preamble, no caveats, no apologies."
    ),
    # Math: allow internal reasoning but output ONLY the final numeric answer
    "math": (
        "Solve the problem step by step, then output ONLY: ANSWER: <number>"
    ),
    # Sentiment: strict label, no prose
    "sentiment": (
        "Output ONLY: SENTIMENT: Positive, SENTIMENT: Negative, "
        "SENTIMENT: Neutral, or SENTIMENT: Mixed. No explanation."
    ),
    # Summarization: produce only the summary, match requested length
    "summarization": (
        "Summarize the provided text. Output ONLY the summary. "
        "No preamble or meta-commentary."
    ),
    # NER: strict JSON, no prose around it
    "ner": (
        'Extract named entities. Output ONLY a JSON array: '
        '[{"text":"...","type":"PERSON|ORG|LOC|DATE|OTHER"}]. '
        "No markdown fences, no explanation."
    ),
    # Code debug: return corrected code only, minimal comment if essential
    "code_debug": (
        "Fix the bug in the code. Return ONLY the corrected, complete code. "
        "No explanation before or after."
    ),
    # Logic: MUST reason to get correct answer, but format is strict
    "logic": (
        "Reason through the problem carefully, then output ONLY: ANSWER: <value>. "
        "Put all reasoning before the ANSWER line."
    ),
    # Code gen: Python only, no prose, no markdown explanation outside code block
    "code_gen": (
        "Write correct, complete, idiomatic Python. "
        "Return ONLY the code. No prose, no explanation."
    ),
}

# -- Output guidance appended to USER message (not duplicated from system) ------
# These are short reminders that reinforce the format. They should NOT repeat
# the system prompt verbatim - that wastes tokens.
OUTPUT_GUIDANCE: Dict[str, str] = {
    "factual":       "Answer in 1-3 sentences max.",
    "math":          "Show brief working if needed, then: ANSWER: <number>",
    "sentiment":     "SENTIMENT: <Positive|Negative|Neutral|Mixed>",
    "summarization": "Provide ONLY the summary. Match length requested.",
    "ner":           'JSON array only: [{"text":"...","type":"..."}]',
    "code_debug":    "Corrected code only. No prose.",
    "logic":         "Reason first, then end with: ANSWER: <value>",
    "code_gen":      "Python code only. No prose.",
}

# -- Difficulty estimation keywords per category --------------------------------
DIFFICULTY_SIGNALS: Dict[str, Dict[str, list]] = {
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

# -- Category detection keywords ------------------------------------------------
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
