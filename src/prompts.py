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
    # Factual: provide a complete and accurate answer
    "factual": (
        "You are an expert factual assistant. Provide a complete, accurate, and helpful answer. "
        "Do not include unnecessary conversational filler."
    ),
    # Math: strict CoT reasoning required
    "math": (
        "You are an expert mathematician. Think step-by-step to solve the problem. "
        "Write out your complete reasoning clearly. Always end your response with: ANSWER: <number>"
    ),
    # Sentiment: clear label with brief justification
    "sentiment": (
        "Classify the sentiment of the text. First provide a brief one-sentence reason, "
        "then end with: SENTIMENT: Positive, SENTIMENT: Negative, SENTIMENT: Neutral, or SENTIMENT: Mixed."
    ),
    # Summarization: focus on quality and completeness
    "summarization": (
        "You are an expert summarizer. Capture all key themes, critical points, and the main idea accurately. "
        "Output the summary directly without preamble."
    ),
    # NER: strict JSON formatting but robust extraction
    "ner": (
        'You are an expert at Named Entity Recognition. Carefully extract all valid entities. '
        'Output ONLY a valid JSON array: [{"text":"...","type":"PERSON|ORG|LOC|DATE|OTHER"}]. '
        "Ensure the JSON is perfectly formatted."
    ),
    # Code debug: CoT debugging
    "code_debug": (
        "You are a senior software engineer. First, briefly explain the bug and how to fix it. "
        "Then, provide the completely corrected code inside a ```python block. Ensure the code is robust."
    ),
    # Logic: strict CoT reasoning
    "logic": (
        "You are an expert logician. Reason through the problem step-by-step. "
        "State your deductions clearly, then end your response with: ANSWER: <value>"
    ),
    # Code gen: clear requirements and robust code
    "code_gen": (
        "You are a senior software engineer. Write robust, correct, and idiomatic Python code. "
        "Include necessary comments and handle edge cases. Output the code inside a ```python block."
    ),
}

# -- Output guidance appended to USER message (not duplicated from system) ------
# These are short reminders that reinforce the format. They should NOT repeat
# the system prompt verbatim - that wastes tokens.
OUTPUT_GUIDANCE: Dict[str, str] = {
    "factual":       "Answer directly and factually.",
    "math":          "Think step-by-step. End with: ANSWER: <number>",
    "sentiment":     "Reason briefly. End with: SENTIMENT: <Label>",
    "summarization": "Provide a complete and accurate summary.",
    "ner":           'Valid JSON array only: [{"text":"...","type":"..."}]',
    "code_debug":    "Explain the fix, then provide the corrected code.",
    "logic":         "Think step-by-step. End with: ANSWER: <value>",
    "code_gen":      "Write robust Python code.",
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
