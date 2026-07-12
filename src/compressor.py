"""
Prompt compressor — strips token fat before any Fireworks API call.

Techniques applied in order (most to least safe):

1.  Unicode / whitespace normalization
    - Zero-width spaces, non-breaking spaces, smart quotes → ASCII
    - Collapse 3+ newlines → 2; collapse 3+ spaces → 1

2.  Boilerplate preamble removal
    - "Please could you...", "I would like you to...", "Can you please..."

3.  Redundant output-guidance deduplication
    - Strip output formats from the prompt if they are already present.

4.  Filler word stripping (Generic)
    - Removes common filler words ("please", "kindly", "basically", etc.) 
      from the instruction part of the prompt without needing hardcoded phrases.

5.  Markdown / formatting strip (for non-code tasks)
    - Strip # headings, ** bold **, _ italic _

6.  Free Local LLM Compression
    - Pass the prompt to the local 0.5B model and ask it to rewrite it shorter.
    - Since local tokens cost 0 points, this is a free way to reduce Fireworks tokens!

7.  TF-IDF sentence scoring (summarization / NER / factual)
    - Score each sentence by TF-IDF and prune the fluff sentences.

8.  Sentence-boundary truncation (fallback)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from src.token_counter import count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Unicode / whitespace normalization
# ---------------------------------------------------------------------------
_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u200b", ""),    # zero-width space
    ("\u200c", ""),    # zero-width non-joiner
    ("\u200d", ""),    # zero-width joiner
    ("\u00a0", " "),   # non-breaking space
    ("\u2019", "'"),   # right single quotation mark
    ("\u2018", "'"),   # left single quotation mark
    ("\u201c", '"'),   # left double quotation mark
    ("\u201d", '"'),   # right double quotation mark
    ("\u2013", "-"),   # en dash
    ("\u2014", "-"),   # em dash
    ("\u2026", "..."), # ellipsis
    ("\u00ad", ""),    # soft hyphen
]

def _normalize_unicode(text: str) -> str:
    for src, tgt in _UNICODE_REPLACEMENTS:
        text = text.replace(src, tgt)
    return text


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Boilerplate preamble removal
# ---------------------------------------------------------------------------
_PREAMBLE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(please\s+)?(can|could)\s+you\s+(please\s+)?", re.I),
    re.compile(r"^i\s+(would|'d)\s+like\s+(you\s+)?to\s+", re.I),
    re.compile(r"^i\s+need\s+(you\s+)?to\s+", re.I),
    re.compile(r"^i\s+want\s+(you\s+)?to\s+", re.I),
    re.compile(r"^your\s+task\s+is\s+to\s+", re.I),
    re.compile(r"^your\s+job\s+is\s+to\s+", re.I),
    re.compile(r"\.\s*(please\s+be\s+(concise|brief|short)\.?\s*)$", re.I),
    re.compile(r"\.\s*(keep\s+(it\s+)?(concise|brief|short)\.?\s*)$", re.I),
    re.compile(r"\.\s*(do\s+not\s+include\s+unnecessary\s+details?\.?\s*)$", re.I),
    re.compile(r"\.\s*(answer\s+directly\.?\s*)$", re.I),
]

def _strip_preambles(text: str) -> str:
    for pat in _PREAMBLE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Output-guidance deduplication
# ---------------------------------------------------------------------------
def _dedup_output_guidance(text: str, guidance: str) -> str:
    if not guidance or not text:
        return text
    stripped_guidance = re.escape(guidance.strip())
    pattern = re.compile(r"\s*" + stripped_guidance + r"\s*$", re.I)
    return pattern.sub("", text).strip()


# ---------------------------------------------------------------------------
# 4. Filler & Stop Word Stripping
# ---------------------------------------------------------------------------
_FILLER_WORDS = {
    "please", "kindly", "basically", "essentially", "simply", "just",
    "following", "provided", "below", "therein", "moreover", "furthermore"
}

_STOP_WORDS = {
    "a", "an", "the", "of", "to", "in", "for", "with", "on", "at", "from",
    "by", "about", "as", "into", "like", "through", "after", "over",
    "between", "out", "against", "during", "without", "before", "under"
}

def _strip_filler_words(text: str, category: str, aggressive: bool = False) -> str:
    """
    Remove safe filler words from the instruction part of the prompt (first ~150 chars).
    If aggressive=True, also remove standard English stopwords globally.
    """
    if not text:
        return text
    
    # Process filler words on the head
    split_idx = min(150, len(text))
    if split_idx == 150:
        match = re.search(r"[:\n]", text[:150])
        if match:
            split_idx = match.end()

    head = text[:split_idx]
    tail = text[split_idx:]

    head_words = head.split()
    kept_head = []
    for w in head_words:
        clean_w = w.lower().strip(".,:;!?")
        if clean_w not in _FILLER_WORDS:
            kept_head.append(w)
    
    processed_text = " ".join(kept_head) + (" " + tail if tail else "")

    if aggressive:
        # Strip stopwords globally to aggressively shrink tokens
        words = processed_text.split()
        kept_all = []
        for w in words:
            if w.lower().strip(".,:;!?") not in _STOP_WORDS:
                kept_all.append(w)
        processed_text = " ".join(kept_all)

    return processed_text


# ---------------------------------------------------------------------------
# 5. Markdown formatting strip (non-code tasks only)
# ---------------------------------------------------------------------------
_MARKDOWN_PATTERNS: list[re.Pattern] = [
    re.compile(r"^#{1,6}\s+", re.M),
    re.compile(r"\*{2}(.+?)\*{2}", re.S),
    re.compile(r"\*(.+?)\*", re.S),
    re.compile(r"_{2}(.+?)_{2}", re.S),
    re.compile(r"_(.+?)_", re.S),
    re.compile(r"^[-*]{3,}\s*$", re.M),
    re.compile(r"\[(.+?)\]\(.+?\)", re.S),
    re.compile(r"^>\s+", re.M),
]

def _strip_markdown(text: str) -> str:
    for pat in _MARKDOWN_PATTERNS:
        text = pat.sub(r"\1" if pat.groups else "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 6. Free Local LLM Compression
# ---------------------------------------------------------------------------
def _local_llm_compress(text: str, available_tokens: int) -> str:
    """
    Passes the prompt to the local model to rewrite it generically.
    Costs 0 tokens but saves Fireworks tokens!
    """
    from src import local_model
    if not local_model.is_available():
        return text

    # Only compress if prompt is large enough to be worth it (> 40 tokens)
    # but small enough to fit in local model context window (max input ~1024)
    orig_tok = count_tokens(text)
    if orig_tok < 40 or orig_tok > 800:
        return text

    system_msg = (
        "You are an expert prompt compressor. Rewrite the user's text to be as "
        "concise as possible while keeping all instructions, facts, and constraints. "
        "Do NOT answer the prompt. Output ONLY the compressed text."
    )

    answer = local_model.infer(
        system_prompt=system_msg,
        user_prompt=text,
        max_tokens=min(available_tokens, orig_tok),
        temperature=0.05,
    )

    if answer and len(answer) > 10:
        new_tok = count_tokens(answer)
        if new_tok < orig_tok:
            # Basic sanity check: did it refuse or loop?
            if not re.search(r"^(i cannot|i am unable|as an ai)", answer, re.I):
                logger.info("Local LLM compressed prompt: %d -> %d tok", orig_tok, new_tok)
                return answer

    return text


# ---------------------------------------------------------------------------
# 7. TF-IDF sentence scoring + pruning
# ---------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in raw if s.strip()]

def _tfidf_scores(sentences: list[str]) -> list[float]:
    if not sentences:
        return []

    def tokenize(s: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", s.lower())

    tokenized = [tokenize(s) for s in sentences]
    n = len(sentences)

    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    scores: list[float] = []
    for i, tokens in enumerate(tokenized):
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        score = sum(
            (tf[w] / len(tokens)) * math.log((n + 1) / (df[w] + 1))
            for w in tf
        )
        if re.search(r"\d", sentences[i]):
            score *= 1.2
        scores.append(score)

    return scores

def _prune_by_tfidf(text: str, available_tokens: int, keep_first: int = 2) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= keep_first + 1:
        return text

    instruction = sentences[:keep_first]
    body = sentences[keep_first:]

    scores = _tfidf_scores(body)
    ranked = sorted(range(len(body)), key=lambda i: scores[i], reverse=True)

    budget = available_tokens - count_tokens(" ".join(instruction)) - 10
    kept_indices: set[int] = set()
    used = 0
    for idx in ranked:
        tok = count_tokens(body[idx])
        if used + tok > budget:
            break
        kept_indices.add(idx)
        used += tok

    kept_body = [body[i] for i in sorted(kept_indices)]
    return " ".join(instruction + kept_body)


# ---------------------------------------------------------------------------
# 8. Sentence-boundary truncation
# ---------------------------------------------------------------------------
def _truncate_at_sentence(text: str, max_tokens: int) -> str:
    sentences = _split_sentences(text)
    result: list[str] = []
    used = 0
    for sent in sentences:
        tok = count_tokens(sent) + 1
        if used + tok > max_tokens - 5:
            break
        result.append(sent)
        used += tok

    if result:
        return " ".join(result)

    words = text.split()
    out: list[str] = []
    cur = 0
    for w in words:
        wt = count_tokens(w) + 1
        if cur + wt > max_tokens - 5:
            out.append("[...]")
            break
        out.append(w)
        cur += wt
    return " ".join(out)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_PRUNABLE_CATEGORIES = {"summarization", "ner", "factual"}
_STRIP_MD_CATEGORIES = {"factual", "summarization", "math", "logic", "sentiment", "ner"}


# ---------------------------------------------------------------------------
# Main compress function
# ---------------------------------------------------------------------------
def compress_prompt(
    prompt: str,
    category: str,
    available_tokens: int,
    aggressive: bool = False,
    guidance_text: str = "",
) -> str:
    original_tokens = count_tokens(prompt)
    if original_tokens <= available_tokens:
        return prompt

    logger.info(
        "compress[%s]: %d → target %d tokens (aggressive=%s)",
        category, original_tokens, available_tokens, aggressive,
    )

    compressed = prompt

    # 1. Unicode & whitespace
    compressed = _normalize_unicode(compressed)
    compressed = _normalize_whitespace(compressed)
    if count_tokens(compressed) <= available_tokens:
        return compressed

    # 2. Output guidance dedup
    if guidance_text:
        compressed = _dedup_output_guidance(compressed, guidance_text)
    if count_tokens(compressed) <= available_tokens:
        return compressed

    # 3. Preamble removal
    compressed = _strip_preambles(compressed)
    if count_tokens(compressed) <= available_tokens:
        return compressed

    # 4. Generic filler & stopword stripping
    compressed = _strip_filler_words(compressed, category, aggressive=aggressive)
    if count_tokens(compressed) <= available_tokens:
        return compressed

    # 5. Markdown strip
    if category in _STRIP_MD_CATEGORIES:
        compressed = _strip_markdown(compressed)
        if count_tokens(compressed) <= available_tokens:
            return compressed

    # 6. Local LLM Compression (Free tokens!)
    # Only try this if we still need to compress significantly
    if count_tokens(compressed) > available_tokens * 1.2:
        compressed = _local_llm_compress(compressed, available_tokens)
        if count_tokens(compressed) <= available_tokens:
            return compressed

    # 7. TF-IDF pruning
    if category in _PRUNABLE_CATEGORIES:
        budget_for_prune = available_tokens if not aggressive else int(available_tokens * 0.85)
        compressed = _prune_by_tfidf(compressed, budget_for_prune)
        if count_tokens(compressed) <= available_tokens:
            return compressed

    # 8. Sentence-boundary truncation
    if count_tokens(compressed) > available_tokens:
        compressed = _truncate_at_sentence(compressed, available_tokens)
        logger.warning(
            "compress[%s]: sentence-truncated to %d tokens",
            category, count_tokens(compressed),
        )

    return compressed
