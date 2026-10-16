"""Group scored tokens by word: per-word token probabilities, minimum and mean log-probability."""

from proofread_worker.document_text import DocumentText
from proofread_worker.token_scorer import ScoredToken

LOGPROB_DIGITS = 3


def tokens_by_word(document_text: DocumentText, scored_tokens: list[ScoredToken]) -> list[list[ScoredToken]]:
    """Assign every token to the word that contains its last character.

    Tokens that consist only of separators (blank lines) belong to no word.

    Args:
        document_text: Document with word spans.
        scored_tokens: Tokens of the document.

    Returns:
        One token list per word span, in document order.
    """
    word_tokens: list[list[ScoredToken]] = [[] for _ in document_text.word_spans]
    for scored_token in scored_tokens:
        if scored_token.end_offset <= scored_token.start_offset:
            continue
        span_index = document_text.span_containing(scored_token.end_offset - 1)
        if span_index is not None:
            word_tokens[span_index].append(scored_token)
    return word_tokens


def word_entry(document_text: DocumentText, scored_tokens: list[ScoredToken]) -> dict:
    """Result entry of one word.

    Args:
        document_text: Document (token texts are sliced from it).
        scored_tokens: Tokens of the word.

    Returns:
        ``{"tokens": [[text, logprob], ...], "min_logprob", "mean_logprob"}`` (0.0 when unscored).
    """
    token_logprobs = [scored_token.token_logprob for scored_token in scored_tokens]
    return {"tokens": [[document_text.full_text[max(0, scored_token.start_offset):scored_token.end_offset],
                        round(scored_token.token_logprob, LOGPROB_DIGITS)] for scored_token in scored_tokens],
            "min_logprob": round(min(token_logprobs), LOGPROB_DIGITS) if token_logprobs else 0.0,
            "mean_logprob": round(sum(token_logprobs) / len(token_logprobs), LOGPROB_DIGITS) if token_logprobs
            else 0.0}


def block_word_entries(document_text: DocumentText, word_tokens: list[list[ScoredToken]],
                       block_count: int) -> list[dict]:
    """Per-block lists of word entries.

    Args:
        document_text: Document with word spans.
        word_tokens: Result of ``tokens_by_word``.
        block_count: Number of blocks.

    Returns:
        ``[{"words": [word entry, ...]}, ...]`` with one item per block.
    """
    block_entries: list[dict] = [{"words": []} for _ in range(block_count)]
    for word_span, scored_tokens in zip(document_text.word_spans, word_tokens, strict=True):
        block_entries[word_span.block_index]["words"].append(word_entry(document_text, scored_tokens))
    return block_entries
