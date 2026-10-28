"""Per-character confidence of OCR output: the log-probability of the generated token that wrote each character.

Generated tokens do not map one-to-one to characters (a token may hold several characters, a character may need
several byte-level tokens), so the text is decoded incrementally and every new character takes the lowest
log-probability among the tokens that completed it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

LOGPROB_DIGITS = 3
UNKNOWN_LOGPROB = 0.0


@dataclass(frozen=True)
class RecognizedText:
    """OCR output with its confidence.

    Attributes:
        recognized_text: Decoded text.
        char_logprobs: One log-probability (<= 0) per character of ``recognized_text``.
    """

    recognized_text: str
    char_logprobs: tuple[float, ...]


def incremental_char_logprobs(token_count: int, prefix_decoder: Callable[[int], str],
                              token_logprobs: Sequence[float]) -> tuple[str, list[float]]:
    """Characters written by every token, from incremental decoding of growing prefixes.

    Args:
        token_count: Generated tokens.
        prefix_decoder: Decodes the first ``n`` generated tokens to text.
        token_logprobs: Log-probability of every generated token.

    Returns:
        ``(decoded text, one log-probability per character)``.
    """
    char_logprobs: list[float] = []
    decoded_text, pending_logprob = "", 0.0
    for token_index in range(token_count):
        pending_logprob = min(pending_logprob, float(token_logprobs[token_index]))
        prefix_text = prefix_decoder(token_index + 1)
        if prefix_text.endswith("�") or not prefix_text.startswith(decoded_text):
            continue
        char_logprobs.extend([pending_logprob] * (len(prefix_text) - len(decoded_text)))
        decoded_text, pending_logprob = prefix_text, 0.0
    final_text = prefix_decoder(token_count)
    char_logprobs.extend([pending_logprob] * max(0, len(final_text) - len(char_logprobs)))
    return final_text, char_logprobs[:len(final_text)]


def joined_char_logprobs(token_texts: Sequence[str], token_logprobs: Sequence[float],
                         message_text: str) -> list[float]:
    """Per-character log-probabilities from token strings returned by a server (OpenAI ``logprobs``).

    Args:
        token_texts: Token strings in order.
        token_logprobs: Their log-probabilities.
        message_text: Final message text.

    Returns:
        One value per character of ``message_text``; ``UNKNOWN_LOGPROB`` where the tokens do not spell the text.
    """
    char_logprobs = [float(token_logprob) for token_text, token_logprob in zip(token_texts, token_logprobs,
                                                                                strict=True)
                     for _ in token_text]
    if "".join(token_texts) != message_text:
        return [UNKNOWN_LOGPROB] * len(message_text)
    return char_logprobs


def rounded_logprobs(char_logprobs: Sequence[float]) -> list[float]:
    """Compact form for result files.

    Args:
        char_logprobs: Raw values.

    Returns:
        Values rounded to ``LOGPROB_DIGITS`` decimals.
    """
    return [round(char_logprob, LOGPROB_DIGITS) for char_logprob in char_logprobs]
