"""Normalisation of reference and output text before comparison.

Both sides go through the same rules, so differences of convention are not counted as OCR errors:
Unicode NFC, Wikisource and Markdown markup removed, line-break hyphens (``¬``, soft hyphen, ``-`` at a line
end) joined, quotes and dashes unified, long ``ſ`` read as ``s``, whitespace collapsed. Letter case and the
old Russian letters (ѣ, і, ъ) are kept: they are part of the printed text. Chinese and Japanese are written
without spaces, so every Han / kana character is a word of its own (word scores become character scores).
"""

import html
import re
import unicodedata

WIKI_MARKUP_PATTERN = re.compile(r"'{2,}|\{\{|\}\}|\|")
MARKDOWN_PATTERN = re.compile(r"(?m)(?:^|(?<=\s))#{1,6}(?=\s|$)|\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)|^\s*[-*+]\s+(?=\S)")
STYLE_TAG_PATTERN = re.compile(r"</?[a-zA-Z][^<>]{0,200}>")
LINE_HYPHEN_PATTERN = re.compile(r"(\w)(?:[¬­]\s*|[-‐]\s*\n\s*)(\w)")
QUOTE_PATTERN = re.compile(r"[«»„“”‟\"]")
APOSTROPHE_PATTERN = re.compile(r"[‘’‚‛`´]")
DASH_PATTERN = re.compile(r"[‐‑‒–—―]")
SPACE_PATTERN = re.compile(r"\s+")
CJK_CHARACTER_PATTERN = re.compile(r"([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3005\u3006])")


def normalize_text(source_text: str) -> str:
    """Normalise a text for comparison.

    Usage:
        Apply to the reference and to the pipeline output alike.

    Args:
        source_text: Raw text (may contain markup, XML entities, line breaks).

    Returns:
        One-line normalised text.
    """
    plain_text = unicodedata.normalize("NFC", html.unescape(source_text))
    plain_text = STYLE_TAG_PATTERN.sub(" ", plain_text)
    plain_text = WIKI_MARKUP_PATTERN.sub("", plain_text)
    plain_text = MARKDOWN_PATTERN.sub("", plain_text)
    plain_text = LINE_HYPHEN_PATTERN.sub(r"\1\2", plain_text)
    plain_text = QUOTE_PATTERN.sub('"', APOSTROPHE_PATTERN.sub("'", plain_text))
    plain_text = DASH_PATTERN.sub("-", plain_text).replace("ſ", "s").replace("…", "...")
    return SPACE_PATTERN.sub(" ", plain_text).strip()


def comparison_words(source_text: str) -> list[str]:
    """Words of a text after normalisation.

    Args:
        source_text: Raw text.

    Returns:
        Whitespace-separated normalised words.
    """
    return CJK_CHARACTER_PATTERN.sub(r" \1 ", normalize_text(source_text)).split()


def token_form(raw_token: str) -> str:
    """Normalised form of one output token (empty for pure markup such as ``#`` or ``**``).

    Args:
        raw_token: Whitespace token of the pipeline output.

    Returns:
        The token as it would appear among ``comparison_words``.
    """
    return normalize_text(raw_token).replace(" ", "")


def match_key(word_text: str) -> str:
    """Loose form used to find matching passages (case and punctuation ignored).

    Args:
        word_text: Normalised word.

    Returns:
        Lower-case letters and digits of the word.
    """
    return "".join(word_char for word_char in word_text.casefold() if word_char.isalnum())
