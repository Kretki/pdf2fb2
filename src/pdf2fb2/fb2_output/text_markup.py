"""Text normalisation and Markdown -> FB2 inline markup conversion."""

import html
import re

# B405/S405: parses only escaped fragments built by this program, never external XML.
from xml.etree import ElementTree  # nosec B405  # noqa: S405

from pdf2fb2.fb2_output.fb2_namespaces import XLINK_NAMESPACE

SENTENCE_END_PATTERN = re.compile(r"[.!?…:;»\"”’)\]]\s*$")
LIST_ITEM_PATTERN = re.compile(r"^([-*+•]|\d+[.)])\s+")
BULLET_PATTERN = re.compile(r"^[-*+•]\s+")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?", re.M)
HEADING_NUMBER_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s")
HYPHENATED_END_PATTERN = re.compile(r"\w[-­]$")
THINKING_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.S)
CODE_FENCE_PATTERN = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.M)
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # invalid in XML 1.0
NOTE_TOKEN_PATTERN = re.compile(r"\x00NOTE:(\w+):([^\x00]+)\x00")
NOTE_LINK_TEMPLATE = r'<a l:href="#\1" type="note">\2</a>'
PLAIN_MARKUP_PATTERN = re.compile(r"</?(?:sup|sub|b|i|strong|em)>|\*\*|__|~~|`")
MAX_HEADING_DEPTH = 4

# Applied in order to HTML-escaped text; adding a rule never requires touching the others.
INLINE_MARKUP_RULES = (
    (re.compile(r"&lt;(/?)(sup|sub)&gt;"), r"<\1\2>"),
    (re.compile(r"&lt;(/?)(?:b|strong)&gt;"), r"<\1strong>"),
    (re.compile(r"&lt;(/?)(?:i|em)&gt;"), r"<\1emphasis>"),
    (re.compile(r"&lt;br\s*/?&gt;"), " "),
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"__(.+?)__"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])"), r"<emphasis>\1</emphasis>"),
    (re.compile(r"(?<![_\w])_(?=\S)(.+?)(?<=\S)_(?![_\w])"), r"<emphasis>\1</emphasis>"),
    (re.compile(r"~~(.+?)~~"), r"<strikethrough>\1</strikethrough>"),
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\$\^\{?(\w+)\}?\$"), r"<sup>\1</sup>"),
    (re.compile(r"\$_\{?(\w+)\}?\$"), r"<sub>\1</sub>"),
    (NOTE_TOKEN_PATTERN, NOTE_LINK_TEMPLATE),
)


def escape_text(raw_value: object) -> str:
    """Escape a value for use as XML text content.

    Args:
        raw_value: Any value; converted with ``str``.

    Returns:
        Text with ``&``, ``<`` and ``>`` escaped.
    """
    return html.escape(str(raw_value), quote=False)


def clean_ocr_text(raw_text: str | None) -> str:
    """Strip model artefacts from OCR output.

    Usage:
        Call on every text returned by the OCR worker before caching it.

    Args:
        raw_text: Raw model output (may be None).

    Returns:
        Text without ``<think>`` blocks, code fences and XML-invalid control characters.
    """
    cleaned_text = THINKING_BLOCK_PATTERN.sub("", raw_text or "")
    cleaned_text = CODE_FENCE_PATTERN.sub("", cleaned_text)
    return CONTROL_CHARACTER_PATTERN.sub("", cleaned_text).strip()


def join_text_fragments(leading_text: str, trailing_text: str) -> str:
    """Join two pieces of one paragraph, undoing end-of-line hyphenation.

    Args:
        leading_text: Text before the break.
        trailing_text: Text after the break.

    Returns:
        Joined text: hyphen removed if a hyphenated word continues in lowercase, else a space.
    """
    if HYPHENATED_END_PATTERN.search(leading_text) and trailing_text[:1].islower():
        return leading_text[:-1] + trailing_text
    return leading_text + " " + trailing_text


def collapse_to_single_line(markdown_text: str) -> str:
    """Flatten a Markdown block into one line (used for titles and captions).

    Args:
        markdown_text: OCR Markdown, possibly with ``#`` heading markers.

    Returns:
        Single-line text with heading markers removed and line breaks joined.
    """
    text_lines = [line_text.strip() for line_text in MARKDOWN_HEADING_PATTERN.sub("", markdown_text).splitlines()]
    single_line = ""
    for line_text in filter(None, text_lines):
        single_line = join_text_fragments(single_line, line_text) if single_line else line_text
    return single_line


def _split_chunk_paragraphs(markdown_chunk: str) -> list[tuple[str, bool]]:
    """Split one blank-line-separated chunk into paragraphs and list items.

    Args:
        markdown_chunk: Lines of one Markdown block.

    Returns:
        List of ``(paragraph_text, is_list_item)``.
    """
    chunk_paragraphs = []
    paragraph_buffer = ""
    for line_text in filter(None, (raw_line.strip() for raw_line in markdown_chunk.splitlines())):
        if LIST_ITEM_PATTERN.match(line_text):
            if paragraph_buffer:
                chunk_paragraphs.append((paragraph_buffer, False))
                paragraph_buffer = ""
            chunk_paragraphs.append((BULLET_PATTERN.sub("• ", line_text), True))
        else:
            paragraph_buffer = join_text_fragments(paragraph_buffer, line_text) if paragraph_buffer else line_text
    if paragraph_buffer:
        chunk_paragraphs.append((paragraph_buffer, False))
    return chunk_paragraphs


def split_markdown_paragraphs(markdown_text: str) -> list[tuple[str, bool]]:
    """Split a Markdown block into paragraphs and list items.

    Usage:
        Call for body text, abstracts and footnotes.

    Args:
        markdown_text: OCR Markdown of one segment.

    Returns:
        List of ``(paragraph_text, is_list_item)`` in reading order.
    """
    stripped_text = BLOCKQUOTE_PATTERN.sub("", MARKDOWN_HEADING_PATTERN.sub("", markdown_text))
    all_paragraphs = []
    for markdown_chunk in re.split(r"\n\s*\n", stripped_text):
        all_paragraphs.extend(_split_chunk_paragraphs(markdown_chunk))
    return all_paragraphs


def heading_depth(heading_text: str) -> int:
    """Derive a section depth from a numbered heading ("1.2.3 Title" -> 3).

    Args:
        heading_text: Single-line heading text.

    Returns:
        Depth 1..4; 1 when the heading is not numbered.
    """
    number_match = HEADING_NUMBER_PATTERN.match(heading_text)
    return min(len(number_match.group(1).split(".")), MAX_HEADING_DEPTH) if number_match else 1


def make_note_token(note_identifier: str, marker_text: str) -> str:
    """Build the placeholder that ``markdown_to_fb2_inline`` turns into a note link.

    Args:
        note_identifier: FB2 id of the note section, e.g. ``n3``.
        marker_text: Visible marker, e.g. ``3``.

    Returns:
        Placeholder string delimited by NUL characters (never present in cleaned OCR text).
    """
    return f"\x00NOTE:{note_identifier}:{marker_text}\x00"


def is_well_formed_fragment(xml_fragment: str) -> bool:
    """Check that an inline FB2 fragment is well-formed XML.

    Args:
        xml_fragment: Inline markup; all OCR text inside it is already escaped.

    Returns:
        True if the fragment parses.
    """
    try:
        ElementTree.fromstring(f'<x xmlns:l="{XLINK_NAMESPACE}">{xml_fragment}</x>')  # nosec B314  # noqa: S314
    except ElementTree.ParseError:
        return False
    return True


def markdown_to_fb2_inline(markdown_text: str) -> str:
    """Convert inline Markdown/HTML from the OCR into FB2 inline markup.

    Usage:
        Call for every paragraph, title, caption, table cell and note.

    Args:
        markdown_text: OCR text; may contain note tokens from ``make_note_token``.

    Returns:
        Well-formed FB2 inline markup; falls back to escaped plain text (keeping
        note links) when the converted markup would not be well-formed.
    """
    converted_text = escape_text(markdown_text)
    for markup_pattern, replacement in INLINE_MARKUP_RULES:
        converted_text = markup_pattern.sub(replacement, converted_text)
    if is_well_formed_fragment(converted_text):
        return converted_text
    plain_text = escape_text(PLAIN_MARKUP_PATTERN.sub("", markdown_text))
    return NOTE_TOKEN_PATTERN.sub(NOTE_LINK_TEMPLATE, plain_text)
