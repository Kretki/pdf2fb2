"""How paragraph texts become FB2 inline markup, as an injectable strategy.

The section tree and the notes body ask a ``ParagraphMarkup`` for every paragraph, title and caption: the plain
strategy only converts the Markdown, the multilingual one (``text_languages.language_markup``) also adds
``xml:lang`` to paragraphs and to foreign text runs.
"""

from typing import Protocol

from pdf2fb2.fb2_output.text_markup import markdown_to_fb2_inline


class ParagraphMarkup(Protocol):
    """Renders the text of one FB2 paragraph."""

    @property
    def main_language(self) -> str:
        """Language code of the book, written to the FB2 ``<lang>``."""
        ...

    def render_paragraph(self, markdown_text: str) -> tuple[str, str]:
        """Render one paragraph text.

        Args:
            markdown_text: OCR Markdown inline text of the paragraph.

        Returns:
            ``(attribute_text, inline_markup)``: attributes for the ``<p>`` element (empty or starting with a
            space, e.g. `` xml:lang="ja"``) and its well-formed content.
        """
        ...


class PlainParagraphMarkup:
    """Single-language rendering: Markdown conversion only.

    Usage:
        Default of the section tree; used for books with one declared language.
    """

    def __init__(self, main_language: str) -> None:
        """Store the book language.

        Args:
            main_language: Language code of the book, e.g. ``en``.
        """
        self._main_language = main_language

    @property
    def main_language(self) -> str:
        """Language code of the book.

        Returns:
            The code given at construction.
        """
        return self._main_language

    def render_paragraph(self, markdown_text: str) -> tuple[str, str]:
        """Convert the Markdown of one paragraph.

        Args:
            markdown_text: OCR Markdown inline text.

        Returns:
            ``("", inline_markup)``.
        """
        return "", markdown_to_fb2_inline(markdown_text)
