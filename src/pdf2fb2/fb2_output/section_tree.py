"""Nests the book flow into FB2 ``<section>`` elements and renders the main body."""

from collections.abc import Callable

from pdf2fb2.fb2_output.paragraph_markup import ParagraphMarkup, PlainParagraphMarkup
from pdf2fb2.fb2_output.table_markup import render_fb2_table
from pdf2fb2.fb2_output.text_markup import escape_text

EMPTY_BODY = "<section><empty-line/></section>"


class BookSection:
    """One FB2 section: optional title, content items and child sections.

    Usage:
        Built by ``SectionTreeBuilder``; ``render_xml`` produces the markup.
    """

    def __init__(self, nesting_level: int, title_markup: str | None) -> None:
        """Create an empty section.

        Args:
            nesting_level: Heading depth (0 = root).
            title_markup: Rendered title paragraph (``<p>...</p>``) or None.
        """
        self.nesting_level = nesting_level
        self.title_markup = title_markup
        self.content_items: list[str] = []
        self.child_sections: list[BookSection] = []

    def render_xml(self) -> str:
        """Render the section and its children.

        Returns:
            ``<section>...</section>`` markup.
        """
        section_parts = ["<section>"]
        if self.title_markup:
            section_parts.append(f"<title>{self.title_markup}</title>")
        section_parts.extend(self._render_content())
        section_parts.append("</section>")
        return "".join(section_parts)

    def _render_content(self) -> list[str]:
        """Render content; FB2 forbids mixing content and subsections, so leading content is wrapped.

        Returns:
            Markup parts of the section body.
        """
        rendered_children = [child_section.render_xml() for child_section in self.child_sections]
        if self.content_items and self.child_sections:
            preface_section = BookSection(self.nesting_level + 1, None)
            preface_section.content_items = self.content_items
            return [preface_section.render_xml(), *rendered_children]
        if self.content_items or self.child_sections:
            return [*self.content_items, *rendered_children]
        return ["<empty-line/>"]


class SectionTreeBuilder:
    """Consumes flow items one by one and builds the section tree.

    Usage:
        ``add_flow_item`` for each item, then ``render_body_xml``. New item kinds or
        page-marker modes are added by registering a handler.
    """

    def __init__(self, page_marker_mode: str, paragraph_markup: ParagraphMarkup | None = None) -> None:
        """Create an empty tree.

        Args:
            page_marker_mode: ``anchor`` (invisible id on the first paragraph of a page),
                ``visible`` (``— N —`` subtitle) or ``none``.
            paragraph_markup: Renders paragraph texts (None: Markdown conversion only).

        Raises:
            KeyError: If the mode is unknown.
        """
        self._paragraph_markup = paragraph_markup or PlainParagraphMarkup("")
        self._root_section = BookSection(0, None)
        self._section_stack = [self._root_section]
        self._pending_anchor: str | None = None
        marker_handlers: dict[str, Callable[[dict], None]] = {
            "anchor": self._remember_anchor, "visible": self._insert_visible_marker, "none": lambda flow_item: None}
        self._kind_handlers: dict[str, Callable[[dict], None]] = {
            "page": marker_handlers[page_marker_mode], "heading": self._open_heading, "p": self._add_paragraph,
            "caption": self._add_caption, "image": self._add_image, "table": self._add_table}

    def add_flow_item(self, flow_item: dict) -> None:
        """Dispatch one flow item to its handler.

        Args:
            flow_item: Item from ``BookFlow.flow_items``.

        Returns:
            None.
        """
        self._kind_handlers[flow_item["kind"]](flow_item)

    def render_body_xml(self) -> str:
        """Render all top-level sections.

        Returns:
            Section markup for the main ``<body>`` (never empty).
        """
        top_sections = []
        if self._root_section.content_items:
            preface_section = BookSection(1, None)
            preface_section.content_items = self._root_section.content_items
            top_sections.append(preface_section)
        top_sections.extend(self._root_section.child_sections)
        return "".join(top_section.render_xml() for top_section in top_sections) or EMPTY_BODY

    def _current_items(self) -> list[str]:
        """Content list of the innermost open section.

        Returns:
            Mutable list of markup items.
        """
        return self._section_stack[-1].content_items

    def _take_anchor_attribute(self) -> str:
        """Consume the pending page anchor.

        Returns:
            `` id="page_N"`` or an empty string.
        """
        anchor_attribute = f' id="{self._pending_anchor}"' if self._pending_anchor else ""
        self._pending_anchor = None
        return anchor_attribute

    def _render_paragraph(self, markdown_text: str, anchor_attribute: str = "") -> str:
        """Render one ``<p>`` element with the paragraph markup.

        Args:
            markdown_text: Paragraph text (Markdown inline).
            anchor_attribute: `` id=".."`` or an empty string.

        Returns:
            ``<p ...>...</p>`` markup.
        """
        language_attribute, paragraph_body = self._paragraph_markup.render_paragraph(markdown_text)
        return f"<p{anchor_attribute}{language_attribute}>{paragraph_body}</p>"

    def _remember_anchor(self, flow_item: dict) -> None:
        """Remember a page anchor for the next paragraph or caption.

        Args:
            flow_item: Page item.
        """
        self._pending_anchor = f"page_{flow_item['page']}"

    def _insert_visible_marker(self, flow_item: dict) -> None:
        """Insert a visible ``— N —`` page subtitle.

        Args:
            flow_item: Page item (uses the printed number when known).
        """
        page_label = flow_item["printed"] or flow_item["page"]
        subtitle_id = f"page_{flow_item['page']}"
        self._current_items().append(f'<subtitle id="{subtitle_id}">— {escape_text(page_label)} —</subtitle>')

    def _open_heading(self, flow_item: dict) -> None:
        """Close deeper or equal sections and open a new one.

        Args:
            flow_item: Heading item with ``level`` and ``text``.
        """
        while len(self._section_stack) > 1 and self._section_stack[-1].nesting_level >= flow_item["level"]:
            self._section_stack.pop()
        new_section = BookSection(flow_item["level"], self._render_paragraph(flow_item["text"]))
        self._section_stack[-1].child_sections.append(new_section)
        self._section_stack.append(new_section)

    def _add_paragraph(self, flow_item: dict) -> None:
        """Add a paragraph (abstracts wrapped in ``<cite>``).

        Args:
            flow_item: Paragraph item.
        """
        paragraph_xml = self._render_paragraph(flow_item["text"], self._take_anchor_attribute())
        self._current_items().append(f"<cite>{paragraph_xml}</cite>" if flow_item["style"] == "cite" else paragraph_xml)

    def _add_caption(self, flow_item: dict) -> None:
        """Add a caption as an emphasised paragraph.

        Args:
            flow_item: Caption item.
        """
        language_attribute, caption_body = self._paragraph_markup.render_paragraph(flow_item["text"])
        self._current_items().append(
            f"<p{self._take_anchor_attribute()}{language_attribute}><emphasis>{caption_body}</emphasis></p>")

    def _add_image(self, flow_item: dict) -> None:
        """Add an image reference.

        Args:
            flow_item: Image item with the binary ``id``.
        """
        self._current_items().append(f'<image l:href="#{flow_item["id"]}"/>')

    def _add_table(self, flow_item: dict) -> None:
        """Add a table.

        Args:
            flow_item: Table item with ``rows``.
        """
        self._current_items().append(render_fb2_table(flow_item["rows"]))


def render_body_sections(flow_items: list[dict], page_marker_mode: str,
                         paragraph_markup: ParagraphMarkup | None = None) -> str:
    """Render the whole flow as nested FB2 sections.

    Args:
        flow_items: Items from ``BookFlow.flow_items``.
        page_marker_mode: ``anchor``, ``visible`` or ``none``.
        paragraph_markup: Renders paragraph texts (None: Markdown conversion only).

    Returns:
        Section markup for the main body.
    """
    tree_builder = SectionTreeBuilder(page_marker_mode, paragraph_markup)
    for flow_item in flow_items:
        tree_builder.add_flow_item(flow_item)
    return tree_builder.render_body_xml()
