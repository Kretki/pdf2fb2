"""Parsing OCR tables (HTML or Markdown) and rendering them as FB2 tables."""

import re
from html.parser import HTMLParser

from pdf2fb2.text_markup import markdown_to_fb2_inline

SEPARATOR_CELL_PATTERN = re.compile(r":?-{2,}:?")
SPAN_ATTRIBUTES = ("colspan", "rowspan")


class TableHtmlParser(HTMLParser):
    """Collects rows and cells (with colspan/rowspan) from an HTML table.

    Usage:
        ``feed`` the OCR HTML, then read ``table_rows``. Each cell is
        ``{"th": bool, "text": str, "colspan": str | None, "rowspan": str | None}``.
    """

    def __init__(self) -> None:
        """Initialise an empty parser state."""
        super().__init__()
        self.table_rows: list[list[dict]] = []
        self.current_row: list[dict] | None = None
        self.current_cell: dict | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Open rows/cells and turn ``<br>`` into spaces.

        Args:
            tag: Lower-case tag name (HTMLParser API).
            attrs: Tag attributes (HTMLParser API).
        """
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            if self.current_row is None:
                self.current_row = []
            attribute_map = dict(attrs)
            self.current_cell = {"th": tag == "th", "text": "", "colspan": attribute_map.get("colspan"),
                                 "rowspan": attribute_map.get("rowspan")}
        elif tag == "br" and self.current_cell is not None:
            self.current_cell["text"] += " "

    def handle_endtag(self, tag: str) -> None:
        """Close cells and rows.

        Args:
            tag: Lower-case tag name (HTMLParser API).
        """
        if tag in ("td", "th") and self.current_cell is not None and self.current_row is not None:
            self.current_cell["text"] = self.current_cell["text"].strip()
            self.current_row.append(self.current_cell)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.table_rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        """Append text to the open cell.

        Args:
            data: Text content (HTMLParser API).
        """
        if self.current_cell is not None:
            self.current_cell["text"] += data


def _parse_html_table(table_text: str) -> list[list[dict]] | None:
    """Parse an HTML table.

    Args:
        table_text: OCR output containing ``<table>``.

    Returns:
        Rows of cells, or None when no row was found.
    """
    table_parser = TableHtmlParser()
    table_parser.feed(table_text)
    return table_parser.table_rows or None


def _is_separator_row(cell_texts: list[str]) -> bool:
    """Tell whether a Markdown table row is the ``|---|---|`` header separator.

    Args:
        cell_texts: Stripped cell texts of the row.

    Returns:
        True if every non-empty cell consists of dashes (and optional colons).
    """
    return all(SEPARATOR_CELL_PATTERN.fullmatch(cell_text) for cell_text in cell_texts if cell_text)


def _parse_markdown_table(table_text: str) -> list[list[dict]] | None:
    """Parse a pipe-delimited Markdown table; the row above the separator becomes the header.

    Args:
        table_text: OCR output with ``|``-prefixed lines.

    Returns:
        Rows of cells, or None when fewer than two table lines exist.
    """
    table_lines = [line_text.strip() for line_text in table_text.splitlines() if line_text.strip().startswith("|")]
    if len(table_lines) < 2:
        return None
    table_rows: list[list[dict]] = []
    header_marked = False
    for line_text in table_lines:
        cell_texts = [cell_text.strip() for cell_text in line_text.strip("|").split("|")]
        if _is_separator_row(cell_texts):
            if table_rows and not header_marked:
                for header_cell in table_rows[-1]:
                    header_cell["th"] = True
                header_marked = True
            continue
        table_rows.append([{"th": False, "text": cell_text} for cell_text in cell_texts])
    return table_rows or None


def parse_table_text(table_text: str) -> list[list[dict]] | None:
    """Parse OCR table output in HTML or Markdown form.

    Usage:
        Call for every ``table`` segment; None means the table should be embedded as a picture.

    Args:
        table_text: OCR output of the table segment.

    Returns:
        Rows of cells, or None if no table could be parsed.
    """
    if "<table" in table_text.lower():
        return _parse_html_table(table_text)
    return _parse_markdown_table(table_text)


def _span_attributes(table_cell: dict) -> str:
    """Render colspan/rowspan attributes greater than 1.

    Args:
        table_cell: Parsed cell.

    Returns:
        Attribute string such as `` colspan="2"`` (may be empty).
    """
    return "".join(f' {attribute_name}="{int(table_cell[attribute_name])}"' for attribute_name in SPAN_ATTRIBUTES
                   if str(table_cell.get(attribute_name) or "").isdigit() and int(table_cell[attribute_name]) > 1)


def render_fb2_table(table_rows: list[list[dict]]) -> str:
    """Render parsed rows as an FB2 ``<table>``.

    Args:
        table_rows: Rows from ``parse_table_text``.

    Returns:
        FB2 table markup.
    """
    markup_parts = ["<table>"]
    for table_row in table_rows:
        markup_parts.append("<tr>")
        for table_cell in table_row:
            cell_tag = "th" if table_cell.get("th") else "td"
            cell_body = markdown_to_fb2_inline(table_cell["text"])
            markup_parts.append(f"<{cell_tag}{_span_attributes(table_cell)}>{cell_body}</{cell_tag}>")
        markup_parts.append("</tr>")
    markup_parts.append("</table>")
    return "".join(markup_parts)
