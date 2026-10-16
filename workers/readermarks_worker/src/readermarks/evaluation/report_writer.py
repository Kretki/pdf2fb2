"""Write the evaluation report as JSON and Markdown (STAGE_CLEAN.md §9)."""

import json
from pathlib import Path

ACCEPTANCE_TARGETS = (
    ("hand_underline precision (patch, synthetic test)", ">= 0.85"),
    ("hand_underline recall (patch, synthetic test)", ">= 0.80"),
    ("printed_rule false-positive rate", "< 1%"),
    ("page-level has_marks accuracy (real)", ">= 0.97"),
    ("unmarked page altered", "never"),
    ("full pipeline on marked page", "<= 2 s"),
    ("peak VRAM", "<= 10 GB"),
)


def format_value(metric_value: object) -> str:
    """Render a metric for a Markdown table.

    Args:
        metric_value: Number, None or other value.

    Returns:
        Text (``n/a`` for None).
    """
    if metric_value is None:
        return "n/a"
    if isinstance(metric_value, float):
        return f"{metric_value:.4f}"
    return str(metric_value)


def metrics_table(row_metrics: dict[str, dict], metric_names: tuple[str, ...]) -> list[str]:
    """Markdown table with one row per entry.

    Args:
        row_metrics: Row name -> metrics dict.
        metric_names: Columns.

    Returns:
        Markdown lines.
    """
    table_lines = ["| | " + " | ".join(metric_names) + " |", "|---" * (len(metric_names) + 1) + "|"]
    for row_name, row_values in row_metrics.items():
        table_lines.append(
            f"| {row_name} | "
            + " | ".join(format_value(row_values.get(metric_name)) for metric_name in metric_names)
            + " |"
        )
    return table_lines


def acceptance_lines(report_document: dict) -> list[str]:
    """Acceptance-target table filled from the report.

    Args:
        report_document: Full report dict.

    Returns:
        Markdown lines.
    """
    patch_classes = report_document.get("synthetic_patches", {}).get("classes", {})
    underline = patch_classes.get("hand_underline", {})
    real_pages = report_document.get("real_pages", {})
    measured = (
        underline.get("precision"),
        underline.get("recall"),
        report_document.get("synthetic_patches", {}).get("printed_rule_flagged_rate"),
        real_pages.get("page_level", {}).get("accuracy"),
        real_pages.get("unmarked_altered"),
        real_pages.get("timings", {}).get("marked_page_seconds"),
        real_pages.get("peak_vram_gb"),
    )
    table_lines = ["| Metric | Target | Measured |", "|---|---|---|"]
    for (metric_name, target_text), measured_value in zip(ACCEPTANCE_TARGETS, measured, strict=True):
        table_lines.append(f"| {metric_name} | {target_text} | {format_value(measured_value)} |")
    return table_lines


PAGE_LEVEL_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives",
)


def patch_section(report_document: dict) -> list[str]:
    """Patch-level table of the synthetic test split.

    Args:
        report_document: Full report dict.

    Returns:
        Markdown lines (empty when the section is absent).
    """
    if "synthetic_patches" not in report_document:
        return []
    patch_rows = {
        **report_document["synthetic_patches"]["classes"],
        "mark_any": report_document["synthetic_patches"]["mark_any"],
    }
    return ["## Patch level (synthetic test)", "", *metrics_table(patch_rows, ("precision", "recall", "f_score")), ""]


def pixel_section(report_document: dict) -> list[str]:
    """Pixel-level table of the synthetic test split.

    Args:
        report_document: Full report dict.

    Returns:
        Markdown lines (empty when the section is absent).
    """
    if "synthetic_pixels" not in report_document:
        return []
    pixel_rows = {
        pixel_class: report_document["synthetic_pixels"][pixel_class]
        for pixel_class in ("mark_any", "printed_ink_in_regions")
    }
    return [
        "## Pixel level (synthetic test)",
        "",
        *metrics_table(pixel_rows, ("iou", "f_score", "precision", "recall")),
        "",
    ]


def page_section(report_document: dict) -> list[str]:
    """Page-level table of the real pages.

    Args:
        report_document: Full report dict.

    Returns:
        Markdown lines (empty when the section is absent).
    """
    if "real_pages" not in report_document:
        return []
    page_rows = {"has_marks": report_document["real_pages"]["page_level"]}
    return ["## Page level (real pages)", "", *metrics_table(page_rows, PAGE_LEVEL_COLUMNS), ""]


def notes_section(report_document: dict) -> list[str]:
    """Bullet list of the report notes (skipped metrics, data caveats).

    Args:
        report_document: Full report dict.

    Returns:
        Markdown lines.
    """
    return ["## Notes", "", *[f"- {note_line}" for note_line in report_document.get("notes", [])], ""]


REPORT_SECTIONS = (patch_section, pixel_section, page_section, notes_section)


def render_markdown(report_document: dict) -> str:
    """Render the whole report as Markdown.

    Args:
        report_document: Full report dict.

    Returns:
        Markdown text.
    """
    markdown_lines = [
        f"# Reader-mark evaluation: {report_document['split']}",
        "",
        "## Acceptance targets",
        "",
        *acceptance_lines(report_document),
        "",
    ]
    for section_renderer in REPORT_SECTIONS:
        markdown_lines += section_renderer(report_document)
    return "\n".join(markdown_lines)


def write_report(report_document: dict, report_directory: Path) -> tuple[Path, Path]:
    """Write ``<split>.json`` and ``<split>.md``.

    Usage:
        Called at the end of ``python -m readermarks eval``.

    Args:
        report_document: Full report dict (must contain ``split``).
        report_directory: Output directory (created).

    Returns:
        ``(json_path, markdown_path)``.
    """
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / f"{report_document['split']}.json"
    markdown_path = report_directory / f"{report_document['split']}.md"
    json_path.write_text(json.dumps(report_document, indent=1), encoding="utf-8")
    markdown_path.write_text(render_markdown(report_document), encoding="utf-8")
    return json_path, markdown_path
