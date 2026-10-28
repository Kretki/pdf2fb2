"""Rendered training pages: reference text set in period fonts, printed to page images and degraded like a scan.

``pdf2fb2-bench render-references`` turns the reference chunks of a split (``references_<split>.jsonl``, which
holds only that split's books) into one image-only PDF per language plus a manifest with the exact words of
every page. The real pipeline then reads these PDFs (``run-rendered``) and ``export-rendered`` labels its output
against the known words: exact labels with the pipeline's own error distribution, from any language and script.
"""

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageEnhance, ImageFilter

from pdf2fb2.benchmark.rendered_fonts import available_fonts, text_script

RENDERED_PDF = "rendered.pdf"
MANIFEST_FILE = "manifest.json"
PAGE_SIZE = (420, 640)            # points, a small octavo page
MARGIN_POINTS = 36
RENDER_DPI = 200
FRAKTUR_SHARE = 0.4               # share of German pages set in Fraktur
CJK_SCRIPT = "cjk"


@dataclass(frozen=True)
class PageStyle:
    """Typesetting and degradation of one rendered page.

    Attributes:
        font_path: Font file.
        font_size: Points.
        line_height: Line height factor.
        blur_radius: Gaussian blur radius in pixels.
        noise_level: Standard deviation of pixel noise (0..255).
        rotation_degrees: Skew of the scan.
        jpeg_quality: Compression quality of the scan image.
    """

    font_path: Path
    font_size: float
    line_height: float
    blur_radius: float
    noise_level: float
    rotation_degrees: float
    jpeg_quality: int


def page_style(page_words: list[str], language_name: str, style_generator: random.Random) -> PageStyle | None:
    """Random style for a page in a font that can render its script.

    Args:
        page_words: Words of the page.
        language_name: Corpus language (German pages are sometimes set in Fraktur).
        style_generator: Random generator.

    Returns:
        The style, or None when no font covers the script.
    """
    script_name = text_script(" ".join(page_words))
    font_choices = available_fonts(script_name)
    fraktur_fonts = available_fonts("fraktur") if language_name == "german" else []
    if fraktur_fonts and style_generator.random() < FRAKTUR_SHARE:
        font_choices = fraktur_fonts
    if not font_choices:
        return None
    return PageStyle(style_generator.choice(font_choices), style_generator.uniform(9.0, 13.0),
                     style_generator.uniform(1.1, 1.5), style_generator.uniform(0.0, 1.1),
                     style_generator.uniform(0.0, 18.0), style_generator.uniform(-0.8, 0.8),
                     style_generator.randint(35, 90))


def fitting_text(text_page: pymupdf.Page, page_words: list[str], page_style_value: PageStyle,
                 word_joiner: str) -> int:
    """Insert as many words as fit the text box (binary search) and return their count.

    Args:
        text_page: Empty page to typeset on (mutated when text is inserted).
        page_words: Words to set.
        page_style_value: Font and size.
        word_joiner: ``" "`` (or ``""`` for CJK).

    Returns:
        Number of words set on the page (0 when not even one fits).
    """
    text_box = pymupdf.Rect(MARGIN_POINTS, MARGIN_POINTS, PAGE_SIZE[0] - MARGIN_POINTS, PAGE_SIZE[1] - MARGIN_POINTS)
    lower_count, upper_count = 0, len(page_words)
    while lower_count < upper_count:
        middle_count = (lower_count + upper_count + 1) // 2
        probe_page = pymupdf.open().new_page(width=PAGE_SIZE[0], height=PAGE_SIZE[1])
        spare_height = probe_page.insert_textbox(text_box, word_joiner.join(page_words[:middle_count]),
                                                 fontsize=page_style_value.font_size, fontfile=str(
                                                     page_style_value.font_path), fontname="F0",
                                                 lineheight=page_style_value.line_height)
        lower_count, upper_count = (middle_count, upper_count) if spare_height >= 0 else (lower_count,
                                                                                            middle_count - 1)
    if lower_count:
        text_page.insert_textbox(text_box, word_joiner.join(page_words[:lower_count]),
                                 fontsize=page_style_value.font_size, fontfile=str(page_style_value.font_path),
                                 fontname="F0", lineheight=page_style_value.line_height, align=3)
    return lower_count


def degraded_scan(text_page: pymupdf.Page, page_style_value: PageStyle, noise_generator: random.Random) -> bytes:
    """Raster image of a typeset page with blur, noise, skew, uneven contrast and JPEG artefacts.

    Args:
        text_page: Typeset page.
        page_style_value: Degradation settings.
        noise_generator: Random generator (seeds the noise).

    Returns:
        JPEG bytes.
    """
    page_pixmap = text_page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY)
    page_image = Image.frombytes("L", (page_pixmap.width, page_pixmap.height), page_pixmap.samples)
    page_image = page_image.rotate(page_style_value.rotation_degrees, resample=Image.Resampling.BICUBIC,
                                   fillcolor=255).filter(ImageFilter.GaussianBlur(page_style_value.blur_radius))
    page_image = ImageEnhance.Contrast(page_image).enhance(noise_generator.uniform(0.7, 1.1))
    noise_image = Image.effect_noise(page_image.size, max(page_style_value.noise_level, 0.1))
    page_image = Image.blend(page_image, noise_image, page_style_value.noise_level / 255.0)
    image_buffer = io.BytesIO()
    page_image.save(image_buffer, "JPEG", quality=page_style_value.jpeg_quality)
    return image_buffer.getvalue()


def rendered_page(scan_document: pymupdf.Document, page_words: list[str], language_name: str,
                  style_generator: random.Random) -> list[str]:
    """Typeset, degrade and append one page.

    Args:
        scan_document: Output document (a page is appended when words fit).
        page_words: Words to set (the page takes as many as fit).
        language_name: Corpus language.
        style_generator: Random generator.

    Returns:
        Words actually on the page (empty when nothing was rendered).
    """
    style_value = page_style(page_words, language_name, style_generator)
    if style_value is None:
        return []
    word_joiner = "" if text_script(" ".join(page_words)) == CJK_SCRIPT else " "
    text_document = pymupdf.open()
    text_page = text_document.new_page(width=PAGE_SIZE[0], height=PAGE_SIZE[1])
    set_count = fitting_text(text_page, page_words, style_value, word_joiner)
    if not set_count:
        return []
    scan_page = scan_document.new_page(width=PAGE_SIZE[0], height=PAGE_SIZE[1])
    scan_page.insert_image(scan_page.rect, stream=degraded_scan(text_page, style_value, style_generator))
    return page_words[:set_count]


def language_chunks(references_path: Path) -> dict[str, list[tuple[str, list[str]]]]:
    """Reference chunks grouped by language.

    Args:
        references_path: ``references_<split>.jsonl``.

    Returns:
        Language -> ``(book key, words)`` chunks.
    """
    chunks_by_language: dict[str, list[tuple[str, list[str]]]] = {}
    with references_path.open(encoding="utf-8") as references_file:
        for line_text in references_file:
            if line_text.strip():
                line_object = json.loads(line_text)
                chunks_by_language.setdefault(str(line_object["language"]), []).append(
                    (str(line_object["book_key"]), [str(word_text) for word_text in line_object["words"]]))
    return chunks_by_language


def render_language(language_chunks_list: list[tuple[str, list[str]]], language_name: str, output_directory: Path,
                    page_limit: int, style_generator: random.Random) -> int:
    """Render up to ``page_limit`` pages of one language with its manifest.

    Args:
        language_chunks_list: ``(book key, words)`` chunks.
        language_name: Corpus language.
        output_directory: ``rendered_<split>/<language>``.
        page_limit: Largest number of pages.
        style_generator: Random generator.

    Returns:
        Pages rendered.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    scan_document = pymupdf.open()
    manifest_pages = []
    for book_key, chunk_words in style_generator.sample(language_chunks_list, len(language_chunks_list)):
        if len(manifest_pages) >= page_limit:
            break
        page_words = rendered_page(scan_document, chunk_words, language_name, style_generator)
        if page_words:
            manifest_pages.append({"book_key": book_key, "words": page_words})
    if manifest_pages:
        scan_document.save(output_directory / RENDERED_PDF)
    (output_directory / MANIFEST_FILE).write_text(json.dumps({"language": language_name, "pages": manifest_pages},
                                                             ensure_ascii=False), encoding="utf-8")
    return len(manifest_pages)


def render_references(data_directory: Path, split_name: str, page_limit: int, random_seed: int = 1234) -> Path:
    """Render the reference chunks of a split, one PDF per language.

    Usage:
        Called by ``pdf2fb2-bench render-references`` after ``export-references`` for that split.

    Args:
        data_directory: ``benchmark_data``.
        split_name: Split whose references are rendered (``train`` for training data).
        page_limit: Pages per language.
        random_seed: Seed of styles and degradation.

    Returns:
        ``rendered_<split>`` directory.

    Raises:
        FileNotFoundError: If the reference export is missing.
    """
    references_path = data_directory / f"references_{split_name}.jsonl"
    if not references_path.exists():
        raise FileNotFoundError(f"{references_path} is missing; run export-references first.")
    style_generator = random.Random(random_seed)  # nosec B311  # noqa: S311 - augmentation, not cryptography
    rendered_root = data_directory / f"rendered_{split_name}"
    for language_name, chunk_list in sorted(language_chunks(references_path).items()):
        page_count = render_language(chunk_list, language_name, rendered_root / language_name, page_limit,
                                     style_generator)
        print(f"rendered {language_name}: {page_count} pages", flush=True)
    return rendered_root
