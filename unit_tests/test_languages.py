"""Unit tests of script and language detection, CJK-aware joining and multilingual FB2 markup."""

import argparse
from xml.etree import ElementTree  # nosec B405  # noqa: S405

import pytest

from pdf2fb2.fb2_output.paragraph_flow import merge_or_append_paragraph
from pdf2fb2.fb2_output.paragraph_markup import PlainParagraphMarkup
from pdf2fb2.fb2_output.section_tree import render_body_sections
from pdf2fb2.fb2_output.text_markup import join_text_fragments
from pdf2fb2.interface.input_validation import parse_language_list
from pdf2fb2.text_languages.language_detection import LanguageDetector
from pdf2fb2.text_languages.language_markup import LanguageParagraphMarkup, build_paragraph_markup
from pdf2fb2.text_languages.language_profiles import AUTO_LANGUAGE, find_profile, resolve_book_languages
from pdf2fb2.text_languages.script_detection import (
    HAN_SCRIPT,
    KANA_SCRIPT,
    LATIN_SCRIPT,
    SPACELESS_LETTER_WEIGHT,
    count_scripts,
)

JAPANESE_TEXT = "これは日本語の文章です。"
CHINESE_TEXT = "这是中文的句子。"
ENGLISH_QUOTE = "To be or not to be, that is the question"
LATIN_QUOTE = "Cogito ergo sum, et non est quod timeas"
XML_NAMESPACE = "{http://www.w3.org/XML/1998/namespace}lang"


def build_detector(*language_codes: str) -> LanguageDetector:
    """Detector over built-in profiles.

    Args:
        language_codes: Declared languages, main first.

    Returns:
        The detector.
    """
    return LanguageDetector(tuple(find_profile(language_code) for language_code in language_codes))


def parse_fragment(inline_markup: str) -> ElementTree.Element:
    """Parse inline FB2 markup inside a wrapper element.

    Args:
        inline_markup: Markup to check.

    Returns:
        The wrapper element (raises ``ParseError`` when the markup is not well-formed).
    """
    return ElementTree.fromstring(f"<wrapper>{inline_markup}</wrapper>")  # nosec B314  # noqa: S314


def test_scripts_are_counted_per_letter() -> None:
    """Han, kana and Latin letters are counted separately (CJK weighted); punctuation is not counted."""
    script_counts = count_scripts("日本語のABC。")
    assert script_counts[HAN_SCRIPT] == 3 * SPACELESS_LETTER_WEIGHT
    assert script_counts[KANA_SCRIPT] == SPACELESS_LETTER_WEIGHT
    assert script_counts[LATIN_SCRIPT] == 3


def test_detector_tells_languages_of_one_script_apart() -> None:
    """Kana decides Japanese, Han alone Chinese, function words English versus Latin."""
    language_detector = build_detector("ja", "zh", "en", "la", "ru")
    assert language_detector.detect_language(JAPANESE_TEXT) == "ja"
    assert language_detector.detect_language(CHINESE_TEXT) == "zh"
    assert language_detector.detect_language(ENGLISH_QUOTE) == "en"
    assert language_detector.detect_language(LATIN_QUOTE) == "la"
    assert language_detector.detect_language("Это русский текст и он не короткий") == "ru"
    assert language_detector.detect_language("12.") is None


def test_cjk_lines_join_without_space() -> None:
    """Lines of Chinese or Japanese join directly; other scripts keep the space and hyphen handling."""
    assert join_text_fragments("これは日本", "語の文章です。") == "これは日本語の文章です。"
    assert join_text_fragments("first line", "second line") == "first line second line"
    assert join_text_fragments("hyphen-", "ated") == "hyphenated"


def test_cjk_paragraph_continues_across_pages() -> None:
    """A Japanese paragraph broken mid-sentence by a page is joined without a space."""
    flow_items = []
    merge_or_append_paragraph(flow_items, "これは日本", 1, "p", False)
    merge_or_append_paragraph(flow_items, "語の文章です。", 2, "p", False)
    assert [flow_item["text"] for flow_item in flow_items] == ["これは日本語の文章です。"]
    merge_or_append_paragraph(flow_items, "次の段落。", 2, "p", False)
    assert len(flow_items) == 2


def test_foreign_paragraphs_and_runs_get_language() -> None:
    """English paragraphs in a Japanese book get xml:lang; an English quotation inside Japanese text is wrapped."""
    paragraph_markup = LanguageParagraphMarkup(build_detector("ja", "en", "la"), "ja")
    assert paragraph_markup.render_paragraph(JAPANESE_TEXT)[0] == ""
    assert paragraph_markup.render_paragraph(ENGLISH_QUOTE)[0] == ' xml:lang="en"'
    assert paragraph_markup.render_paragraph(LATIN_QUOTE)[0] == ' xml:lang="la"'
    attribute_text, inline_markup = paragraph_markup.render_paragraph(
        f"彼は舞台の上で「*{ENGLISH_QUOTE}*」と静かに言った。&")
    assert attribute_text == ""
    style_elements = parse_fragment(inline_markup).iter("style")
    assert [(style_element.get(XML_NAMESPACE), style_element.text) for style_element in style_elements] == [
        ("en", ENGLISH_QUOTE)]


def test_short_foreign_words_are_not_wrapped() -> None:
    """A single Latin word (a name) in Russian text keeps no language markup."""
    paragraph_markup = LanguageParagraphMarkup(build_detector("ru", "en"), "ru")
    assert "<style" not in paragraph_markup.render_paragraph("Звезда телеканала CBS и частый гость шоу")[1]


def test_auto_language_takes_the_language_with_most_text() -> None:
    """``auto`` resolves the main language from the paragraphs; one declared language means plain markup."""
    main_language, book_languages = resolve_book_languages((AUTO_LANGUAGE,))
    paragraph_markup = build_paragraph_markup(book_languages, main_language,
                                              [JAPANESE_TEXT * 5, ENGLISH_QUOTE])
    assert paragraph_markup.main_language == "ja"
    assert isinstance(build_paragraph_markup((), "en", [ENGLISH_QUOTE]), PlainParagraphMarkup)


def test_sections_render_language_attributes() -> None:
    """Headings and paragraphs carry xml:lang in the rendered, well-formed body."""
    paragraph_markup = LanguageParagraphMarkup(build_detector("ja", "en"), "ja")
    body_markup = render_body_sections([
        {"kind": "heading", "level": 1, "text": "First Chapter of the Book"},
        {"kind": "p", "text": JAPANESE_TEXT, "style": "p", "list": False, "pages": {1}},
        {"kind": "p", "text": ENGLISH_QUOTE, "style": "cite", "list": False, "pages": {1}}],
        "none", paragraph_markup)
    paragraph_languages = [paragraph_element.get(XML_NAMESPACE)
                           for paragraph_element in parse_fragment(body_markup).iter("p")]
    assert paragraph_languages == ["en", None, "en"]


def test_language_list_validation() -> None:
    """Lists must hold known, distinct codes; a single unknown code is still accepted."""
    assert parse_language_list("ja,en,la") == ("ja", "en", "la")
    assert parse_language_list("AUTO") == (AUTO_LANGUAGE,)
    assert parse_language_list("eo") == ("eo",)
    assert resolve_book_languages(("en",)) == ("en", ())
    for invalid_text in ("en,en", "en,eo", "en,,ja"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_language_list(invalid_text)
