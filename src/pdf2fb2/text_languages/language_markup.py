"""Multilingual FB2 markup: ``xml:lang`` on paragraphs and on foreign runs inside them.

A paragraph whose detected language differs from the book's main language gets ``<p xml:lang="..">``. Inside a
paragraph, a run of letters in a script the paragraph's language does not use (an English quotation in Japanese
text, a Latin line in Russian text) is wrapped in ``<style name="foreign" xml:lang="..">`` when it is long enough
(two words, or two characters of a script written without spaces) and its language was declared. Runs are found
in the text nodes of the converted markup, so tags and entities are never split.
"""

import html
import re

from pdf2fb2.fb2_output.paragraph_markup import ParagraphMarkup, PlainParagraphMarkup
from pdf2fb2.fb2_output.text_markup import markdown_to_fb2_inline
from pdf2fb2.text_languages.language_detection import WORD_PATTERN, LanguageDetector
from pdf2fb2.text_languages.language_profiles import AUTO_LANGUAGE, find_profile
from pdf2fb2.text_languages.script_detection import CJK_PUNCTUATION, SPACELESS_SCRIPTS, character_script

MARKUP_PIECE_PATTERN = re.compile(r"(<[^>]*>|&#?\w+;)")  # tags and entities: never inside a run
FOREIGN_STYLE_NAME = "foreign"
MINIMUM_RUN_WORDS = 2
MINIMUM_SPACELESS_LETTERS = 2
ScriptRun = tuple[int, int, str]  # (start, end, script) of a run inside one text node


def plain_text_of(inline_markup: str) -> str:
    """Visible text of FB2 inline markup.

    Args:
        inline_markup: Well-formed inline markup.

    Returns:
        The text with tags removed and entities decoded.
    """
    return html.unescape(re.sub(r"<[^>]*>", "", inline_markup))


def find_script_runs(text_node: str) -> list[ScriptRun]:
    """Split a text node into runs of letters of one script.

    Args:
        text_node: Text without tags or entities.

    Returns:
        ``(start, end, script)`` runs from the first to the last letter of each script change; spaces, digits
        and punctuation between letters of one script stay inside the run.
    """
    script_runs: list[ScriptRun] = []
    for character_index, text_character in enumerate(text_node):
        script_name = character_script(text_character)
        if script_name is None or script_name == CJK_PUNCTUATION:
            continue
        if script_runs and script_runs[-1][2] == script_name:
            script_runs[-1] = (script_runs[-1][0], character_index + 1, script_name)
        else:
            script_runs.append((character_index, character_index + 1, script_name))
    return script_runs


def is_long_run(run_text: str, script_name: str) -> bool:
    """Tell whether a run is long enough to carry a language of its own.

    Args:
        run_text: Text of the run.
        script_name: Its script.

    Returns:
        True for two or more words, or two or more characters of a spaceless script.
    """
    if script_name in SPACELESS_SCRIPTS:
        return len(run_text) >= MINIMUM_SPACELESS_LETTERS
    return len(WORD_PATTERN.findall(run_text)) >= MINIMUM_RUN_WORDS


class LanguageParagraphMarkup:
    """``ParagraphMarkup`` that marks the language of paragraphs and foreign runs.

    Usage:
        Built by ``build_paragraph_markup`` for books declared with several languages (or ``auto``).
    """

    def __init__(self, language_detector: LanguageDetector, main_language: str) -> None:
        """Store the detector and the book language.

        Args:
            language_detector: Detects the declared languages.
            main_language: Language code of the book (paragraphs in it get no attribute).
        """
        self._language_detector = language_detector
        self._main_language = main_language

    @property
    def main_language(self) -> str:
        """Language code of the book.

        Returns:
            The code given at construction.
        """
        return self._main_language

    def render_paragraph(self, markdown_text: str) -> tuple[str, str]:
        """Convert a paragraph and mark its languages.

        Args:
            markdown_text: OCR Markdown inline text.

        Returns:
            ``(attribute_text, inline_markup)``; the attribute is `` xml:lang=".."`` for paragraphs not in the
            main language.
        """
        inline_markup = markdown_to_fb2_inline(markdown_text)
        paragraph_language = self._language_detector.detect_language(plain_text_of(inline_markup))
        effective_language = paragraph_language or self._main_language
        attribute_text = "" if effective_language == self._main_language else f' xml:lang="{effective_language}"'
        markup_pieces = MARKUP_PIECE_PATTERN.split(inline_markup)
        return attribute_text, "".join(
            markup_piece if piece_index % 2 else self._mark_foreign_runs(markup_piece, effective_language)
            for piece_index, markup_piece in enumerate(markup_pieces))

    def _mark_foreign_runs(self, text_node: str, paragraph_language: str) -> str:
        """Wrap the foreign runs of one text node.

        Args:
            text_node: Text between tags (already XML-escaped, without entities).
            paragraph_language: Language of the paragraph.

        Returns:
            The text with ``<style name="foreign" xml:lang="..">`` around each foreign run.
        """
        paragraph_profile = self._language_detector.find_profile(paragraph_language) or find_profile(
            paragraph_language)
        native_scripts = paragraph_profile.script_names if paragraph_profile is not None else frozenset()
        marked_parts, copied_until = [], 0
        for segment_start, segment_end, script_name in find_script_runs(text_node):
            run_text = text_node[segment_start:segment_end]
            if script_name in native_scripts or not is_long_run(run_text, script_name):
                continue
            run_language = self._language_detector.detect_language(run_text)
            if run_language is None or run_language == paragraph_language:
                continue
            marked_parts += [text_node[copied_until:segment_start],
                             f'<style name="{FOREIGN_STYLE_NAME}" xml:lang="{run_language}">{run_text}</style>']
            copied_until = segment_end
        return "".join(marked_parts) + text_node[copied_until:]


def build_paragraph_markup(book_languages: tuple[str, ...], main_language: str,
                           paragraph_texts: list[str]) -> ParagraphMarkup:
    """Choose the paragraph markup of a book.

    Usage:
        Call once per book before rendering its sections.

    Args:
        book_languages: Declared languages in order (empty or one entry: single-language book).
        main_language: Main language code, or ``auto`` to take the language with the most letters.
        paragraph_texts: Texts of the book's paragraphs, titles and notes (used for ``auto``).

    Returns:
        ``PlainParagraphMarkup`` for single-language books, else ``LanguageParagraphMarkup``.
    """
    language_profiles = tuple(filter(None, (find_profile(language_code) for language_code in book_languages)))
    if len(language_profiles) < 2:
        fallback_language = language_profiles[0].language_code if language_profiles else "en"
        return PlainParagraphMarkup(fallback_language if main_language == AUTO_LANGUAGE else main_language)
    language_detector = LanguageDetector(language_profiles)
    if main_language == AUTO_LANGUAGE:
        main_language = language_detector.choose_main_language(
            [plain_text_of(paragraph_text) for paragraph_text in paragraph_texts], language_profiles[0].language_code)
    return LanguageParagraphMarkup(language_detector, main_language)
