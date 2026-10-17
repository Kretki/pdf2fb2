"""Language of a text among the languages declared for the book.

A declared language scores the share of the text's letters written in its scripts; ties (languages of one
script) are broken by how many of the language's frequent words the text contains, then by the narrower
profile (Chinese before Japanese for Han-only text), then by the order the languages were declared in. Function
words count only from two hits on, so a short line sharing one word with another language keeps the main one.
"""

import re

from pdf2fb2.text_languages.language_profiles import LanguageProfile
from pdf2fb2.text_languages.script_detection import count_scripts

WORD_PATTERN = re.compile(r"[^\W\d_]+")
MINIMUM_LETTERS = 2  # fewer letters (a number, an initial) give no language
MINIMUM_SCRIPT_SHARE = 0.5  # share of the letters a language's scripts must cover
MINIMUM_STOP_WORD_HITS = 2  # one shared short word ("no", "i") is no evidence for a language


def stop_word_evidence(lowercase_words: list[str], language_profile: LanguageProfile) -> int:
    """How strongly a text's function words point to a language.

    Args:
        lowercase_words: Words of the text in lowercase.
        language_profile: Candidate language.

    Returns:
        Number of the text's words among the language's frequent words, or 0 below ``MINIMUM_STOP_WORD_HITS``.
    """
    stop_word_hits = sum(word_text in language_profile.stop_words for word_text in lowercase_words)
    return stop_word_hits if stop_word_hits >= MINIMUM_STOP_WORD_HITS else 0


class LanguageDetector:
    """Picks the most likely declared language of a text.

    Usage:
        Build once with the book's language profiles (declaration order matters for ties) and call
        ``detect_language`` for every paragraph or text run.
    """

    def __init__(self, language_profiles: tuple[LanguageProfile, ...]) -> None:
        """Store the candidate languages.

        Args:
            language_profiles: Profiles of the declared languages, main language first.
        """
        self._language_profiles = language_profiles

    @property
    def language_profiles(self) -> tuple[LanguageProfile, ...]:
        """Candidate languages.

        Returns:
            The profiles given at construction, in declaration order.
        """
        return self._language_profiles

    def find_profile(self, language_code: str) -> LanguageProfile | None:
        """Declared profile of a language code.

        Args:
            language_code: Code as declared.

        Returns:
            The profile, or None when the language was not declared.
        """
        return next((language_profile for language_profile in self._language_profiles
                     if language_profile.language_code == language_code), None)

    def detect_language(self, plain_text: str) -> str | None:
        """Language of a text.

        Args:
            plain_text: Text without markup.

        Returns:
            Code of the best declared language, or None when the text has fewer than ``MINIMUM_LETTERS``
            letters or no declared language covers half of them.
        """
        script_counts = count_scripts(plain_text)
        letter_count = sum(script_counts.values())
        if letter_count < MINIMUM_LETTERS:
            return None
        lowercase_words = [word_text.lower() for word_text in WORD_PATTERN.findall(plain_text)]
        ranked_profiles = [
            (sum(script_counts[script_name] for script_name in language_profile.script_names),
             stop_word_evidence(lowercase_words, language_profile),
             -len(language_profile.script_names), -profile_index, language_profile.language_code)
            for profile_index, language_profile in enumerate(self._language_profiles)]
        best_rank = max(ranked_profiles, default=None)
        if best_rank is None or best_rank[0] < MINIMUM_SCRIPT_SHARE * letter_count:
            return None
        return best_rank[-1]

    def choose_main_language(self, plain_texts: list[str], fallback_language: str) -> str:
        """Language with the most letters over a whole book.

        Usage:
            Call with every paragraph when the main language is ``auto``.

        Args:
            plain_texts: Texts without markup.
            fallback_language: Returned when no text has a detectable language.

        Returns:
            Code of the language covering the most letters.
        """
        letters_by_language: dict[str, int] = {}
        for plain_text in plain_texts:
            detected_language = self.detect_language(plain_text)
            if detected_language is not None:
                letters_by_language[detected_language] = (letters_by_language.get(detected_language, 0)
                                                          + sum(count_scripts(plain_text).values()))
        return max(letters_by_language, key=letters_by_language.__getitem__, default=fallback_language)
