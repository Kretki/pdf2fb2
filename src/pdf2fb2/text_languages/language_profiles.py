"""Built-in language profiles: the script a language is written in and its most frequent short words.

Languages sharing a script (English, Latin, German, ... or Russian, Ukrainian) are told apart by counting their
frequent function words, which needs no trained model and works on a single sentence. Japanese is Han with kana;
Chinese is Han without kana. A new language is supported by adding one profile here.
"""

from dataclasses import dataclass

from pdf2fb2.text_languages.script_detection import (
    ARABIC_SCRIPT,
    CYRILLIC_SCRIPT,
    DEVANAGARI_SCRIPT,
    GREEK_SCRIPT,
    HAN_SCRIPT,
    HANGUL_SCRIPT,
    HEBREW_SCRIPT,
    KANA_SCRIPT,
    LATIN_SCRIPT,
    THAI_SCRIPT,
)

AUTO_LANGUAGE = "auto"  # main language chosen from the text, among all built-in languages


@dataclass(frozen=True)
class LanguageProfile:
    """How a language is recognised.

    Attributes:
        language_code: ISO 639-1 code used in FB2 ``xml:lang``, e.g. ``en``.
        script_names: Scripts the language is written in (Japanese: Han and kana).
        stop_words: Frequent lowercase words, used to tell languages of one script apart (may be empty).
    """

    language_code: str
    script_names: frozenset[str]
    stop_words: frozenset[str] = frozenset()


def build_profile(language_code: str, script_names: tuple[str, ...], stop_text: str = "") -> LanguageProfile:
    """Create a profile from a space-separated word list.

    Args:
        language_code: ISO 639-1 code.
        script_names: Scripts of the language.
        stop_text: Frequent words separated by spaces.

    Returns:
        The profile.
    """
    return LanguageProfile(language_code, frozenset(script_names), frozenset(stop_text.split()))


KNOWN_PROFILES = {language_profile.language_code: language_profile for language_profile in (
    build_profile("en", (LATIN_SCRIPT,), "the of and to in is that it was for with as his he be by on not this "
                  "are which from at have but they you we i a an or no she her had were all there their said would "
                  "so its my me him been has what when who will"),
    build_profile("la", (LATIN_SCRIPT,), "et in est non ad cum quod qui quae sed ut ex de enim esse sunt nec "
                  "aut atque per hoc quam vel ab tamen etiam"),
    build_profile("de", (LATIN_SCRIPT,), "der die und das ist nicht mit den von zu ein eine sich auf des dem im "
                  "auch es wie aber"),
    build_profile("fr", (LATIN_SCRIPT,), "le la les et des est une un du que qui dans pour pas ne sur au avec "
                  "il elle sont ce"),
    build_profile("es", (LATIN_SCRIPT,), "el la los las y que de en un una es por con para del al lo se no su "
                  "como pero"),
    build_profile("it", (LATIN_SCRIPT,), "il la di che e un una per non sono gli le del della nel con si ma "
                  "come anche"),
    build_profile("pt", (LATIN_SCRIPT,), "o a os as e que de em um uma do da dos das para com não se por mais"),
    build_profile("pl", (LATIN_SCRIPT,), "i w na nie się z że do to jest jak o po ale od za co tak"),
    build_profile("ru", (CYRILLIC_SCRIPT,), "и в не на что с он как это по но из у за от то же для так все был "
                  "она его бы"),
    build_profile("uk", (CYRILLIC_SCRIPT,), "і в не на що з він як це по але із у за від та ж для так всі був "
                  "вона його"),
    build_profile("el", (GREEK_SCRIPT,)),
    build_profile("zh", (HAN_SCRIPT,)),
    build_profile("ja", (HAN_SCRIPT, KANA_SCRIPT)),
    build_profile("ko", (HANGUL_SCRIPT, HAN_SCRIPT)),
    build_profile("ar", (ARABIC_SCRIPT,)),
    build_profile("he", (HEBREW_SCRIPT,)),
    build_profile("hi", (DEVANAGARI_SCRIPT,)),
    build_profile("th", (THAI_SCRIPT,)),
)}


def primary_subtag(language_code: str) -> str:
    """Language part of a BCP-47 code.

    Args:
        language_code: Code such as ``pt-BR`` or ``en``.

    Returns:
        The lowercase part before the first ``-``.
    """
    return language_code.split("-", 1)[0].lower()


def find_profile(language_code: str) -> LanguageProfile | None:
    """Built-in profile of a language code (region subtags are ignored).

    Args:
        language_code: Code such as ``ja`` or ``pt-BR``.

    Returns:
        The profile carrying the given code, or None when the language is not built in.
    """
    known_profile = KNOWN_PROFILES.get(primary_subtag(language_code))
    if known_profile is None:
        return None
    return LanguageProfile(language_code, known_profile.script_names, known_profile.stop_words)


def resolve_book_languages(language_codes: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Main language and candidate languages of a book from the ``--lang`` codes.

    Args:
        language_codes: Validated codes, main first, or ``("auto",)``.

    Returns:
        ``(main_language, book_languages)``: for ``auto`` the main language stays ``auto`` and every built-in
        language is a candidate; one code gives no candidates (single-language book); several codes are
        the candidates themselves.
    """
    if language_codes == (AUTO_LANGUAGE,):
        return AUTO_LANGUAGE, tuple(KNOWN_PROFILES)
    return language_codes[0], language_codes if len(language_codes) > 1 else ()
