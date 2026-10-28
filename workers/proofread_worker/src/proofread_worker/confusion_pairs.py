"""Script look-alikes: Latin and Cyrillic letters of the same shape mixed inside one word.

The confusion pairs themselves live in ``worker_protocol.ocr_confusions`` (shared with the verifier worker).
"""

CYRILLIC_TO_LATIN = dict(zip("аеорсхуАВЕКМНОРСТХ", "aeopcxyABEKMHOPCTX", strict=True))
LATIN_TO_CYRILLIC = {latin_letter: cyrillic_letter for cyrillic_letter, latin_letter in CYRILLIC_TO_LATIN.items()}


def script_lookalikes(word_core: str) -> str | None:
    """Rewrite the minority-script look-alike letters of a word mixing Latin and Cyrillic.

    Args:
        word_core: Word without surrounding punctuation.

    Returns:
        The word in its majority script, or None when the word does not mix the two scripts.
    """
    cyrillic_count = sum(1 for word_letter in word_core if "Ѐ" <= word_letter <= "ӿ")
    latin_count = sum(1 for word_letter in word_core if word_letter.isascii() and word_letter.isalpha())
    if not cyrillic_count or not latin_count:
        return None
    letter_mapping = LATIN_TO_CYRILLIC if cyrillic_count >= latin_count else CYRILLIC_TO_LATIN
    rewritten_word = "".join(letter_mapping.get(word_letter, word_letter) for word_letter in word_core)
    return rewritten_word if rewritten_word != word_core else None
