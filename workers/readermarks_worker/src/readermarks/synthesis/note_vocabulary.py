"""Text of synthetic handwritten notes: real words, pseudo-words, numbers and reader remarks."""

import numpy

REMARK_WORDS = (
    "not true",
    "see p.",
    "cf.",
    "why?",
    "NB",
    "yes",
    "no!",
    "important",
    "compare",
    "sic",
    "!!",
    "wrong",
    "ok",
    "note",
    "myth",
    "fact?",
    "true",
    "source?",
    "Ridge",
    "legend",
    "California",
    "bandit",
    "gold",
    "1853",
    "p. 42",
    "ibid.",
    "hero",
    "check",
    "cited",
    "?",
    "x",
    "see above",
)
SYLLABLES = (
    "ka",
    "ri",
    "mo",
    "ten",
    "sa",
    "lo",
    "ver",
    "di",
    "na",
    "mer",
    "to",
    "an",
    "el",
    "is",
    "tor",
    "pe",
    "on",
    "ra",
    "li",
    "que",
    "ber",
    "st",
    "ch",
    "um",
    "ex",
    "ho",
    "ga",
    "ne",
    "wi",
    "ly",
)


def pseudo_word(random_source: numpy.random.Generator) -> str:
    """Build a pronounceable pseudo-word from 1..4 syllables.

    Args:
        random_source: Random generator.

    Returns:
        Lower-case word.
    """
    syllable_count = int(random_source.integers(1, 5))
    return "".join(SYLLABLES[random_source.integers(len(SYLLABLES))] for _ in range(syllable_count))


def note_text(random_source: numpy.random.Generator, maximum_words: int) -> str:
    """Compose a short reader note.

    Usage:
        Called for every synthetic handwritten note (inline or margin).

    Args:
        random_source: Random generator.
        maximum_words: Upper bound on the number of words (>= 1).

    Returns:
        ASCII text; sometimes upper-case (block letters) or capitalised.
    """
    word_count = int(random_source.integers(1, maximum_words + 1))
    note_words = [
        REMARK_WORDS[random_source.integers(len(REMARK_WORDS))]
        if random_source.random() < 0.4
        else pseudo_word(random_source)
        for _ in range(word_count)
    ]
    composed_text = " ".join(note_words)
    case_draw = random_source.random()
    if case_draw < 0.45:
        return composed_text.upper()
    if case_draw < 0.6:
        return composed_text.capitalize()
    return composed_text
