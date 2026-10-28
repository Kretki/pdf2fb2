"""OCR confusion pairs shared by the workers: letter groups a scan reader typically mistakes for each other.

The proofread worker uses them to propose respellings of low-probability words; the verifier worker uses them to
synthesise realistic misreadings. Every pair works both ways (``rn`` read for ``m`` and ``m`` read for ``rn``);
upper-case forms are derived by ``directed_pairs``.
"""

CONFUSION_GROUPS: tuple[tuple[str, str], ...] = (
    # Cyrillic
    ("с", "е"), ("ы", "ые"), ("ы", "ьі"), ("и", "н"), ("п", "н"), ("п", "и"), ("ш", "щ"), ("ц", "щ"), ("ь", "ъ"),
    ("б", "6"), ("з", "3"), ("о", "0"), ("в", "з"), ("т", "г"), ("л", "д"), ("к", "х"), ("ѣ", "е"), ("ѣ", "т"),
    ("іі", "п"), ("ii", "п"), ("ю", "іо"), ("р", "ф"), ("я", "л"),
    # Latin
    ("rn", "m"), ("cl", "d"), ("li", "h"), ("vv", "w"), ("ii", "u"), ("ri", "n"), ("in", "m"), ("e", "c"),
    ("f", "s"), ("f", "ſ"), ("l", "1"), ("l", "i"), ("I", "l"), ("o", "0"), ("h", "b"), ("u", "n"), ("a", "o"),
    ("t", "f"), ("c", "o"), ("é", "e"), ("è", "e"), ("ü", "ii"), ("ö", "o"), ("ä", "a"), ("ß", "fs"),
)


def directed_pairs() -> tuple[tuple[str, str], ...]:
    """All substitutions ``(read text, intended text)``, both directions and upper-case forms.

    Usage:
        Built once per process by the proofread and verifier workers.

    Returns:
        Unique ``(source, target)`` pairs; sources are never empty.
    """
    substitution_pairs: set[tuple[str, str]] = set()
    for first_group, second_group in CONFUSION_GROUPS:
        for source_text, target_text in ((first_group, second_group), (second_group, first_group)):
            substitution_pairs.add((source_text, target_text))
            if source_text.upper() != source_text:
                substitution_pairs.add((source_text.upper(), target_text.upper()))
            if source_text[:1].upper() != source_text[:1]:
                substitution_pairs.add((source_text[:1].upper() + source_text[1:],
                                        target_text[:1].upper() + target_text[1:]))
    return tuple(sorted(substitution_pairs))
