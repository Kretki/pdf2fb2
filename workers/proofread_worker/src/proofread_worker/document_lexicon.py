"""Vocabulary of the book itself, with a delete index for words within two edits (symmetric deletion).

Without a dictionary, the words a book prints several times are the best evidence of how its words are
spelled: a misread word is often one or two edits away from a form that occurs elsewhere in the text.
"""

from collections import Counter, defaultdict

MAX_EDIT_DISTANCE = 2
MIN_LEXICON_COUNT = 2   # a form must occur this often to be offered as a spelling
MAX_INDEXED_LENGTH = 24


def deletion_variants(word_text: str, max_deletions: int) -> set[str]:
    """Every string obtained by deleting up to ``max_deletions`` characters.

    Args:
        word_text: Lower-case word.
        max_deletions: 0..2.

    Returns:
        The variants, including the word itself.
    """
    variant_set = {word_text}
    frontier = {word_text}
    for _ in range(max_deletions):
        frontier = {variant_text[:char_index] + variant_text[char_index + 1:] for variant_text in frontier
                    for char_index in range(len(variant_text))}
        variant_set |= frontier
    return variant_set


def edit_distance(first_word: str, second_word: str, distance_limit: int) -> int:
    """Optimal-string-alignment distance (insert, delete, substitute, swap neighbours), capped.

    Args:
        first_word: First string.
        second_word: Second string.
        distance_limit: Values above it are returned as ``distance_limit + 1``.

    Returns:
        The distance, at most ``distance_limit + 1``.
    """
    if abs(len(first_word) - len(second_word)) > distance_limit:
        return distance_limit + 1
    previous_row: list[int] = []
    current_row = list(range(len(second_word) + 1))
    for first_index in range(1, len(first_word) + 1):
        before_row, previous_row = previous_row, current_row
        current_row = [first_index] + [0] * len(second_word)
        for second_index in range(1, len(second_word) + 1):
            substitution_cost = int(first_word[first_index - 1] != second_word[second_index - 1])
            current_row[second_index] = min(previous_row[second_index] + 1, current_row[second_index - 1] + 1,
                                            previous_row[second_index - 1] + substitution_cost)
            if first_index > 1 and second_index > 1 and first_word[first_index - 1] == second_word[second_index - 2] \
                    and first_word[first_index - 2] == second_word[second_index - 1]:
                current_row[second_index] = min(current_row[second_index], before_row[second_index - 2] + 1)
        if min(current_row) > distance_limit:
            return distance_limit + 1
    return min(current_row[-1], distance_limit + 1)


class DocumentLexicon:
    """Word counts of one document and a lookup of its frequent forms near a given word.

    Usage:
        Build once per job from the lower-case word cores, then call ``count_of`` and ``nearby_forms``.
    """

    def __init__(self, word_cores: list[str]) -> None:
        """Count the forms and index the frequent ones.

        Args:
            word_cores: Lower-case words of the document without punctuation (repetitions included).
        """
        self._form_counts = Counter(word_core for word_core in word_cores if word_core)
        self._delete_index: dict[str, set[str]] = defaultdict(set)
        for word_form, form_count in self._form_counts.items():
            if form_count >= MIN_LEXICON_COUNT and len(word_form) <= MAX_INDEXED_LENGTH:
                for deleted_form in deletion_variants(word_form, MAX_EDIT_DISTANCE):
                    self._delete_index[deleted_form].add(word_form)

    def count_of(self, word_form: str) -> int:
        """Occurrences of a lower-case form in the document.

        Args:
            word_form: Lower-case word.

        Returns:
            The count (0 when absent).
        """
        return self._form_counts.get(word_form, 0)

    def nearby_forms(self, word_form: str) -> list[tuple[str, int]]:
        """Frequent document forms within two edits of a word.

        Args:
            word_form: Lower-case word.

        Returns:
            ``(form, edit distance)`` pairs, the word itself excluded.
        """
        if len(word_form) > MAX_INDEXED_LENGTH:
            return []
        candidate_forms = set().union(*(self._delete_index.get(deleted_form, set())
                                        for deleted_form in deletion_variants(word_form, MAX_EDIT_DISTANCE)))
        candidate_forms.discard(word_form)
        measured_forms = [(candidate_form, edit_distance(word_form, candidate_form, MAX_EDIT_DISTANCE))
                          for candidate_form in candidate_forms]
        return [(candidate_form, form_distance) for candidate_form, form_distance in measured_forms
                if form_distance <= MAX_EDIT_DISTANCE]
