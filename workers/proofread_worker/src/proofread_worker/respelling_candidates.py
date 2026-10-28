"""Candidate spellings of a low-probability word: OCR confusions and nearby forms of the same book.

Candidates are at most two edits from the word read by the OCR. Confusion-pair substitutions (``с``/``е``,
``rn``/``m``, long ``s`` read as ``f``) are offered even when the result never occurs in the book; other edits only
when they lead to a form the book prints repeatedly. The language model then scores each spelling in context
(``rewrite_variants.respell_request``); the orchestrator decides.
"""

import math
import re
from dataclasses import dataclass

from proofread_worker.confusion_pairs import script_lookalikes
from proofread_worker.document_lexicon import DocumentLexicon
from worker_protocol.ocr_confusions import directed_pairs

CONFUSION_COST = 0.5      # one confusion-pair substitution
LOOKALIKE_COST = 0.2      # mixed Latin/Cyrillic look-alikes
LEXICON_EDIT_COST = 1.0   # one generic edit towards a form of the book
LEXICON_BONUS = 0.6       # per log occurrence in the book: any printed form outranks an unseen one
FREQUENT_FORM_COUNT = 3   # a form printed this often is taken as correct and not respelled
MAX_CONFUSION_EDITS = 2
MAX_WORD_LENGTH = 40
MIN_CORE_LENGTH = 2
WORD_SHAPE_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
LETTER_PATTERN = re.compile(r"[^\W\d_]")
SUBSTITUTION_PAIRS = directed_pairs()


@dataclass(frozen=True)
class RespellCandidate:
    """One spelling to test.

    Attributes:
        spelling: Full word with its original surrounding punctuation.
        edit_cost: Prior cost of the edits that lead to it (nats-like units, lower is likelier).
    """

    spelling: str
    edit_cost: float


def split_word(word_text: str) -> tuple[str, str, str]:
    """Split a word into leading punctuation, core and trailing punctuation.

    Args:
        word_text: Plain word as it stands in the text, e.g. ``«слово,``.

    Returns:
        ``(prefix, core, suffix)``.
    """
    shape_match = WORD_SHAPE_PATTERN.match(word_text)
    return shape_match.group(1), shape_match.group(2), shape_match.group(3)


def match_case(model_word: str, candidate_form: str) -> str:
    """Give a lower-case form the capitalisation of the word it replaces.

    Args:
        model_word: Word read by the OCR.
        candidate_form: Lower-case candidate.

    Returns:
        Upper-case, title-case or unchanged form.
    """
    if model_word.isupper() and len(model_word) > 1:
        return candidate_form.upper()
    if model_word[:1].isupper():
        return candidate_form[:1].upper() + candidate_form[1:]
    return candidate_form


def single_substitutions(word_core: str) -> set[str]:
    """Every word obtained by one confusion-pair substitution.

    Args:
        word_core: Word without punctuation.

    Returns:
        Substituted words (the input excluded).
    """
    substituted_words = set()
    for source_text, target_text in SUBSTITUTION_PAIRS:
        search_from = word_core.find(source_text)
        while search_from >= 0:
            substituted_words.add(word_core[:search_from] + target_text + word_core[search_from + len(source_text):])
            search_from = word_core.find(source_text, search_from + 1)
    substituted_words.discard(word_core)
    return substituted_words


def confusion_candidates(word_core: str) -> dict[str, float]:
    """Words within two confusion substitutions, with their cost.

    Args:
        word_core: Word without punctuation.

    Returns:
        Mapping word -> cost.
    """
    candidate_costs: dict[str, float] = {}
    frontier = {word_core}
    for edit_number in range(1, MAX_CONFUSION_EDITS + 1):
        frontier = set().union(*(single_substitutions(frontier_word) for frontier_word in frontier)) - {word_core}
        for frontier_word in frontier:
            candidate_costs.setdefault(frontier_word, CONFUSION_COST * edit_number)
    lookalike_word = script_lookalikes(word_core)
    if lookalike_word:
        candidate_costs[lookalike_word] = LOOKALIKE_COST
    return candidate_costs


def lexicon_candidates(word_core: str, document_lexicon: DocumentLexicon) -> dict[str, float]:
    """Frequent forms of the book within two edits, with their cost and case restored.

    Args:
        word_core: Word without punctuation.
        document_lexicon: Vocabulary of the book.

    Returns:
        Mapping word -> cost.
    """
    return {match_case(word_core, candidate_form): LEXICON_EDIT_COST * form_distance
            for candidate_form, form_distance in document_lexicon.nearby_forms(word_core.lower())}


def ranking_key(word_core: str, candidate_core: str, candidate_cost: float, document_lexicon: DocumentLexicon) -> float:
    """Lower is better: edit cost minus a bonus for forms the book prints.

    Args:
        word_core: Word read by the OCR (its capitalisation is the expected one).
        candidate_core: Candidate word without punctuation.
        candidate_cost: Its edit cost.
        document_lexicon: Vocabulary of the book.

    Returns:
        Sort key; only candidates capitalised like the word read get the bonus.
    """
    lower_form = candidate_core.lower()
    if candidate_core != match_case(word_core, lower_form):
        return candidate_cost
    return candidate_cost - LEXICON_BONUS * math.log1p(document_lexicon.count_of(lower_form))


def respellable(word_core: str, document_lexicon: DocumentLexicon) -> bool:
    """Whether a word is worth respelling at all.

    Args:
        word_core: Word without punctuation.
        document_lexicon: Vocabulary of the book.

    Returns:
        False for words without letters, very short or long words and forms the book prints repeatedly.
    """
    return (MIN_CORE_LENGTH <= len(word_core) <= MAX_WORD_LENGTH and bool(LETTER_PATTERN.search(word_core))
            and document_lexicon.count_of(word_core.lower()) < FREQUENT_FORM_COUNT)


def candidate_spellings(word_text: str, document_lexicon: DocumentLexicon,
                        max_candidates: int) -> list[RespellCandidate]:
    """Best-ranked alternative spellings of one word.

    Usage:
        Called by the proofread job for every low-probability word.

    Args:
        word_text: Plain word as it stands in the text (punctuation attached).
        document_lexicon: Vocabulary of the book.
        max_candidates: Maximum number of spellings returned, 1..7.

    Returns:
        Candidates, best first (empty when the word is not respellable).
    """
    word_prefix, word_core, word_suffix = split_word(word_text)
    if not respellable(word_core, document_lexicon):
        return []
    candidate_costs = lexicon_candidates(word_core, document_lexicon)
    for candidate_core, candidate_cost in confusion_candidates(word_core).items():
        candidate_costs[candidate_core] = min(candidate_cost, candidate_costs.get(candidate_core, candidate_cost))
    candidate_costs.pop(word_core, None)
    ranked_cores = sorted(candidate_costs, key=lambda candidate_core: (
        ranking_key(word_core, candidate_core, candidate_costs[candidate_core], document_lexicon), candidate_core))
    return [RespellCandidate(word_prefix + candidate_core + word_suffix, candidate_costs[candidate_core])
            for candidate_core in ranked_cores[:max_candidates]]
