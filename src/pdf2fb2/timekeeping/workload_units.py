"""Workload of a conversion and the unit each stage's speed is measured in.

Page-level stages are timed per page, OCR and verification per text token (word), proofreading per token the
language model inspects (every token in full mode, the verifier-flagged ones otherwise). Unknown sizes are filled
in pessimistically from the stored text profile: every page is taken for a text-only page.
"""

from dataclasses import dataclass

from pdf2fb2.progress.stage_names import PROOFREADING_STAGE, RECOGNITION_STAGE, VERIFICATION_STAGE

PAGE_UNITS = "pages"
TOKEN_UNITS = "tokens"  # nosec B105  # noqa: S105 - unit name, not a credential
INSPECTED_UNITS = "inspected_tokens"
FULL_SCOPE = "full"
FLAGGED_SCOPE = "flagged"
STAGE_UNIT_KINDS = {RECOGNITION_STAGE: TOKEN_UNITS, VERIFICATION_STAGE: TOKEN_UNITS,
                    PROOFREADING_STAGE: INSPECTED_UNITS}


@dataclass(frozen=True)
class ConversionWorkload:
    """What one conversion has to process; None means not known yet.

    Attributes:
        page_count: Pages of the book.
        text_page_count: Pages with at least one segment to OCR.
        token_count: Words read by OCR.
        inspected_token_count: Words the proofreading language model inspects.
    """

    page_count: int | None = None
    text_page_count: int | None = None
    token_count: int | None = None
    inspected_token_count: int | None = None


@dataclass(frozen=True)
class TextProfile:
    """Pessimistic text density of the books converted so far.

    Attributes:
        tokens_per_text_page: Words on a text-only page (upper estimate, > 0).
        flagged_token_share: Share of words flagged as possible OCR mistakes, 0..1 (upper estimate).
    """

    tokens_per_text_page: float
    flagged_token_share: float


@dataclass(frozen=True)
class ResolvedWorkload:
    """Workload with every size known or estimated.

    Attributes:
        unit_counts: Units per unit kind (``pages``, ``tokens``, ``inspected_tokens``).
    """

    unit_counts: dict[str, float]


def stage_unit_kind(stage_name: str) -> str:
    """Unit a stage's speed is measured in.

    Args:
        stage_name: Progress stage name.

    Returns:
        ``tokens``, ``inspected_tokens`` or ``pages``.
    """
    return STAGE_UNIT_KINDS.get(stage_name, PAGE_UNITS)


def speed_metric_key(stage_name: str, proofread_scope: str) -> str:
    """Key of a stage in the speed metrics file.

    Args:
        stage_name: Progress stage name.
        proofread_scope: ``full`` or ``flagged`` (proofreading speeds differ a lot between the two).

    Returns:
        The stage name, or ``proofreading_<scope>`` for proofreading.
    """
    return f"{stage_name}_{proofread_scope}" if stage_name == PROOFREADING_STAGE else stage_name


def resolve_workload(conversion_workload: ConversionWorkload, text_profile: TextProfile,
                     proofread_scope: str) -> ResolvedWorkload:
    """Fill the unknown sizes of a workload from the text profile.

    Usage:
        Called before every estimate; pages without a known split are all taken for text pages.

    Args:
        conversion_workload: Known sizes (None: unknown).
        text_profile: Stored text density.
        proofread_scope: ``full`` or ``flagged``.

    Returns:
        Unit count of every unit kind.
    """
    page_count = float(conversion_workload.page_count or 0)
    text_pages = page_count if conversion_workload.text_page_count is None \
        else float(conversion_workload.text_page_count)
    token_count = text_pages * text_profile.tokens_per_text_page if conversion_workload.token_count is None \
        else float(conversion_workload.token_count)
    inspected_count = conversion_workload.inspected_token_count
    if inspected_count is None:
        inspected_count = token_count if proofread_scope == FULL_SCOPE \
            else token_count * text_profile.flagged_token_share
    return ResolvedWorkload({PAGE_UNITS: page_count, TOKEN_UNITS: token_count,
                             INSPECTED_UNITS: float(inspected_count)})
