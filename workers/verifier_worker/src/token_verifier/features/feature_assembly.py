"""Assembles the per-unit evidence matrix and the crop matrix of every text, from whichever providers are on.

Width of the evidence matrix: page evidence (``evidence_features``) plus, when a language-model scorer is
configured, its plausibility columns. The crop matrix exists only when a visual embedder is configured.
"""

import random
from dataclasses import dataclass

import numpy

from token_verifier.datasets.verify_texts import VerifyText
from token_verifier.features.evidence_features import EVIDENCE_WIDTH, dropped_evidence, evidence_rows
from token_verifier.features.language_model_scores import LANGUAGE_MODEL_WIDTH, LanguageModelScorer
from token_verifier.features.visual_embeddings import VisualEmbedder


@dataclass(frozen=True)
class UnitEvidence:
    """Evidence of one text.

    Attributes:
        feature_rows: ``(units, feature_width)`` float32.
        visual_rows: ``(units, visual_width)`` float32, or None without a visual embedder.
    """

    feature_rows: numpy.ndarray
    visual_rows: numpy.ndarray | None


class FeatureAssembler:
    """Combines the configured providers.

    Usage:
        One per training run or job; ``assemble`` is called for every text before windowing.
    """

    def __init__(self, language_scorer: LanguageModelScorer | None, visual_embedder: VisualEmbedder | None,
                 evidence_dropout: float = 0.0, random_seed: int = 0) -> None:
        """Bind the providers.

        Args:
            language_scorer: Plausibility scorer, or None.
            visual_embedder: Crop embedder, or None.
            evidence_dropout: Probability of hiding a text's page evidence (training only).
            random_seed: Seed of the dropout decisions.
        """
        self.language_scorer = language_scorer
        self.visual_embedder = visual_embedder
        self.evidence_dropout = evidence_dropout
        self.noise_generator = random.Random(random_seed)  # nosec B311  # noqa: S311 - augmentation, not cryptography

    @property
    def feature_width(self) -> int:
        """Width of a unit's evidence vector.

        Returns:
            Page evidence width plus the language-model columns when configured.
        """
        return EVIDENCE_WIDTH + (LANGUAGE_MODEL_WIDTH if self.language_scorer is not None else 0)

    @property
    def visual_width(self) -> int:
        """Width of a unit's crop vector.

        Returns:
            0 without a visual embedder.
        """
        return self.visual_embedder.visual_width if self.visual_embedder is not None else 0

    def assemble(self, verify_text: VerifyText, allow_dropout: bool = False) -> UnitEvidence:
        """Evidence of one text.

        Args:
            verify_text: Text to describe.
            allow_dropout: Apply evidence dropout (training windows only).

        Returns:
            Its evidence matrices.
        """
        hidden_evidence = allow_dropout and dropped_evidence(self.evidence_dropout, self.noise_generator)
        feature_parts = [evidence_rows(verify_text, hidden_evidence)]
        if self.language_scorer is not None:
            feature_parts.append(self.language_scorer.unit_scores(verify_text.unit_texts))
        visual_rows = None if self.visual_embedder is None else self.visual_embedder.crop_rows(
            verify_text.crop_paths, len(verify_text.unit_texts))
        return UnitEvidence(numpy.concatenate(feature_parts, axis=1), visual_rows)
