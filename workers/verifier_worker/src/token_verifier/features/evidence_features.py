"""Page evidence of the pipeline as a fixed-width vector per unit: (normalised value, presence flag) per feature.

The presence flag lets one model serve texts with and without evidence (synthetic text has none); during
training the evidence of a whole text is sometimes hidden (``evidence_dropout``) so the model never depends on it.
"""

import random

import numpy

from token_verifier.datasets.verify_texts import VerifyText
from worker_protocol.unit_features import UNIT_FEATURE_NAMES, normalized_feature

EVIDENCE_WIDTH = 2 * len(UNIT_FEATURE_NAMES)


def evidence_rows(verify_text: VerifyText, hidden_evidence: bool = False) -> numpy.ndarray:
    """Evidence matrix of a text.

    Usage:
        Called by ``feature_assembly`` for every text before windowing.

    Args:
        verify_text: Text with optional feature columns.
        hidden_evidence: Return the "no evidence" matrix (used for evidence dropout).

    Returns:
        ``(units, EVIDENCE_WIDTH)`` float32 matrix.
    """
    unit_count = len(verify_text.unit_texts)
    evidence_matrix = numpy.zeros((unit_count, EVIDENCE_WIDTH), dtype=numpy.float32)
    if hidden_evidence:
        return evidence_matrix
    for feature_position, feature_name in enumerate(UNIT_FEATURE_NAMES):
        for unit_index, feature_value in enumerate(verify_text.unit_features.get(feature_name, ())):
            if feature_value is not None:
                evidence_matrix[unit_index, 2 * feature_position] = normalized_feature(feature_name, feature_value)
                evidence_matrix[unit_index, 2 * feature_position + 1] = 1.0
    return evidence_matrix


def dropped_evidence(dropout_rate: float, noise_generator: random.Random) -> bool:
    """Decide whether a training text loses its evidence this epoch.

    Args:
        dropout_rate: Probability 0..1.
        noise_generator: Random generator.

    Returns:
        True when the evidence is hidden.
    """
    return dropout_rate > 0 and noise_generator.random() < dropout_rate
