"""Pipeline job: label the units of every item of a ``VerifyJob`` and return probabilities and thresholds."""

from token_verifier.datasets.verify_texts import text_from_item
from token_verifier.inference.unit_prediction import ProgressCallback, predict_units
from token_verifier.networks.model_loading import VerifierBundle, read_thresholds
from worker_protocol.verify_job_schema import VerifyJob

PROBABILITY_DIGITS = 4


def run_verify_job(verify_job: VerifyJob, verifier_bundle: VerifierBundle, progress_callback: ProgressCallback) -> dict:
    """Predict every unit of the job (context, page evidence and crops are used when the items carry them).

    Args:
        verify_job: Validated job.
        verifier_bundle: Fine-tuned verifier loaded from ``verify_job.model_directory``.
        progress_callback: Receives ``(done windows, total windows)``.

    Returns:
        Result document (see ``worker_protocol.verify_job_schema``).

    Raises:
        ValueError: If a crop path leaves the work directory.
    """
    verify_texts = [text_from_item(verify_item, verify_job.work_directory) for verify_item in verify_job.verify_items]
    probability_lists = predict_units(verifier_bundle, verify_texts, progress_callback)
    return {"thresholds": read_thresholds(verify_job.model_directory),
            "items": [{"key": verify_item.item_key,
                       "probabilities": [[round(class_probability, PROBABILITY_DIGITS)
                                          for class_probability in unit_probabilities]
                                         for unit_probabilities in probability_list]}
                      for verify_item, probability_list in zip(verify_job.verify_items, probability_lists,
                                                               strict=True)]}
