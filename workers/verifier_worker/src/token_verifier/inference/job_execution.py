"""Pipeline job: label the units of every item of a ``VerifyJob`` and return probabilities and thresholds."""

from token_verifier.inference.unit_prediction import ProgressCallback, predict_units
from token_verifier.networks.model_loading import LoadedVerifier, read_thresholds, read_window_settings
from worker_protocol.verify_job_schema import VerifyJob

DEFAULT_WINDOW_UNITS = 96
DEFAULT_MAX_TOKENS = 256
PROBABILITY_DIGITS = 4


def run_verify_job(verify_job: VerifyJob, loaded_verifier: LoadedVerifier, progress_callback: ProgressCallback) -> dict:
    """Predict every unit of the job.

    Args:
        verify_job: Validated job.
        loaded_verifier: Fine-tuned model loaded from ``verify_job.model_directory``.
        progress_callback: Receives ``(done windows, total windows)``.

    Returns:
        Result document (see ``worker_protocol.verify_job_schema``).
    """
    window_units, max_tokens = read_window_settings(verify_job.model_directory, DEFAULT_WINDOW_UNITS,
                                                    DEFAULT_MAX_TOKENS)
    probability_lists = predict_units(loaded_verifier, [list(verify_item.item_units)
                                                        for verify_item in verify_job.verify_items],
                                      window_units, max_tokens, progress_callback)
    return {"thresholds": read_thresholds(verify_job.model_directory),
            "items": [{"key": verify_item.item_key,
                       "probabilities": [[round(class_probability, PROBABILITY_DIGITS)
                                          for class_probability in unit_probabilities]
                                         for unit_probabilities in probability_list]}
                      for verify_item, probability_list in zip(verify_job.verify_items, probability_lists,
                                                               strict=True)]}
