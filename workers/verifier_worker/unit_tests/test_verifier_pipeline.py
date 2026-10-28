"""Unit tests of the verifier worker: synthesis, settings, metrics and the job schema (no model needed)."""

import random

from token_verifier.datasets.noise_synthesis import corrupt_chunk
from token_verifier.datasets.reference_screening import is_suspicious_unit
from token_verifier.datasets.span_corruptions import CorruptionResult, merged_units, split_unit
from token_verifier.datasets.unit_misreadings import misread_unit
from token_verifier.evaluation.label_metrics import class_scores, predicted_label
from token_verifier.evaluation.threshold_tuning import tune_thresholds
from token_verifier.verifier_settings import SynthesisSettings, load_settings

from worker_protocol.verify_job_schema import VerifyItem, VerifyJob


def synthesis_settings(insert_rate: float = 1.0) -> SynthesisSettings:
    """Aggressive synthesis settings.

    Args:
        insert_rate: Probability of an inserted span.

    Returns:
        Settings.
    """
    return SynthesisSettings(examples_per_chunk=1, misread_rate=0.5, insert_rate=insert_rate, max_insert_units=3,
                             random_seed=7)


def test_default_settings_load() -> None:
    """The shipped settings file is complete and typed."""
    verifier_settings = load_settings()
    assert verifier_settings.training_settings.class_weights == (1.0, 1.5, 1.5)
    assert verifier_settings.path_settings.output_model.name == "token_verifier"


def test_corruption_keeps_labels_aligned() -> None:
    """Every unit has a label, correct units are reference units, misread units are not, an insertion is made."""
    reference_units = ["время", "the", "modern", "mission", "at", "dawn", "remembered", "everything"]
    token_example = corrupt_chunk("book", "russian", reference_units, ["donor", "words", "here", "now"],
                                  synthesis_settings(), random.Random(3))
    assert len(token_example.unit_texts) == len(token_example.unit_labels)
    assert token_example.unit_labels.count("inserted") >= 1
    for unit_text, unit_label in zip(token_example.unit_texts, token_example.unit_labels, strict=True):
        if unit_label == "correct":
            assert unit_text in reference_units
        if unit_label == "misread":
            assert unit_text not in reference_units
    assert misread_unit("modern", random.Random(1)) != "modern"


def test_misreadings_change_letters_not_only_punctuation() -> None:
    """A misread unit always differs in its letters or digits (punctuation-only changes are ignored in real data)."""
    for seed_value in range(200):
        corrupted_text = misread_unit("dawn,", random.Random(seed_value))
        assert "".join(filter(str.isalnum, corrupted_text)) != "dawn"


def test_boundary_corruptions_follow_real_alignment_labels() -> None:
    """A merge yields one misread unit; a split yields a misread piece and an inserted piece; CJK is never split."""
    assert merged_units("hombre,", "ó") == CorruptionResult((("hombre,ó", "misread"),), 2)
    split_result = split_unit("community", random.Random(2))
    assert [unit_label for _, unit_label in split_result.labelled_units] == ["misread", "inserted"]
    assert "".join(unit_text for unit_text, _ in split_result.labelled_units).replace("-", "") == "community"
    assert split_unit("東京商工会議所", random.Random(2)) is None
    assert merged_units("東", "京") is None


def test_reference_ocr_errors_are_screened() -> None:
    """Units that look like OCR errors of the reference are not taken as correct."""
    assert [is_suspicious_unit(unit_text) for unit_text in ("goo4", "mo&t", "des.^en", "〓", "Пpимер", "(1919),",
                                                            "Théâtre", "1st", "ассоциациям,", "6hosen",
                                                            "\\hhfact", "Lord^", "1836.", "2-го", "word*")] == \
        [True, True, True, True, True, False, False, False, False, True, True, True, False, False, False]
    token_example = corrupt_chunk("book", "english", ["goo4", "word"], [], synthesis_settings(0.0),
                                  random.Random(0))
    assert token_example.unit_labels[0] == "ignore"


def test_segments_cover_chunk_with_bounded_lengths() -> None:
    """Synthetic segments tile the chunk and stay within the configured lengths."""
    from token_verifier.datasets.synthesis_service import segment_ranges

    segment_bounds = segment_ranges(500, synthesis_settings(), random.Random(4))
    assert segment_bounds[0][0] == 0 and segment_bounds[-1][1] == 500
    assert all(previous_bound[1] == next_bound[0] for previous_bound, next_bound in zip(segment_bounds[:-1],
                                                                                        segment_bounds[1:],
                                                                                        strict=True))
    assert all(segment_end - segment_start <= 150 for segment_start, segment_end in segment_bounds)


def test_thresholds_and_scores() -> None:
    """Tuned thresholds separate the classes; ignore units are skipped by the metrics."""
    gold_labels = ["correct", "misread", "inserted", "correct", "ignore"]
    probability_rows = [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1], [0.0, 1.0, 0.0]]
    class_thresholds = tune_thresholds(gold_labels, probability_rows)
    misread_scores = class_scores(gold_labels, probability_rows, "misread", class_thresholds["misread"])
    assert misread_scores.score_summary()["f1"] == 1.0
    assert predicted_label(probability_rows[2], class_thresholds) == "inserted"
    assert predicted_label(probability_rows[0], class_thresholds) == "correct"
    rare_rows = [[0.98, 0.0, 0.02], [0.99, 0.0, 0.01], [0.995, 0.0, 0.005]]
    assert tune_thresholds(["inserted", "inserted", "correct"], rare_rows)["inserted"] == 0.01


def test_verify_job_round_trip(tmp_path) -> None:
    """The job schema validates units and keeps the result path inside the work directory."""
    verify_job = VerifyJob(tmp_path, tmp_path, "cpu", (VerifyItem("p1:s0", ("the", "old")),), "verify_result.json")
    decoded_job = VerifyJob.from_payload(verify_job.to_payload())
    assert decoded_job.verify_items[0].item_units == ("the", "old")
    bad_payload = {**verify_job.to_payload(), "result_file": "../outside.json"}
    try:
        VerifyJob.from_payload(bad_payload)
    except ValueError:
        return
    raise AssertionError("path traversal accepted")


def test_verify_items_carry_context_features_and_crops(tmp_path) -> None:
    """Optional evidence survives a job round trip; malformed evidence is rejected."""
    verify_item = VerifyItem("3:7", ("the", "old"), ("before",), ("after",),
                             {"ocr_min_logprob": (-0.5, None), "box_missing": (0.0, 1.0)}, ("crops/a.png", None))
    verify_job = VerifyJob(tmp_path, tmp_path, "cpu", (verify_item,), "verify_result.json")
    assert VerifyJob.from_payload(verify_job.to_payload()).verify_items[0] == verify_item
    for bad_item in ({"key": "k", "units": ["a"], "features": {"unknown": [1.0]}},
                     {"key": "k", "units": ["a"], "features": {"box_missing": [1.0, 0.0]}},
                     {"key": "k", "units": ["a"], "crops": ["/abs.png"]}):
        try:
            VerifyJob.from_payload({**verify_job.to_payload(), "items": [bad_item]})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad_item}")
