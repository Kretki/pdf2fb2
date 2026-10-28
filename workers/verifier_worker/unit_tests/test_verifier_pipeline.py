"""Unit tests of the verifier worker with a tiny local tokenizer and a random tiny encoder (nothing downloaded)."""

import random

import torch
from token_verifier.datasets.example_files import IGNORE_INDEX, TokenExample
from token_verifier.datasets.noise_synthesis import corrupt_chunk, misread_unit
from token_verifier.datasets.token_encoding import (
    merge_window_predictions,
    prediction_windows,
    training_windows,
    window_ranges,
)
from token_verifier.evaluation.label_metrics import class_scores, predicted_label
from token_verifier.evaluation.threshold_tuning import tune_thresholds
from token_verifier.inference.unit_prediction import predict_units
from token_verifier.networks.model_loading import LoadedVerifier
from token_verifier.training.training_loop import OptimizerState, train_epoch
from token_verifier.verifier_settings import SynthesisSettings, TrainingSettings, load_settings
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from worker_protocol.verify_job_schema import VerifyItem, VerifyJob

TINY_VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "old", "mission", "at", "dawn", "##s", "rn", "##od"]


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-piece tokenizer over a dozen pieces.

    Returns:
        Fast tokenizer with word ids.
    """
    piece_model = models.WordPiece({piece_text: piece_index for piece_index, piece_text in enumerate(TINY_VOCABULARY)},
                                   unk_token="[UNK]")
    backend_tokenizer = Tokenizer(piece_model)
    backend_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend_tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")


def tiny_verifier() -> LoadedVerifier:
    """Random two-layer encoder with a 3-class head on CPU.

    Returns:
        Loaded verifier.
    """
    torch.manual_seed(0)
    encoder_config = BertConfig(vocab_size=len(TINY_VOCABULARY), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, num_labels=3)
    return LoadedVerifier(tiny_tokenizer(), BertForTokenClassification(encoder_config).eval(), torch.device("cpu"))


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
    assert verifier_settings.training_settings.class_weights == (1.0, 3.0, 3.0)
    assert verifier_settings.path_settings.output_model.name == "token_verifier"


def test_corruption_keeps_labels_aligned() -> None:
    """Misread units differ from the reference, inserted units are added, every unit has a label."""
    reference_units = ["время", "the", "modern", "mission", "at", "dawn"]
    token_example = corrupt_chunk("book", "russian", reference_units, ["donor", "words", "here", "now"],
                                  synthesis_settings(), random.Random(3))
    assert len(token_example.unit_texts) == len(token_example.unit_labels)
    assert token_example.unit_labels.count("inserted") >= 1
    kept_units = [unit_text for unit_text, unit_label in zip(token_example.unit_texts, token_example.unit_labels,
                                                              strict=True) if unit_label != "inserted"]
    assert len(kept_units) == len(reference_units)
    for kept_unit, reference_unit, unit_label in zip(kept_units, reference_units, [
            unit_label for unit_label in token_example.unit_labels if unit_label != "inserted"], strict=True):
        assert (kept_unit != reference_unit) == (unit_label == "misread")
    assert misread_unit("modern", random.Random(1)) != "modern"


def test_windows_cover_every_unit_and_label_first_pieces() -> None:
    """Disjoint training windows, overlapping prediction windows, labels only on first sub-words."""
    assert window_ranges(10, 4, False) == [(0, 4), (4, 8), (6, 10)]
    assert window_ranges(10, 4, True)[-1] == (6, 10)
    token_example = TokenExample("book", "english", ("the", "rnod", "mission"), ("correct", "misread", "correct"))
    (encoded_window,) = training_windows(tiny_tokenizer(), token_example, 8, 32)
    labelled_ids = [label_id for label_id in encoded_window.label_identifiers if label_id != IGNORE_INDEX]
    assert labelled_ids == [0, 1, 0]
    assert len(encoded_window.input_identifiers) == len(encoded_window.label_identifiers)


def test_overlapping_windows_merge_into_one_row_per_unit() -> None:
    """Every unit gets the prediction of the window where it has most context."""
    unit_texts = ["the", "old", "mission", "at", "dawn", "the", "old"]
    encoded_windows = prediction_windows(tiny_tokenizer(), unit_texts, 4, 32)
    fake_rows = [[[0.0, float(window_number), 0.0] for _ in encoded_window.input_identifiers]
                 for window_number, encoded_window in enumerate(encoded_windows)]
    merged_rows = merge_window_predictions(len(unit_texts), encoded_windows, fake_rows)
    assert len(merged_rows) == len(unit_texts)
    assert merged_rows[0][1] == 0.0 and merged_rows[-1][1] == float(len(encoded_windows) - 1)


def test_prediction_and_one_training_epoch_run_on_cpu() -> None:
    """The random tiny encoder predicts one triple per unit and a training epoch lowers nothing to NaN."""
    loaded_verifier = tiny_verifier()
    probability_lists = predict_units(loaded_verifier, [["the", "old", "mission"], ["dawn"]], 2, 16)
    assert [len(probability_list) for probability_list in probability_lists] == [3, 1]
    assert abs(sum(probability_lists[0][0]) - 1.0) < 1e-4
    training_settings = TrainingSettings("cpu", 16, 4, 2, 1, 1e-3, 0.0, 0.0, 1, (1.0, 3.0, 3.0), 1, False, 0)
    token_example = TokenExample("book", "english", ("the", "rnod", "mission", "at"),
                                 ("correct", "misread", "correct", "inserted"))
    encoded_windows = training_windows(loaded_verifier.text_tokenizer, token_example, 4, 16) * 2
    weight_optimizer = torch.optim.AdamW(loaded_verifier.encoder_model.parameters(), lr=1e-3)
    constant_rate = torch.optim.lr_scheduler.LambdaLR(weight_optimizer, lambda _: 1.0)
    optimizer_state = OptimizerState(weight_optimizer, constant_rate,
                                     torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 3.0])))
    epoch_loss = train_epoch(loaded_verifier, encoded_windows, optimizer_state, training_settings, random.Random(0),
                             lambda done_count, total_count, running_loss: None)
    assert epoch_loss == epoch_loss and epoch_loss > 0


def test_thresholds_and_scores() -> None:
    """Tuned thresholds separate the classes; ignore units are skipped by the metrics."""
    gold_labels = ["correct", "misread", "inserted", "correct", "ignore"]
    probability_rows = [[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1], [0.0, 1.0, 0.0]]
    class_thresholds = tune_thresholds(gold_labels, probability_rows)
    misread_scores = class_scores(gold_labels, probability_rows, "misread", class_thresholds["misread"])
    assert misread_scores.score_summary()["f1"] == 1.0
    assert predicted_label(probability_rows[2], class_thresholds) == "inserted"
    assert predicted_label(probability_rows[0], class_thresholds) == "correct"


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


def test_real_examples_are_balanced_across_languages() -> None:
    """A language with few examples is repeated towards the largest one, up to the cap."""
    from token_verifier.training.training_service import language_balanced

    english_examples = [TokenExample(f"en-{book_number}", "english", ("the",), ("correct",))
                        for book_number in range(6)]
    french_examples = [TokenExample("fr-0", "french", ("le",), ("correct",))]
    balanced_examples = language_balanced(english_examples + french_examples, 4)
    assert sum(token_example.language_name == "french" for token_example in balanced_examples) == 4
    assert sum(token_example.language_name == "english" for token_example in balanced_examples) == 6
