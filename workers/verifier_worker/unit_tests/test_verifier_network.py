"""Unit tests of the verifier network with a tiny local tokenizer and random tiny encoders (nothing downloaded)."""

import random
from collections import Counter

import numpy
import torch
from token_verifier.datasets.example_files import IGNORE_INDEX, TokenExample
from token_verifier.datasets.token_encoding import merge_window_predictions, text_windows, window_ranges
from token_verifier.datasets.verify_texts import VerifyText
from token_verifier.features.evidence_features import EVIDENCE_WIDTH, evidence_rows
from token_verifier.features.feature_assembly import FeatureAssembler
from token_verifier.inference.unit_prediction import averaged_rows, predict_units
from token_verifier.networks.backbone_adapters import ByteBackbone, SubwordBackbone
from token_verifier.networks.model_loading import (
    EvidenceSources,
    LoadedVerifier,
    VerifierBundle,
    WindowShape,
    load_member,
    save_verifier,
)
from token_verifier.networks.verifier_network import VerifierNetwork, pooled_units
from token_verifier.training.parameter_groups import freeze_word_embeddings, parameter_groups
from token_verifier.training.relabel_service import book_folds, relabelled_example
from token_verifier.training.training_data import TrainingCorpus, language_balanced
from token_verifier.training.training_loop import averaged_view, averaged_weights, optimizer_state, train_epoch
from token_verifier.training.training_stages import MemberRun, fit_member
from token_verifier.verifier_settings import TrainingSettings
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, T5Config, T5EncoderModel

TINY_VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "old", "mission", "at", "dawn", "##s", "rn", "##od"]
WINDOW_SHAPE = WindowShape(window_units=4, max_tokens=32, context_units=2)


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


def tiny_verifier(feature_width: int = EVIDENCE_WIDTH, seed_value: int = 0) -> LoadedVerifier:
    """Random two-layer encoder with the unit head on CPU.

    Args:
        feature_width: Evidence width.
        seed_value: Initialisation seed.

    Returns:
        Loaded verifier.
    """
    torch.manual_seed(seed_value)
    encoder_config = BertConfig(vocab_size=len(TINY_VOCABULARY), hidden_size=16, num_hidden_layers=2,
                                num_attention_heads=2, intermediate_size=32)
    backbone_adapter = SubwordBackbone(tiny_tokenizer(), BertModel(encoder_config, add_pooling_layer=False))
    verifier_network = VerifierNetwork(backbone_adapter.encoder_module, 16, feature_width, 0)
    return LoadedVerifier(backbone_adapter, verifier_network.eval(), torch.device("cpu"), "subword")


def training_settings(epoch_count: int = 1, patience: int = 0) -> TrainingSettings:
    """Small CPU training settings.

    Args:
        epoch_count: Epochs of the mixed stage.
        patience: Early-stop patience.

    Returns:
        Settings.
    """
    return TrainingSettings("cpu", 32, 4, 2, 1, 1e-3, 0.0, 0.0, epoch_count, (1.0, 1.5, 1.5), 1, False, 0,
                            early_stop_patience=patience, context_units=2, rdrop_weight=0.5, ema_decay=0.9,
                            real_stage_epochs=0)


LABELLED_TEXT = VerifyText(("the", "rnod", "mission", "at", "dawn"), ("correct", "misread", "correct", "inserted",
                                                                     "correct"), ("old",), ("dawn",),
                           {"ocr_min_logprob": (-0.1, -9.0, None, -0.2, -0.1), "box_missing": (0, 0, 0, 1, 0)})


def test_pooling_uses_every_piece_of_a_unit() -> None:
    """Max and mean run over all pieces of a unit; pieces of no unit are ignored."""
    piece_states = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
    piece_units = torch.tensor([[-1, 0, 0, 1]])
    unit_states = pooled_units(piece_states, piece_units, 3)
    assert unit_states[0, 0].tolist() == [5.0, 4.0]
    assert unit_states[0, 1].tolist() == [7.0, 7.0]
    assert unit_states[0, 2].tolist() == [0.0, 0.0]


def test_windows_carry_context_labels_and_evidence() -> None:
    """Context pieces belong to no unit, labels and evidence are per unit, windows cover the text."""
    loaded_verifier = tiny_verifier()
    unit_evidence = FeatureAssembler(None, None).assemble(LABELLED_TEXT)
    encoded_windows = text_windows(loaded_verifier.backbone_adapter, LABELLED_TEXT, unit_evidence, (4, 32, 2), False)
    assert [encoded_window.first_unit for encoded_window in encoded_windows] == [0, 1]
    assert encoded_windows[0].unit_labels == (0, 1, 0, 2)
    assert encoded_windows[0].encoded_units() == {0, 1, 2, 3}
    assert encoded_windows[0].feature_rows.shape == (4, EVIDENCE_WIDTH)
    assert window_ranges(10, 4, True)[-1] == (6, 10)


def test_evidence_rows_hold_value_and_presence() -> None:
    """Missing evidence is zero with a cleared flag; present evidence is normalised with a set flag."""
    evidence_matrix = evidence_rows(LABELLED_TEXT)
    assert evidence_matrix[1, 0] < -0.4 and evidence_matrix[1, 1] == 1.0
    assert evidence_matrix[2, 0] == 0.0 and evidence_matrix[2, 1] == 0.0
    assert numpy.all(evidence_rows(LABELLED_TEXT, hidden_evidence=True) == 0)


def test_byte_backbone_maps_bytes_to_units() -> None:
    """Every byte of a unit points to it; spaces, separators and context bytes point nowhere."""
    encoder_config = T5Config(vocab_size=259, d_model=8, d_kv=4, d_ff=16, num_layers=1, num_heads=2)
    byte_backbone = ByteBackbone(T5EncoderModel(encoder_config))
    piece_encoding = byte_backbone.encode_units(["ab", "ё"], ["x"], [], 64)
    assert piece_encoding.piece_units == (-1, -1, 0, 0, -1, 1, 1, -1)
    assert piece_encoding.input_identifiers[-1] == 1


def test_prediction_merges_and_averages_members() -> None:
    """Every unit gets one probability row; an ensemble averages its members."""
    verifier_members = [tiny_verifier(seed_value=0), tiny_verifier(seed_value=1)]
    verifier_bundle = VerifierBundle(verifier_members, FeatureAssembler(None, None), WINDOW_SHAPE)
    probability_lists = predict_units(verifier_bundle, [LABELLED_TEXT, VerifyText(("dawn",))])
    assert [len(probability_list) for probability_list in probability_lists] == [5, 1]
    assert abs(sum(probability_lists[0][0]) - 1.0) < 1e-4
    assert averaged_rows([[[[1.0, 0.0, 0.0]]], [[[0.0, 1.0, 0.0]]]]) == [[[0.5, 0.5, 0.0]]]
    assert merge_window_predictions(2, [], []) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_frozen_embeddings_and_layer_decay() -> None:
    """Frozen word embeddings get no param_group; lower layers get smaller learning rates than the head."""
    loaded_verifier = tiny_verifier()
    freeze_word_embeddings(loaded_verifier.backbone_adapter.encoder_module)
    param_groups_list = parameter_groups(loaded_verifier.verifier_network, 2, 1e-3, 0.01, 0.5)
    grouped_ids = {id(parameter_value) for param_group in param_groups_list
                   for parameter_value in param_group["params"]}
    word_embedding = loaded_verifier.backbone_adapter.encoder_module.get_input_embeddings().weight
    assert id(word_embedding) not in grouped_ids
    learning_rates = sorted({param_group["lr"] for param_group in param_groups_list})
    assert learning_rates[0] < learning_rates[-1] == 1e-3


def test_training_epoch_with_rdrop_and_ema() -> None:
    """An epoch with R-Drop and EMA runs; the EMA view differs from the live weights after an update."""
    loaded_verifier = tiny_verifier()
    unit_evidence = FeatureAssembler(None, None).assemble(LABELLED_TEXT)
    encoded_windows = text_windows(loaded_verifier.backbone_adapter, LABELLED_TEXT, unit_evidence, (4, 32, 2),
                                   False) * 2
    settings = training_settings()
    loop_state = optimizer_state(loaded_verifier, settings, 1e-2, 4)
    ema_model = averaged_weights(loaded_verifier, 0.5)
    epoch_loss = train_epoch(loaded_verifier, encoded_windows, loop_state, settings, random.Random(0), ema_model)
    assert epoch_loss == epoch_loss and epoch_loss > 0
    averaged = averaged_view(loaded_verifier, ema_model)
    assert averaged.verifier_network is not loaded_verifier.verifier_network
    assert averaged.backbone_adapter.encoder_module is averaged.verifier_network.encoder_module


def test_member_training_stops_early_and_reports_languages(capsys) -> None:
    """With patience 1 a second worse epoch ends a stage; every epoch prints per-language scores."""
    loaded_verifier = tiny_verifier()
    assembler = FeatureAssembler(None, None)
    unit_evidence = assembler.assemble(LABELLED_TEXT)
    tiny_windows = text_windows(loaded_verifier.backbone_adapter, LABELLED_TEXT, unit_evidence, (4, 32, 2), False)
    tiny_corpus = TrainingCorpus(tiny_windows, [], [(LABELLED_TEXT, "english")])
    for parameter_value in loaded_verifier.verifier_network.parameters():
        parameter_value.requires_grad_(False)
    loaded_verifier.verifier_network.classifier[-1].weight.requires_grad_(True)
    settings = TrainingSettings("cpu", 32, 4, 2, 1, 0.0, 0.0, 0.0, 5, (1.0, 1.5, 1.5), 1, False, 0,
                                early_stop_patience=1, context_units=2, rdrop_weight=0.0, ema_decay=0.0,
                                real_stage_epochs=0)
    saved_scores: list[dict] = []
    member_run = MemberRun(tiny_corpus, settings, assembler, WINDOW_SHAPE,
                           lambda _, thresholds: saved_scores.append(thresholds), random.Random(0), None)
    fit_member(loaded_verifier, member_run)
    printed_text = capsys.readouterr().out
    assert printed_text.count("validation mean F1") == 2 and "early stop" in printed_text
    assert "english" in printed_text and len(saved_scores) == 1


def test_save_and_load_round_trip(tmp_path) -> None:
    """A saved verifier loads back with identical head weights and window settings."""
    loaded_verifier = tiny_verifier()
    save_verifier(loaded_verifier, tmp_path, EvidenceSources(None, None), WINDOW_SHAPE)
    reloaded = load_member(tmp_path, torch.device("cpu"))
    original_head = loaded_verifier.verifier_network.head_state()
    for parameter_name, parameter_value in reloaded.verifier_network.head_state().items():
        assert torch.equal(parameter_value, original_head[parameter_name])


def test_relabelling_overrules_only_confident_contradictions() -> None:
    """Labels change only when the fold model is confident; ignore stays; fold_sets split books."""
    token_example = TokenExample("book", "english", ("a", "b", "c"), ("misread", "correct", "ignore"))
    change_counter: Counter = Counter()
    relabelled = relabelled_example(token_example, [[0.95, 0.03, 0.02], [0.6, 0.4, 0.0], [0.0, 1.0, 0.0]], 0.9,
                                    change_counter)
    assert relabelled.unit_labels == ("correct", "correct", "ignore")
    assert change_counter == Counter({"misread->correct": 1})
    fold_sets = book_folds([TokenExample(f"book-{book_number}", "english", ("a",), ("correct",))
                        for book_number in range(6)], 3)
    assert sorted(len(fold_books) for fold_books in fold_sets) == [2, 2, 2]
    assert IGNORE_INDEX == -100


def test_real_examples_are_balanced_across_languages() -> None:
    """A language with few examples is repeated towards the largest one, up to the cap; near-equal ones are not."""
    english_examples = [TokenExample(f"en-{book_number}", "english", ("the",), ("correct",))
                        for book_number in range(6)]
    french_examples = [TokenExample("fr-0", "french", ("le",), ("correct",))]
    balanced_examples = language_balanced(english_examples + french_examples, 4)
    assert sum(token_example.language_name == "french" for token_example in balanced_examples) == 4
    japanese_examples = [TokenExample(f"ja-{book_number}", "japanese", ("の",), ("correct",))
                         for book_number in range(5)]
    assert len(language_balanced(english_examples + japanese_examples, 3)) == 11
