"""Two-stage fine-tuning of one verifier: mixed data first, then real data with a little synthetic replay.

Stage ``mixed`` (``epoch_count`` epochs at ``learning_rate``) learns from everything; stage ``real``
(``real_stage_epochs`` epochs at ``real_stage_learning_rate``) moves the final weights towards the distribution
of real pipeline output. After every epoch the (EMA) weights are validated, a report is printed and the best
weights are handed to the save callback; a stage stops after ``early_stop_patience`` epochs without improvement.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

from torch.optim.swa_utils import AveragedModel

from token_verifier.datasets.token_encoding import EncodedWindow
from token_verifier.datasets.verify_texts import VerifyText
from token_verifier.evaluation.evaluation_service import grouped_texts_scores
from token_verifier.evaluation.label_metrics import ERROR_CLASSES
from token_verifier.evaluation.threshold_tuning import tune_thresholds
from token_verifier.features.feature_assembly import FeatureAssembler
from token_verifier.inference.unit_prediction import predict_units
from token_verifier.networks.model_loading import LoadedVerifier, VerifierBundle, WindowShape
from token_verifier.training.epoch_report import epoch_report
from token_verifier.training.training_data import TrainingCorpus, replay_sample
from token_verifier.training.training_loop import averaged_view, averaged_weights, optimizer_state, train_epoch
from token_verifier.verifier_settings import TrainingSettings

SaveCallback = Callable[[LoadedVerifier, dict[str, float]], None]


@dataclass(frozen=True)
class StagePlan:
    """One training stage.

    Attributes:
        stage_name: ``mixed`` or ``real``.
        epoch_count: Epochs of the stage.
        learning_rate: Peak learning rate.
        replay_only: True when synthetic windows are only replayed (sampled), not all used.
    """

    stage_name: str
    epoch_count: int
    learning_rate: float
    replay_only: bool


@dataclass(frozen=True)
class EpochValidation:
    """Validation result of one epoch.

    Attributes:
        mean_score: Mean F1 of the error classes over all languages (the selection score).
        class_thresholds: Thresholds tuned on the selection data.
        score_groups: Scores per class, overall (``ALL``) and per language.
    """

    mean_score: float
    class_thresholds: dict[str, float]
    score_groups: dict[str, dict]


def stage_plans(training_settings: TrainingSettings, data_corpus: TrainingCorpus) -> list[StagePlan]:
    """Stages to run.

    Args:
        training_settings: Epochs and learning rates.
        data_corpus: Training windows (the real stage needs real windows).

    Returns:
        One or two stages.
    """
    stage_list = [StagePlan("mixed", training_settings.epoch_count, training_settings.learning_rate, False)]
    if training_settings.real_stage_epochs > 0 and data_corpus.real_windows:
        stage_list.append(StagePlan("real", training_settings.real_stage_epochs,
                               training_settings.real_stage_learning_rate, True))
    return stage_list


def stage_windows(stage_plan: StagePlan, data_corpus: TrainingCorpus, training_settings: TrainingSettings,
                  sample_generator: random.Random) -> list[EncodedWindow]:
    """Windows of one epoch of a stage.

    Args:
        stage_plan: Stage.
        data_corpus: All windows.
        training_settings: Replay share.
        sample_generator: Sampling generator.

    Returns:
        Windows of the epoch.
    """
    if not stage_plan.replay_only:
        return data_corpus.synthetic_windows + data_corpus.real_windows
    return data_corpus.real_windows + replay_sample(data_corpus.synthetic_windows, len(data_corpus.real_windows),
                                               training_settings.synthetic_replay_share, sample_generator)


def selection_score(loaded_verifier: LoadedVerifier, feature_assembler: FeatureAssembler,
                    validation_pairs: list[tuple[VerifyText, str]], window_shape: WindowShape) -> EpochValidation:
    """Mean F1 of the error classes at tuned thresholds, with per-language scores.

    Args:
        loaded_verifier: Weights to validate.
        feature_assembler: Evidence providers.
        validation_pairs: Selection texts (with labels) and their language names.
        window_shape: Window settings.

    Returns:
        The validation result.
    """
    probability_lists = predict_units(VerifierBundle([loaded_verifier], feature_assembler, window_shape),
                                      [text_entry for text_entry, _ in validation_pairs])
    gold_labels = [unit_label for text_entry, _ in validation_pairs for unit_label in text_entry.unit_labels]
    class_thresholds = tune_thresholds(gold_labels, [probability_row for probability_list in probability_lists
                                                     for probability_row in probability_list])
    score_groups = grouped_texts_scores(validation_pairs, probability_lists, class_thresholds)
    f1_values = [score_groups["ALL"][class_name]["f1"] for class_name in ERROR_CLASSES]
    return EpochValidation(sum(f1_values) / len(f1_values), class_thresholds, score_groups)


def run_stage(loaded_verifier: LoadedVerifier, stage_plan: StagePlan, run_context: "MemberRun",
              best_score: float) -> float:
    """Train and validate the epochs of one stage.

    Args:
        loaded_verifier: Network being trained.
        stage_plan: Stage.
        run_context: Corpus, settings, EMA, providers and save callback of the member.
        best_score: Best score so far (across stages).

    Returns:
        Best score after the stage.
    """
    training_settings = run_context.training_settings
    first_windows = stage_windows(stage_plan, run_context.data_corpus, training_settings, run_context.shuffle_generator)
    windows_per_update = training_settings.batch_size * training_settings.gradient_accumulation
    updates_per_epoch = -(-len(first_windows) // windows_per_update)
    loop_state = optimizer_state(loaded_verifier, training_settings, stage_plan.learning_rate,
                                 stage_plan.epoch_count * updates_per_epoch)
    stale_epochs = 0
    for epoch_number in range(1, stage_plan.epoch_count + 1):
        epoch_windows = first_windows if epoch_number == 1 else stage_windows(
            stage_plan, run_context.data_corpus, training_settings, run_context.shuffle_generator)
        epoch_loss = train_epoch(loaded_verifier, epoch_windows, loop_state, training_settings,
                                 run_context.shuffle_generator, run_context.ema_model)
        validated_weights = averaged_view(loaded_verifier, run_context.ema_model)
        validation = selection_score(validated_weights, run_context.feature_assembler,
                                     run_context.data_corpus.validation_pairs, run_context.window_shape)
        print(f"[{stage_plan.stage_name}] " + epoch_report(epoch_number, epoch_loss, validation.mean_score,
                                                           validation.class_thresholds, validation.score_groups),
              flush=True)
        if validation.mean_score > best_score:
            best_score, stale_epochs = validation.mean_score, 0
            run_context.save_callback(validated_weights, validation.class_thresholds)
        else:
            stale_epochs += 1
        if 0 < training_settings.early_stop_patience <= stale_epochs:
            print(f"[{stage_plan.stage_name}] early stop: no better validation score for {stale_epochs} epoch(s)",
                  flush=True)
            break
    return best_score


@dataclass
class MemberRun:
    """Everything one member's training needs besides the network.

    Attributes:
        data_corpus: Training windows and selection texts.
        training_settings: Settings.
        feature_assembler: Evidence providers.
        window_shape: Window settings.
        save_callback: Receives the best weights and their thresholds.
        shuffle_generator: Shuffling and sampling generator.
        ema_model: EMA copy, or None.
    """

    data_corpus: TrainingCorpus
    training_settings: TrainingSettings
    feature_assembler: FeatureAssembler
    window_shape: WindowShape
    save_callback: SaveCallback
    shuffle_generator: random.Random
    ema_model: AveragedModel | None


def fit_member(loaded_verifier: LoadedVerifier, run_context: MemberRun) -> float:
    """Run all stages of one member.

    Usage:
        Called by ``training_service`` for every ensemble member and by ``relabel_service`` for every fold.

    Args:
        loaded_verifier: Network with a fresh head.
        run_context: Member run context (its ``ema_model`` is created here when enabled).

    Returns:
        Best validation mean F1.
    """
    run_context.ema_model = averaged_weights(loaded_verifier, run_context.training_settings.ema_decay)
    best_score = -1.0
    for stage_plan in stage_plans(run_context.training_settings, run_context.data_corpus):
        best_score = run_stage(loaded_verifier, stage_plan, run_context, best_score)
    return best_score
