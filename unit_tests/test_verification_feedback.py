"""Unit tests of the verifier feedback loop with a fake verifier and a fake OCR reader."""

from pathlib import Path

from pdf2fb2.model_stages.verify_stage import VerifyOutcome
from pdf2fb2.text_verification.feedback_loop import VerificationFeedback
from pdf2fb2.text_verification.text_splicing import TextEdit, apply_edits
from pdf2fb2.text_verification.uncertain_units import is_uncertain, judged_row
from pdf2fb2.text_verification.unit_evidence import confidence_columns, page_contexts, reread_columns
from pdf2fb2.text_verification.unit_flags import error_mass, flag_units
from pdf2fb2.text_verification.verify_units import verify_units
from worker_protocol.verify_job_schema import VerifyItem

THRESHOLDS = {"misread": 0.5, "inserted": 0.5}
BAD_WORDS = {"rnodern": [0.1, 0.9, 0.0], "HALLUCINATEDCONTINUATION": [0.1, 0.0, 0.9]}


class SilentProgress:
    """Progress sink that ignores everything (test double)."""

    def advance_stage(self, done_count: int, total_count: int, step_text: str) -> None:
        """Ignore progress.

        Args:
            done_count: Done units.
            total_count: All units.
            step_text: Step description.

        Returns:
            None.
        """

    def report_message(self, message_text: str) -> None:
        """Ignore messages.

        Args:
            message_text: Message.

        Returns:
            None.
        """


class FakeLabeler:
    """Labels known bad words, everything else correct; records its jobs."""

    def __init__(self) -> None:
        """Start without jobs."""
        self.job_names: list[str] = []
        self.job_items: list[list[VerifyItem]] = []

    def label_items(self, verify_items: list[VerifyItem], work_directory: Path, job_name: str,
                    stage_progress: SilentProgress) -> VerifyOutcome:
        """Probabilities of items (recorded) from the ``BAD_WORDS`` table.

        Args:
            verify_items: Items.
            work_directory: Unused.
            job_name: Recorded.
            stage_progress: Unused.

        Returns:
            Outcome with the test thresholds.
        """
        self.job_items.append(list(verify_items))
        return self.label_units({verify_item.item_key: list(verify_item.item_units) for verify_item in verify_items},
                                work_directory, job_name, stage_progress)

    def label_units(self, unit_lists: dict[str, list[str]], work_directory: Path, job_name: str,
                    stage_progress: SilentProgress) -> VerifyOutcome:
        """Probabilities from the ``BAD_WORDS`` table.

        Args:
            unit_lists: Item key -> units.
            work_directory: Unused.
            job_name: Recorded.
            stage_progress: Unused.

        Returns:
            Outcome with the test thresholds.
        """
        self.job_names.append(job_name)
        return VerifyOutcome({item_key: [BAD_WORDS.get(unit_text, [0.95, 0.03, 0.02]) for unit_text in item_units]
                              for item_key, item_units in unit_lists.items()}, THRESHOLDS)


class FakeReader:
    """OCR double: every re-read region reads ``the modern age``."""

    def __init__(self) -> None:
        """Start without calls."""
        self.read_boxes: list[list[int]] = []

    def recognize_segments(self, page_records: list[dict], work_directory: Path,
                           stage_progress: SilentProgress) -> None:
        """Set the text of every region segment.

        Args:
            page_records: Pseudo page records (mutated).
            work_directory: Unused.
            stage_progress: Unused.

        Returns:
            None.
        """
        for page_record in page_records:
            for page_segment in page_record["segments"]:
                self.read_boxes.append(page_segment["bbox"])
                page_segment["text"] = "the modern age"


def word_box(left_edge: int, top_edge: int, character_count: int) -> dict:
    """Detected word of 10 px per character.

    Args:
        left_edge: Left x.
        top_edge: Top y.
        character_count: Characters of the word.

    Returns:
        Word entry.
    """
    return {"bbox": [left_edge, top_edge, left_edge + 10 * character_count, top_edge + 20]}


def sample_pages() -> list[dict]:
    """One page, one paragraph of two lines; the hallucinated unit has no word box.

    Returns:
        Page records.
    """
    segment_words = [word_box(0, 0, 3), word_box(40, 0, 6), word_box(110, 0, 3),
                     word_box(0, 40, 4), word_box(50, 40, 6)]
    return [{"page": 1, "image": "page.png", "size": [500, 300], "segments": [
        {"bbox": [0, 0, 200, 70], "label": "text", "text": "the rnodern age HALLUCINATEDCONTINUATION went fast.",
         "words": segment_words}]}]


def test_flags_and_error_mass() -> None:
    """Inserted wins over misread; error mass skips deleted units."""
    unit_flags = flag_units([[0.1, 0.6, 0.6], [0.2, 0.8, 0.0], [0.9, 0.05, 0.05]], THRESHOLDS)
    assert unit_flags.inserted_units == (0,) and unit_flags.misread_units == (1,)
    assert error_mass([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], frozenset({0})) == 0.0


def test_units_split_cjk_and_keep_offsets() -> None:
    """CJK characters are single units, other text splits on whitespace."""
    assert [verify_unit.unit_text for verify_unit in verify_units("赤い腦 the end")] == ["赤", "い", "腦", "the", "end"]
    assert verify_units("a  bc")[1].start_offset == 3


def test_edits_apply_from_the_end() -> None:
    """Edits are applied right to left so earlier offsets stay valid."""
    assert apply_edits("one two three", [TextEdit(0, 3, "1"), TextEdit(8, 13, "3")]) == "1 two 3"


def test_feedback_rereads_misread_line_and_deletes_unprinted_text(tmp_path) -> None:
    """The misread line is re-read from its own crop; the unprinted unit is deleted; the result is kept."""
    page_records = sample_pages()
    fake_labeler, fake_reader = FakeLabeler(), FakeReader()
    verify_summary = VerificationFeedback(fake_labeler, fake_reader).verify_segments(
        page_records, tmp_path, SilentProgress())
    assert fake_labeler.job_names == ["verify_job", "verify_recheck_job"]
    assert len(fake_reader.read_boxes) == 1 and fake_reader.read_boxes[0][3] < 40
    page_segment = page_records[0]["segments"][0]
    assert page_segment["text"] == "the modern age went fast."
    assert page_segment["pre_verify_text"].startswith("the rnodern")
    assert verify_summary["changed_segments"] == 1 and verify_summary["deleted_units"] == 1
    assert (tmp_path / "verify_report.json").exists()
    first_item = fake_labeler.job_items[0][0]
    assert first_item.unit_features["box_width_ratio"][3] < 0.5 < first_item.unit_features["box_width_ratio"][0]
    assert sorted(verify_item.item_key for verify_item in fake_labeler.job_items[1]) == ["1:0|candidate",
                                                                                        "1:0|original"]


def test_unit_evidence_maps_confidence_and_second_read() -> None:
    """Confidences follow the raw text through cleaning; a different second read marks the unit."""
    segment_text = "the modern age"
    segment_units = verify_units(segment_text)
    confidence = {"raw_text": "**the modern age**", "char_logprobs": [0.0, 0.0] + [-0.1] * 4 + [-3.0] * 6 +
                  [-0.2] * 4 + [0.0, 0.0]}
    confidence_cols = confidence_columns(segment_text, segment_units, confidence)
    assert confidence_cols["ocr_min_logprob"][1] == -3.0 and confidence_cols["ocr_min_logprob"][0] == -0.1
    assert reread_columns(segment_units, "the madern age")["reread_disagrees"] == (0.0, 1.0, 0.0)
    assert page_contexts([["a", "b"], ["c"], ["d", "e"]]) == [((), ("c", "d", "e")), (("a", "b"), ("d", "e")),
                                                             (("a", "b", "c"), ())]


def test_judge_blends_uncertain_rows() -> None:
    """The judge's error probability is averaged with the verifier's, keeping the misread/inserted split."""
    assert is_uncertain([0.6, 0.3, 0.1], (0.15, 0.6)) and not is_uncertain([0.95, 0.03, 0.02], (0.15, 0.6))
    combined_row = judged_row([0.6, 0.3, 0.1], 1.0)
    assert abs(combined_row[0] - 0.3) < 1e-9 and abs(combined_row[1] - 0.525) < 1e-9


def test_temporary_images_are_removed_after_verification(tmp_path) -> None:
    """Crop directories go away when verification ends; the second read's text cache stays."""
    for relative_name in ("verify_crops", "judge_lines", "second_read/ocr_crops"):
        (tmp_path / relative_name).mkdir(parents=True)
        (tmp_path / relative_name / "crop.png").write_bytes(b"png")
    (tmp_path / "second_read" / "ocr_cache.json").write_text("{}")
    VerificationFeedback(FakeLabeler(), FakeReader()).verify_segments(sample_pages(), tmp_path, SilentProgress())
    assert not any((tmp_path / relative_name).exists()
                   for relative_name in ("verify_crops", "judge_lines", "second_read/ocr_crops"))
    assert (tmp_path / "second_read" / "ocr_cache.json").exists() and (tmp_path / "verify_report.json").exists()
