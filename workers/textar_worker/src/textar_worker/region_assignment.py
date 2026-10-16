"""Assign detected words to the requested text regions and order them for reading."""

from collections.abc import Sequence

from textar_worker.attribute_inference import WordAttributes

RegionBox = Sequence[float]


def reading_order(word_boxes: Sequence[Sequence[int]]) -> list[int]:
    """Indexes of the boxes sorted top-to-bottom, then left-to-right.

    Args:
        word_boxes: ``(x0, y0, x1, y1)`` boxes.

    Returns:
        Permutation of ``range(len(word_boxes))``.
    """
    return sorted(range(len(word_boxes)), key=lambda word_index: (word_boxes[word_index][1], word_boxes[word_index][0]))


def containing_region(word_box: Sequence[int], text_regions: Sequence[RegionBox]) -> int | None:
    """Smallest region that contains the centre of a word.

    Args:
        word_box: ``(x0, y0, x1, y1)`` box.
        text_regions: Region boxes.

    Returns:
        Region index, or None when the word lies outside every region.
    """
    centre_x = (word_box[0] + word_box[2]) / 2
    centre_y = (word_box[1] + word_box[3]) / 2
    matching_regions = [((region_box[2] - region_box[0]) * (region_box[3] - region_box[1]), region_index)
                        for region_index, region_box in enumerate(text_regions)
                        if region_box[0] <= centre_x <= region_box[2] and region_box[1] <= centre_y <= region_box[3]]
    return min(matching_regions)[1] if matching_regions else None


def word_payload(word_attributes: WordAttributes) -> dict:
    """JSON form of one word (see ``worker_protocol.style_job_schema``).

    Args:
        word_attributes: Predicted attributes.

    Returns:
        Dictionary with bbox, classes and rounded scores.
    """
    return {"bbox": list(word_attributes.word_box),
            "emphasis_class": word_attributes.emphasis_class,
            "emphasis_score": round(word_attributes.emphasis_score, 4),
            "line_class": word_attributes.line_class, "line_score": round(word_attributes.line_score, 4)}


def group_by_region(page_words: Sequence[WordAttributes], text_regions: Sequence[RegionBox]) -> list[list[dict]]:
    """Distribute the words of a page over its text regions.

    Args:
        page_words: Words in reading order.
        text_regions: Requested regions.

    Returns:
        One list of word payloads per region (words outside all regions are dropped).
    """
    region_words: list[list[dict]] = [[] for _ in text_regions]
    for word_attributes in page_words:
        region_index = containing_region(word_attributes.word_box, text_regions)
        if region_index is not None:
            region_words[region_index].append(word_payload(word_attributes))
    return region_words
