"""Proofreading report: the probability of every token of the FB2 text and every edit with its reason."""

import json
import math
from pathlib import Path

from pdf2fb2.proofreading.edit_decisions import FlowEdits
from pdf2fb2.proofreading.flow_blocks import FlowBlock

PROBABILITY_DIGITS = 6


def block_report(flow_block: FlowBlock, result_block: dict) -> dict:
    """Words of one block with their token probabilities.

    Args:
        flow_block: Block sent to the worker.
        result_block: Worker result entry of the block.

    Returns:
        ``{"kind", "words": [{"word", "probability", "tokens": [[text, probability], ...]}]}``;
        ``probability`` of a word is the product of its token probabilities.
    """
    word_reports = []
    for word_text, word_result in zip(flow_block.plain_words, result_block["words"], strict=True):
        word_logprob = sum(token_entry[1] for token_entry in word_result["tokens"])
        word_reports.append({"word": word_text, "probability": round(math.exp(word_logprob), PROBABILITY_DIGITS),
                             "tokens": [[token_entry[0], round(math.exp(token_entry[1]), PROBABILITY_DIGITS)]
                                        for token_entry in word_result["tokens"]]})
    return {"kind": flow_block.block_kind, "words": word_reports}


def write_proofread_report(report_path: Path, flow_blocks: list[FlowBlock], worker_result: dict,
                           flow_edits: FlowEdits) -> dict:
    """Write the report and return its summary.

    Usage:
        Call after the edits are decided; the report describes the text before editing.

    Args:
        report_path: Output JSON file.
        flow_blocks: Blocks sent to the worker.
        worker_result: Worker result document.
        flow_edits: Decided edits.

    Returns:
        Summary counts (deleted words, deleted headings, deleted paragraphs, merged paragraphs).
    """
    deleted_kinds = [flow_blocks[block_index].block_kind for block_index in flow_edits.deleted_blocks]
    edit_summary = {"deleted_words": len(flow_edits.deleted_words), "deleted_headings": deleted_kinds.count("heading"),
                    "deleted_paragraphs": deleted_kinds.count("p"), "merged_paragraphs": len(flow_edits.merged_blocks)}
    report_document = {"summary": edit_summary, "edits": list(flow_edits.decision_notes),
                       "blocks": [block_report(flow_block, result_block) for flow_block, result_block
                                  in zip(flow_blocks, worker_result["blocks"], strict=True)]}
    report_path.write_text(json.dumps(report_document, ensure_ascii=False, indent=1), encoding="utf-8")
    return edit_summary
