"""Per-token log-probabilities from the causal language model, batched and length-bucketed.

Logits are computed only at the positions that are measured, in chunks, so the 248k-entry
vocabulary never materialises for a whole batch. Sequences are right-padded to a multiple of 64
tokens (fewer distinct shapes for the Triton kernels); right padding never changes earlier
positions of a causal model.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from proofread_worker.model_loading import LoadedLanguageModel

PASSAGE_LEAD_IN = "The following is a passage from a book.\n\n"
PADDING_MULTIPLE = 64
TOKENS_PER_BATCH = 8192
LOGIT_CHUNK = 512
WINDOW_TOKENS = 1024
WINDOW_CONTEXT = 256

BatchCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class ScoredToken:
    """One token of a scored text.

    Attributes:
        start_offset: First character of the token in the scored text (without the lead-in).
        end_offset: Character after the token.
        token_logprob: Natural-log probability of the token given everything before it.
    """

    start_offset: int
    end_offset: int
    token_logprob: float


class TokenScorer:
    """Scores texts with the loaded language model.

    Usage:
        Build once per job; use ``score_document`` for the whole book and ``continuation_logprobs``
        for the deletion tests.
    """

    def __init__(self, language_model: LoadedLanguageModel) -> None:
        """Store the model.

        Args:
            language_model: Loaded tokenizer and model.
        """
        self._language_model = language_model
        self._padding_id = language_model.tokenizer.pad_token_id or 0

    def encode_text(self, scored_text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Tokenise lead-in + text.

        Args:
            scored_text: Text to score.

        Returns:
            ``(token_ids, offsets)``; offsets are relative to ``scored_text`` (lead-in tokens negative).
        """
        encoding = self._language_model.tokenizer(PASSAGE_LEAD_IN + scored_text, return_offsets_mapping=True,
                                                  add_special_tokens=False)
        return encoding["input_ids"], [(token_start - len(PASSAGE_LEAD_IN), token_end - len(PASSAGE_LEAD_IN))
                                       for token_start, token_end in encoding["offset_mapping"]]

    def score_document(self, document_text: str, batch_callback: BatchCallback) -> list[ScoredToken]:
        """Log-probability of every token of the document (sliding windows with 256 tokens of context).

        Args:
            document_text: Whole book text.
            batch_callback: Called with ``(done, total)`` windows after every batch.

        Returns:
            Scored tokens of the document in order (lead-in excluded).
        """
        token_ids, token_offsets = self.encode_text(document_text)
        first_scored = next((token_index for token_index, (_, token_end) in enumerate(token_offsets)
                             if token_end > 0), len(token_ids))
        window_requests = []
        for window_start in range(max(1, first_scored), len(token_ids), WINDOW_TOKENS - WINDOW_CONTEXT):
            context_start = max(0, window_start - WINDOW_CONTEXT)
            window_end = min(len(token_ids), window_start + WINDOW_TOKENS - WINDOW_CONTEXT)
            window_requests.append((token_ids[context_start:window_end],
                                    list(range(window_start - context_start, window_end - context_start)),
                                    context_start))
        window_logprobs = self.measure_positions([(window_ids, measured) for window_ids, measured, _
                                                  in window_requests], batch_callback)
        scored_tokens = []
        for (_, measured, context_start), measured_logprobs in zip(window_requests, window_logprobs, strict=True):
            for local_position, token_logprob in zip(measured, measured_logprobs, strict=True):
                token_start, token_end = token_offsets[context_start + local_position]
                scored_tokens.append(ScoredToken(token_start, token_end, token_logprob))
        return scored_tokens

    def continuation_logprobs(self, scored_texts: list[tuple[str, int]], batch_callback: BatchCallback) -> list[float]:
        """Summed log-probability of the tokens that end after a given character.

        Args:
            scored_texts: ``(text, measure_from)`` pairs; tokens with end offset > ``measure_from`` count.
            batch_callback: Called with ``(done, total)`` texts after every batch.

        Returns:
            One sum per text (0.0 when nothing is measured).
        """
        requests = []
        for scored_text, measure_from in scored_texts:
            token_ids, token_offsets = self.encode_text(scored_text)
            requests.append((token_ids, [token_index for token_index, (_, token_end) in enumerate(token_offsets)
                                         if token_index > 0 and token_end > measure_from]))
        return [sum(position_logprobs) for position_logprobs in self.measure_positions(requests, batch_callback)]

    def measure_positions(self, requests: list[tuple[list[int], list[int]]],
                          batch_callback: BatchCallback) -> list[list[float]]:
        """Log-probabilities of selected positions of many sequences.

        Args:
            requests: ``(token_ids, positions)``; every position must be >= 1.
            batch_callback: Called with ``(done, total)`` sequences after every batch.

        Returns:
            Log-probabilities per request, in the order of its positions.
        """
        request_order = sorted(range(len(requests)), key=lambda request_index: len(requests[request_index][0]))
        request_results: list[list[float]] = [[] for _ in requests]
        finished_count = 0
        for batch_indexes in length_batches(request_order, requests):
            batch_results = self._run_batch([requests[request_index] for request_index in batch_indexes])
            for request_index, position_logprobs in zip(batch_indexes, batch_results, strict=True):
                request_results[request_index] = position_logprobs
            finished_count += len(batch_indexes)
            batch_callback(finished_count, len(requests))
        return request_results

    @torch.inference_mode()
    def _run_batch(self, batch_requests: list[tuple[list[int], list[int]]]) -> list[list[float]]:
        """Forward one padded batch and gather the requested log-probabilities.

        Args:
            batch_requests: ``(token_ids, positions)`` of the batch.

        Returns:
            Log-probabilities per request.
        """
        padded_length = padded_size(max(len(token_ids) for token_ids, _ in batch_requests))
        input_ids = torch.full((len(batch_requests), padded_length), self._padding_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row_index, (token_ids, _) in enumerate(batch_requests):
            input_ids[row_index, :len(token_ids)] = torch.tensor(token_ids)
            attention_mask[row_index, :len(token_ids)] = 1
        model_device = self._language_model.torch_device
        hidden_states = self._language_model.decoder_model(
            input_ids=input_ids.to(model_device), attention_mask=attention_mask.to(model_device)).last_hidden_state
        row_indexes = [row_index for row_index, (_, positions) in enumerate(batch_requests) for _ in positions]
        target_positions = [position for _, positions in batch_requests for position in positions]
        gathered_logprobs = self._gather_logprobs(hidden_states, input_ids.to(model_device), row_indexes,
                                                  target_positions)
        batch_results, read_offset = [], 0
        for _, positions in batch_requests:
            batch_results.append(gathered_logprobs[read_offset:read_offset + len(positions)])
            read_offset += len(positions)
        return batch_results

    def _gather_logprobs(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, row_indexes: list[int],
                         target_positions: list[int]) -> list[float]:
        """Log-probability of ``input_ids[row, position]`` predicted from ``hidden[row, position - 1]``.

        Args:
            hidden_states: ``(batch, length, hidden)`` decoder output.
            input_ids: ``(batch, length)`` token ids on the model device.
            row_indexes: Row of every target.
            target_positions: Position of every target (>= 1).

        Returns:
            One log-probability per target.
        """
        gathered_logprobs: list[float] = []
        for chunk_start in range(0, len(target_positions), LOGIT_CHUNK):
            chunk_rows = torch.tensor(row_indexes[chunk_start:chunk_start + LOGIT_CHUNK], device=input_ids.device)
            chunk_positions = torch.tensor(target_positions[chunk_start:chunk_start + LOGIT_CHUNK],
                                           device=input_ids.device)
            chunk_logits = self._language_model.output_head(hidden_states[chunk_rows, chunk_positions - 1]).float()
            chunk_logprobs = torch.log_softmax(chunk_logits, dim=-1).gather(
                1, input_ids[chunk_rows, chunk_positions].unsqueeze(1)).squeeze(1)
            gathered_logprobs.extend(chunk_logprobs.cpu().tolist())
        return gathered_logprobs


def padded_size(sequence_length: int) -> int:
    """Round a sequence length up to the padding multiple.

    Args:
        sequence_length: Real length.

    Returns:
        Padded length.
    """
    return -(-sequence_length // PADDING_MULTIPLE) * PADDING_MULTIPLE


def length_batches(request_order: list[int], requests: list[tuple[list[int], list[int]]]) -> list[list[int]]:
    """Group length-sorted requests into batches of at most ``TOKENS_PER_BATCH`` padded tokens.

    Args:
        request_order: Request indexes sorted by length.
        requests: All requests.

    Returns:
        Batches of request indexes.
    """
    request_batches: list[list[int]] = []
    current_batch: list[int] = []
    for request_index in request_order:
        candidate_length = padded_size(len(requests[request_index][0]))
        if current_batch and candidate_length * (len(current_batch) + 1) > TOKENS_PER_BATCH:
            request_batches.append(current_batch)
            current_batch = []
        current_batch.append(request_index)
    if current_batch:
        request_batches.append(current_batch)
    return request_batches
