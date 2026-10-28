"""Encoders the verifier can be built on, behind one interface (sub-word or byte level, any language).

* ``SubwordBackbone``: any Hugging Face encoder with a fast tokenizer (XLM-RoBERTa, mDeBERTa-v3). Keeps the
  pretrained multilingual embedding space, so similar words of different languages stay close.
* ``ByteBackbone``: the ByT5 encoder; every UTF-8 byte is one input, so a single-letter OCR change is visible to
  the model directly instead of being hidden by a sub-word vocabulary.

An adapter turns ``context_before + units + context_after`` into input ids and records, for every input piece,
the index of the unit it belongs to (``-1`` for context, separators and special pieces).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

BYTE_OFFSET = 3          # ByT5: 0 pad, 1 end of sequence, 2 unknown, then bytes
BYTE_PAD_ID = 0
BYTE_END_ID = 1
SEPARATOR_TEXT = "\n"
NO_UNIT_INDEX = -1
BYTE_MARKER_FILE = "byt5_marker"


@dataclass(frozen=True)
class PieceEncoding:
    """Encoder input of one window.

    Attributes:
        input_identifiers: Piece ids, special pieces included.
        piece_units: Unit index (within the window) of every piece, ``NO_UNIT_INDEX`` for other pieces.
    """

    input_identifiers: tuple[int, ...]
    piece_units: tuple[int, ...]


class BackboneAdapter(Protocol):
    """Encoder plus its input encoding.

    Attributes:
        encoder_module: The torch encoder.
        padding_identifier: Id used to pad batches.
        hidden_size: Width of the encoder states.
        layer_count: Number of transformer layers (for layer-wise learning-rate decay).
    """

    encoder_module: torch.nn.Module
    padding_identifier: int
    hidden_size: int
    layer_count: int

    def encode_units(self, unit_texts: list[str], context_before: list[str], context_after: list[str],
                     max_tokens: int) -> PieceEncoding:
        """Encode a window with its context.

        Args:
            unit_texts: Units to label.
            context_before: Unlabelled units before them.
            context_after: Unlabelled units after them.
            max_tokens: Piece limit; context is shortened first, then the window is truncated.

        Returns:
            The encoding.
        """
        ...

    def hidden_states(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Encoder states of a batch.

        Args:
            input_tensor: ``(batch, pieces)`` ids.
            mask_tensor: ``(batch, pieces)`` attention mask.

        Returns:
            ``(batch, pieces, hidden_size)`` states.
        """
        ...

    def save_backbone(self, output_directory: Path) -> None:
        """Write the encoder (and tokenizer) to a directory.

        Args:
            output_directory: Target directory.

        Returns:
            None.
        """
        ...


def shortened_contexts(context_before: list[str], context_after: list[str]) -> tuple[list[str], list[str]]:
    """Halve both contexts, keeping the units nearest to the window.

    Args:
        context_before: Units before the window.
        context_after: Units after the window.

    Returns:
        Shorter contexts (empty when they were at most one unit).
    """
    return context_before[len(context_before) - len(context_before) // 2:] if len(context_before) > 1 else [], \
        context_after[:len(context_after) // 2]


class SubwordBackbone:
    """Encoder with a fast sub-word tokenizer (``word_ids`` support).

    Usage:
        Built by ``load_backbone`` for XLM-RoBERTa, mDeBERTa-v3 and other ``AutoModel`` encoders.
    """

    def __init__(self, text_tokenizer: object, encoder_module: torch.nn.Module) -> None:
        """Bind tokenizer and encoder.

        Args:
            text_tokenizer: Fast tokenizer.
            encoder_module: ``AutoModel`` encoder.
        """
        self.text_tokenizer = text_tokenizer
        self.encoder_module = encoder_module
        self.padding_identifier = text_tokenizer.pad_token_id or 0
        self.hidden_size = int(encoder_module.config.hidden_size)
        self.layer_count = int(encoder_module.config.num_hidden_layers)

    def _encode_words(self, unit_texts: list[str], context_before: list[str], context_after: list[str],
                      max_tokens: int) -> tuple[object, int]:
        """Tokenise context and units as one word list.

        Args:
            unit_texts: Units to label.
            context_before: Units before them.
            context_after: Units after them.
            max_tokens: Piece limit.

        Returns:
            ``(encoding, index of the first unit word)``.
        """
        separator_text = self.text_tokenizer.sep_token or SEPARATOR_TEXT
        before_words = [*context_before, separator_text] if context_before else []
        after_words = [separator_text, *context_after] if context_after else []
        encoding = self.text_tokenizer(before_words + unit_texts + after_words, is_split_into_words=True,
                                       truncation=True, max_length=max_tokens)
        return encoding, len(before_words)

    def encode_units(self, unit_texts: list[str], context_before: list[str], context_after: list[str],
                     max_tokens: int) -> PieceEncoding:
        """Encode a window with its context (see ``BackboneAdapter.encode_units``).

        Args:
            unit_texts: Units to label.
            context_before: Unlabelled units before them.
            context_after: Unlabelled units after them.
            max_tokens: Piece limit.

        Returns:
            The encoding.
        """
        while True:
            encoding, first_word = self._encode_words(unit_texts, context_before, context_after, max_tokens)
            word_indexes = encoding.word_ids()
            fits_whole = len(encoding["input_ids"]) < max_tokens or not (context_before or context_after)
            if fits_whole:
                break
            context_before, context_after = shortened_contexts(context_before, context_after)
        piece_units = tuple(word_index - first_word if word_index is not None
                            and first_word <= word_index < first_word + len(unit_texts) else NO_UNIT_INDEX
                            for word_index in word_indexes)
        return PieceEncoding(tuple(encoding["input_ids"]), piece_units)

    def hidden_states(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Encoder states of a batch (see ``BackboneAdapter.hidden_states``).

        Args:
            input_tensor: Ids.
            mask_tensor: Attention mask.

        Returns:
            Last hidden states.
        """
        return self.encoder_module(input_ids=input_tensor, attention_mask=mask_tensor).last_hidden_state

    def save_backbone(self, output_directory: Path) -> None:
        """Write encoder and tokenizer.

        Args:
            output_directory: Target directory.

        Returns:
            None.
        """
        self.encoder_module.save_pretrained(str(output_directory))
        self.text_tokenizer.save_pretrained(str(output_directory))


def byte_pieces(word_list: list[str], unit_offset: int | None) -> tuple[list[int], list[int]]:
    """Byte ids of words joined by spaces, with the unit index of every byte.

    Args:
        word_list: Words to encode.
        unit_offset: Unit index of the first word, or None for context words.

    Returns:
        ``(ids, piece units)``.
    """
    piece_ids: list[int] = []
    piece_units: list[int] = []
    for word_position, word_text in enumerate(word_list):
        if word_position:
            piece_ids.append(ord(" ") + BYTE_OFFSET)
            piece_units.append(NO_UNIT_INDEX)
        word_bytes = word_text.encode("utf-8")
        piece_ids.extend(byte_value + BYTE_OFFSET for byte_value in word_bytes)
        piece_units.extend([NO_UNIT_INDEX if unit_offset is None else unit_offset + word_position] * len(word_bytes))
    return piece_ids, piece_units


class ByteBackbone:
    """ByT5 encoder over UTF-8 bytes.

    Usage:
        Built by ``load_backbone`` when the base model is a ByT5 checkpoint (``models/byt5-small``).
    """

    def __init__(self, encoder_module: torch.nn.Module) -> None:
        """Bind the encoder.

        Args:
            encoder_module: ``T5EncoderModel``.
        """
        self.encoder_module = encoder_module
        self.padding_identifier = BYTE_PAD_ID
        self.hidden_size = int(encoder_module.config.d_model)
        self.layer_count = int(encoder_module.config.num_layers)

    def _assembled(self, unit_texts: list[str], context_before: list[str],
                   context_after: list[str]) -> tuple[list[int], list[int]]:
        """Bytes of context, separator, units, separator, context and end-of-sequence.

        Args:
            unit_texts: Units to label.
            context_before: Units before them.
            context_after: Units after them.

        Returns:
            ``(ids, piece units)`` before truncation.
        """
        separator_ids = [byte_value + BYTE_OFFSET for byte_value in SEPARATOR_TEXT.encode("utf-8")]
        piece_ids, piece_units = [], []
        for word_list, unit_offset in ((context_before, None), (unit_texts, 0), (context_after, None)):
            if not word_list:
                continue
            if piece_ids:
                piece_ids.extend(separator_ids)
                piece_units.extend([NO_UNIT_INDEX] * len(separator_ids))
            part_ids, part_units = byte_pieces(word_list, unit_offset)
            piece_ids.extend(part_ids)
            piece_units.extend(part_units)
        return piece_ids + [BYTE_END_ID], piece_units + [NO_UNIT_INDEX]

    def encode_units(self, unit_texts: list[str], context_before: list[str], context_after: list[str],
                     max_tokens: int) -> PieceEncoding:
        """Encode a window with its context (see ``BackboneAdapter.encode_units``).

        Args:
            unit_texts: Units to label.
            context_before: Unlabelled units before them.
            context_after: Unlabelled units after them.
            max_tokens: Byte limit.

        Returns:
            The encoding.
        """
        piece_ids, piece_units = self._assembled(unit_texts, context_before, context_after)
        while len(piece_ids) > max_tokens and (context_before or context_after):
            context_before, context_after = shortened_contexts(context_before, context_after)
            piece_ids, piece_units = self._assembled(unit_texts, context_before, context_after)
        if len(piece_ids) > max_tokens:
            piece_ids, piece_units = piece_ids[:max_tokens - 1] + [BYTE_END_ID], piece_units[:max_tokens - 1] + [
                NO_UNIT_INDEX]
        return PieceEncoding(tuple(piece_ids), tuple(piece_units))

    def hidden_states(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Encoder states of a batch (see ``BackboneAdapter.hidden_states``).

        Args:
            input_tensor: Byte ids.
            mask_tensor: Attention mask.

        Returns:
            Last hidden states.
        """
        return self.encoder_module(input_ids=input_tensor, attention_mask=mask_tensor).last_hidden_state

    def save_backbone(self, output_directory: Path) -> None:
        """Write the encoder (bytes need no tokenizer files).

        Args:
            output_directory: Target directory.

        Returns:
            None.
        """
        self.encoder_module.save_pretrained(str(output_directory))
        (output_directory / BYTE_MARKER_FILE).write_text("byte-level backbone\n", encoding="utf-8")


def backbone_kind(model_directory: Path) -> str:
    """Tell which adapter a checkpoint needs.

    Args:
        model_directory: Local checkpoint.

    Returns:
        ``byte`` for ByT5 checkpoints, otherwise ``subword``.
    """
    config_values = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    tokenizer_names = " ".join(str(config_values.get(key_name, ""))
                               for key_name in ("tokenizer_class", "_name_or_path"))
    return "byte" if "byt5" in tokenizer_names.lower() or (model_directory / BYTE_MARKER_FILE).exists() else "subword"


def load_subword_backbone(model_directory: Path) -> SubwordBackbone:
    """Load a sub-word encoder from local files.

    Args:
        model_directory: Checkpoint directory.

    Returns:
        The adapter.
    """
    # B615: weights and tokenizer are loaded from a local directory only (local_files_only=True).
    text_tokenizer = AutoTokenizer.from_pretrained(str(model_directory), local_files_only=True)  # nosec B615
    encoder_module = AutoModel.from_pretrained(str(model_directory), local_files_only=True)  # nosec B615
    return SubwordBackbone(text_tokenizer, encoder_module)


def load_byte_backbone(model_directory: Path) -> ByteBackbone:
    """Load the ByT5 encoder from local files (the decoder is not loaded).

    Args:
        model_directory: Checkpoint directory.

    Returns:
        The adapter.
    """
    # B615: local directory only.
    return ByteBackbone(T5EncoderModel.from_pretrained(str(model_directory), local_files_only=True))  # nosec B615


BACKBONE_LOADERS = {"subword": load_subword_backbone, "byte": load_byte_backbone}


def load_backbone(model_directory: Path, kind_name: str = "auto") -> tuple[BackboneAdapter, str]:
    """Load an encoder with the adapter its checkpoint needs.

    Usage:
        Called when training starts (base model) and when a fine-tuned verifier is loaded (its ``backbone``).

    Args:
        model_directory: Local checkpoint.
        kind_name: ``subword``, ``byte`` or ``auto`` (detected from the checkpoint).

    Returns:
        ``(adapter, kind)``.

    Raises:
        KeyError: For an unknown kind.
        OSError: If the directory does not hold a model.
    """
    resolved_kind = backbone_kind(model_directory) if kind_name == "auto" else kind_name
    return BACKBONE_LOADERS[resolved_kind](model_directory), resolved_kind
