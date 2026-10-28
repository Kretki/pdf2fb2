"""Locked train / valid / test split of the benchmark books, by book (never by page) and per language.

Books are ordered by a SHA-256 hash of their key and the first share of every language goes to ``test``, the
next to ``valid``, the rest to ``train``, with at least one test and one valid book when a language has three or
more. The split is written once to ``splits.json`` and reused, so a later catalog change only assigns the new
books. Hand-proofread ``old_russian`` books are the only true ground truth: half of them are kept for testing.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pdf2fb2.benchmark.corpus_catalog import CatalogBook

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
SPLIT_NAMES = (TRAIN_SPLIT, VALID_SPLIT, TEST_SPLIT)
DEFAULT_SHARES = {TEST_SPLIT: 0.15, VALID_SPLIT: 0.15}
LANGUAGE_SHARES = {"old_russian": {TEST_SPLIT: 0.5, VALID_SPLIT: 0.2}}
SPLITS_FILE = "splits.json"


def hashed_order(book_keys: list[str]) -> list[str]:
    """Book keys in a stable pseudo-random order.

    Args:
        book_keys: Keys of one language.

    Returns:
        Keys sorted by the SHA-256 of their UTF-8 bytes.
    """
    return sorted(book_keys, key=lambda book_key: hashlib.sha256(book_key.encode("utf-8")).hexdigest())


def language_split(book_keys: list[str], language_name: str) -> dict[str, str]:
    """Split the books of one language.

    Args:
        book_keys: Keys of the language's books.
        language_name: Corpus language folder.

    Returns:
        Mapping book key -> split name.
    """
    split_shares = LANGUAGE_SHARES.get(language_name, DEFAULT_SHARES)
    ordered_keys = hashed_order(book_keys)
    minimum_count = 1 if len(ordered_keys) >= 3 else 0
    test_count = max(minimum_count, round(len(ordered_keys) * split_shares[TEST_SPLIT]))
    valid_count = max(minimum_count, round(len(ordered_keys) * split_shares[VALID_SPLIT]))
    assigned_splits = {}
    for order_index, book_key in enumerate(ordered_keys):
        if order_index < test_count:
            assigned_splits[book_key] = TEST_SPLIT
        elif order_index < test_count + valid_count:
            assigned_splits[book_key] = VALID_SPLIT
        else:
            assigned_splits[book_key] = TRAIN_SPLIT
    return assigned_splits


def assign_splits(catalog_books: list[CatalogBook]) -> dict[str, str]:
    """Split all books, language by language.

    Args:
        catalog_books: Benchmark books.

    Returns:
        Mapping book key -> split name.
    """
    keys_by_language: dict[str, list[str]] = defaultdict(list)
    for catalog_book in catalog_books:
        keys_by_language[catalog_book.language_name].append(catalog_book.book_key)
    return {book_key: split_name for language_name, book_keys in keys_by_language.items()
            for book_key, split_name in language_split(book_keys, language_name).items()}


def locked_splits(catalog_books: list[CatalogBook], data_directory: Path) -> dict[str, str]:
    """Split stored in ``splits.json``, created or extended for books it does not know yet.

    Usage:
        Call before every run or export so all of them see the same split.

    Args:
        catalog_books: Benchmark books.
        data_directory: Directory holding ``splits.json``.

    Returns:
        Mapping book key -> split name for every given book.

    Raises:
        ValueError: If the stored file is not a mapping to known split names.
    """
    splits_path = data_directory / SPLITS_FILE
    stored_splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {}
    if not isinstance(stored_splits, dict) or any(split_name not in SPLIT_NAMES
                                                  for split_name in stored_splits.values()):
        raise ValueError(f"{splits_path} must map book keys to one of {SPLIT_NAMES}.")
    new_books = [catalog_book for catalog_book in catalog_books if catalog_book.book_key not in stored_splits]
    if new_books or not splits_path.exists():
        stored_splits.update(assign_splits(new_books))
        data_directory.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(stored_splits, ensure_ascii=False, indent=1, sort_keys=True),
                               encoding="utf-8")
    return {catalog_book.book_key: stored_splits[catalog_book.book_key] for catalog_book in catalog_books}
