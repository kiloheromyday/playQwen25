from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
from typing import Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def clean_document(text: str) -> str:
    """Apply small, deterministic cleaning rules suitable for a teaching corpus."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def document_fingerprint(text: str) -> str:
    return hashlib.sha256(clean_document(text).encode("utf-8")).hexdigest()


def deduplicate_documents(documents: Iterable[str]) -> list[str]:
    """Keep the first occurrence after cleaning, preserving source order."""
    unique: list[str] = []
    seen: set[str] = set()
    for document in documents:
        cleaned = clean_document(document)
        if not cleaned:
            continue
        fingerprint = document_fingerprint(cleaned)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(cleaned)
    return unique


def load_text_documents(path: str | Path, suffixes: Sequence[str] = (".txt", ".md")) -> list[str]:
    """Load text files from a file or recursively from a directory."""
    source = Path(path)
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        allowed = {suffix.lower() for suffix in suffixes}
        candidates = sorted(
            item for item in source.rglob("*") if item.is_file() and item.suffix.lower() in allowed
        )
    else:
        raise FileNotFoundError(f"data path does not exist: {source}")
    documents = [item.read_text(encoding="utf-8") for item in candidates]
    return deduplicate_documents(documents)


def split_documents(
    documents: Sequence[str],
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split complete documents before tokenization to prevent leakage across sets."""
    if not documents:
        raise ValueError("documents must contain at least one non-empty document")
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1)")
    items = list(documents)
    random.Random(seed).shuffle(items)
    if len(items) == 1 or valid_ratio == 0.0:
        return items, []
    valid_count = max(1, round(len(items) * valid_ratio))
    valid_count = min(valid_count, len(items) - 1)
    return items[valid_count:], items[:valid_count]


def pack_token_ids(
    token_ids: list[int],
    sequence_length: int,
    eos_token_id: int | None = None,
) -> list[list[int]]:
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than 1")
    stream = list(token_ids)
    if eos_token_id is not None and (not stream or stream[-1] != eos_token_id):
        stream.append(eos_token_id)
    usable = len(stream) // sequence_length * sequence_length
    return [stream[start : start + sequence_length] for start in range(0, usable, sequence_length)]


def tokenize_and_pack(
    documents: Sequence[str],
    tokenizer,
    sequence_length: int,
) -> list[list[int]]:
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(clean_document(document), add_eos=True))
    return pack_token_ids(token_ids, sequence_length, eos_token_id=None)


class PackedTokenDataset(Dataset):
    def __init__(self, sequences: list[list[int]]):
        if not sequences:
            raise ValueError("dataset must contain at least one sequence")
        self.tokens = torch.tensor(sequences, dtype=torch.long)

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> Tensor:
        return self.tokens[index]


@dataclass
class DataBundle:
    train_documents: list[str]
    valid_documents: list[str]
    train_dataset: PackedTokenDataset
    valid_dataset: PackedTokenDataset | None


def build_data_bundle(
    documents: Sequence[str],
    tokenizer,
    sequence_length: int,
    valid_ratio: float = 0.1,
    seed: int = 0,
) -> DataBundle:
    train_documents, valid_documents = split_documents(documents, valid_ratio, seed)
    train_sequences = tokenize_and_pack(train_documents, tokenizer, sequence_length)
    if not train_sequences:
        raise ValueError("training documents are shorter than one complete sequence")
    valid_sequences = tokenize_and_pack(valid_documents, tokenizer, sequence_length)
    return DataBundle(
        train_documents=train_documents,
        valid_documents=valid_documents,
        train_dataset=PackedTokenDataset(train_sequences),
        valid_dataset=PackedTokenDataset(valid_sequences) if valid_sequences else None,
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


@dataclass
class Batch:
    input_ids: Tensor
    labels: Tensor


def make_next_token_batch(sequences: Tensor) -> Batch:
    if sequences.ndim != 2 or sequences.size(1) < 2:
        raise ValueError("sequences must have shape [batch, seq_len >= 2]")
    return Batch(input_ids=sequences, labels=sequences.clone())
