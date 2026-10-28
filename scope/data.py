"""Token batches for activation collection.

A mix of web text (FineWeb-Edu) and chat conversations (UltraChat) rendered
with Glimmer's own chat template, so assistant-side features (reasoning
channel, refusals, persona, formatting) show up next to world knowledge.
Documents are packed back to back into fixed-length rows.
"""

import random
from collections.abc import Iterator

import torch

from .config import OUTPUT_DIR

DATA_DIR = f"{OUTPUT_DIR}/data"
# Local copies (see modal_app.download_data): one shard each for training and
# a held-out shard each for the feature catalog.
SHARDS = {
    "web": {
        "repo": "HuggingFaceFW/fineweb-edu",
        "train": ["sample/10BT/000_00000.parquet"],
        "catalog": ["sample/10BT/001_00000.parquet"],
    },
    "chat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "train": ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet"],
        "catalog": ["data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"],
    },
}


def _rows(kind: str, split: str) -> Iterator[dict]:
    from datasets import load_dataset

    files = [f"{DATA_DIR}/{kind}/{path.rsplit('/', 1)[-1]}" for path in SHARDS[kind][split]]
    yield from load_dataset("parquet", data_files=files, split="train", streaming=True)


def _web_texts(split: str) -> Iterator[str]:
    for row in _rows("web", split):
        yield row["text"]


def _chat_texts(tokenizer, split: str) -> Iterator[str]:
    for row in _rows("chat", split):
        try:
            yield tokenizer.apply_chat_template(row["messages"], tokenize=False)
        except Exception:
            continue


def documents(tokenizer, chat_fraction: float, seed: int, split: str = "train") -> Iterator[str]:
    rng = random.Random(seed)
    web, chat = _web_texts(split), _chat_texts(tokenizer, split)
    while True:
        source = chat if rng.random() < chat_fraction else web
        yield next(source)


def token_batches(
    tokenizer, seq_len: int, batch_size: int, chat_fraction: float, seed: int, split: str = "train"
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (input_ids, attention_mask) of shape [batch_size, seq_len]."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    row_len = seq_len - (1 if bos is not None else 0)
    pending: list[int] = []
    rows: list[list[int]] = []
    for text in documents(tokenizer, chat_fraction, seed, split):
        pending.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos is not None:
            pending.append(eos)
        while len(pending) >= row_len:
            chunk, pending = pending[:row_len], pending[row_len:]
            rows.append(([bos] if bos is not None else []) + chunk)
            if len(rows) == batch_size:
                ids = torch.tensor(rows, dtype=torch.long)
                yield ids, torch.ones_like(ids)
                rows = []
