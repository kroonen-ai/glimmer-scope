"""Token batches for activation collection.

A mix of web text (FineWeb-Edu) and chat conversations (UltraChat) rendered
with Glimmer's own chat template, so assistant-side features (reasoning
channel, refusals, persona, formatting) show up next to world knowledge.
Documents are packed back to back into fixed-length rows.
"""

import random
from collections.abc import Iterator

import torch

WEB = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")
CHAT = ("HuggingFaceH4/ultrachat_200k", None, "train_sft")


def _web_texts(skip: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(WEB[0], name=WEB[1], split=WEB[2], streaming=True).skip(skip)
    for row in ds:
        yield row["text"]


def _chat_texts(tokenizer, skip: int) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset(CHAT[0], split=CHAT[2], streaming=True).skip(skip)
    for row in ds:
        try:
            yield tokenizer.apply_chat_template(row["messages"], tokenize=False)
        except Exception:
            continue


def documents(tokenizer, chat_fraction: float, seed: int, skip: int = 0) -> Iterator[str]:
    rng = random.Random(seed)
    web, chat = _web_texts(skip), _chat_texts(tokenizer, skip)
    while True:
        source = chat if rng.random() < chat_fraction else web
        yield next(source)


def token_batches(
    tokenizer, seq_len: int, batch_size: int, chat_fraction: float, seed: int, skip: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (input_ids, attention_mask) of shape [batch_size, seq_len]."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    row_len = seq_len - (1 if bos is not None else 0)
    pending: list[int] = []
    rows: list[list[int]] = []
    for text in documents(tokenizer, chat_fraction, seed, skip):
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
