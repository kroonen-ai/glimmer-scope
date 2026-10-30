"""Glimmer as its own labeler and judge, for when no outside API is available.

Prompts are the same ones Muse Spark gets. Answers start in Glimmer's user channel (no
reasoning) and are generated greedily in batches capped by prompt tokens, halving the batch on
an out-of-memory error. Results carry "by": "glimmer" so they stay distinguishable from Muse
Spark's.
"""

import torch

from .steerability import answer_prefix, generate


class LocalLLM:
    def __init__(self, model, tokenizer, max_batch: int = 32, max_prompt_tokens: int = 96_000):
        self.model, self.tokenizer = model, tokenizer
        self.max_batch, self.max_prompt_tokens = max_batch, max_prompt_tokens

    def batches(self, prefixes: list[str]):
        lengths = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prefixes]
        start = 0
        while start < len(prefixes):
            stop = start + 1
            # left padding makes every row as long as the longest one
            while stop < len(prefixes) and stop - start < self.max_batch and max(lengths[start : stop + 1]) * (stop + 1 - start) <= self.max_prompt_tokens:
                stop += 1
            yield start, stop
            start = stop

    def complete(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        prefixes = [answer_prefix(self.tokenizer, p) for p in prompts]
        out = [""] * len(prompts)
        for start, stop in self.batches(prefixes):
            out[start:stop] = self._generate(prefixes[start:stop], max_new_tokens)
        return out

    def _generate(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        try:
            return generate(self.model, self.tokenizer, prefixes, 0, None, max_new_tokens)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prefixes) == 1:
                raise
            half = len(prefixes) // 2
            return self._generate(prefixes[:half], max_new_tokens) + self._generate(prefixes[half:], max_new_tokens)
