import gzip
import json

import torch

import scope.local_llm as local_llm
from scope.label import label_catalog_local
from scope.local_llm import LocalLLM


class Tok:
    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": text.split()})()

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"]


def test_batches_respect_row_and_token_caps():
    llm = LocalLLM(None, Tok(), max_batch=3, max_prompt_tokens=12)
    prefixes = ["a b", "a b c d", "a", "a b c d e f", "a"]
    spans = list(llm.batches(prefixes))
    assert spans[0][0] == 0 and spans[-1][1] == len(prefixes)
    assert all(b == c for (_, b), (c, _) in zip(spans, spans[1:]))
    for start, stop in spans:
        longest = max(len(p.split()) for p in prefixes[start:stop])
        assert stop - start <= 3 and (stop - start == 1 or longest * (stop - start) <= 12)


def test_out_of_memory_halves_the_batch(monkeypatch):
    calls = []

    def fake_generate(model, tok, prefixes, layer, vectors, n):
        calls.append(len(prefixes))
        if len(prefixes) > 2:
            raise torch.OutOfMemoryError("too big")
        return [p.upper() for p in prefixes]

    monkeypatch.setattr(local_llm, "generate", fake_generate)
    monkeypatch.setattr(local_llm, "answer_prefix", lambda tok, p: p)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    llm = LocalLLM(None, Tok(), max_batch=8, max_prompt_tokens=1000)
    assert llm.complete(["a", "b", "c", "d", "e"], 5) == ["A", "B", "C", "D", "E"]
    assert calls[0] == 5


def test_local_labels_are_tagged_and_bad_replies_skipped(tmp_path):
    catalog = tmp_path / "catalog.jsonl.gz"
    rows = [{"latent": i, "freq": 1e-4, "max_act": 3.0,
             "examples": [{"act": 1.0, "left": "x", "token": "t", "right": "y"}] * 4} for i in range(25)]
    with gzip.open(catalog, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    class Fake:
        max_batch = 2

        def complete(self, prompts, max_new_tokens):
            out = []
            for prompt in prompts:
                ids = [int(line.split()[1]) for line in prompt.splitlines() if line.startswith("Feature ")]
                out.append("not json" if 20 in ids else json.dumps([{"latent": i, "label": f"f{i}", "kind": "concept", "steer": 2} for i in ids]))
            return out

    out = tmp_path / "labels.jsonl"
    written = label_catalog_local(Fake(), str(catalog), str(out), limit=100, done={3})
    labels = [json.loads(line) for line in open(out)]
    assert written == len(labels) == 14  # 24 candidates, the batch holding 20 failed to parse
    assert all(r["by"] == "glimmer" for r in labels) and 3 not in {r["latent"] for r in labels}
