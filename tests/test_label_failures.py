import gzip
import json

import scope.label as label


def test_failed_and_missing_features_are_logged_with_reasons(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.jsonl.gz"
    with gzip.open(catalog, "wt") as f:
        for i in range(30):
            freq = 1e-8 if i == 29 else 1e-4  # 29 is rarer than the default filter allows
            f.write(json.dumps({"latent": i, "freq": freq, "max_act": 2.0,
                                "examples": [{"act": 1.0, "left": "a", "token": "b", "right": "c"}] * 4}) + "\n")

    def fake_chat(prompt):
        ids = [int(line.split()[1]) for line in prompt.splitlines() if line.startswith("Feature ")]
        if 15 in ids:
            raise RuntimeError("Meta API 400: filtered")
        return json.dumps([{"latent": i, "label": f"f{i}", "kind": "concept", "steer": 1} for i in ids if i != 3])

    monkeypatch.setattr(label, "chat", fake_chat)
    out, fails = tmp_path / "labels.jsonl", tmp_path / "failures.jsonl"
    written = label.label_catalog(str(catalog), str(out), 100, workers=2, failures_path=str(fails))
    failures = [json.loads(line) for line in open(fails)]
    assert written == 18  # 29 candidates (29 filtered out), batch 10-19 failed, 3 left out
    assert {"latents": list(range(10, 20)), "reason": "Meta API 400: filtered"} in failures
    assert {"latents": [3], "reason": "Muse Spark's reply left these features out"} in failures

    out2 = tmp_path / "labels2.jsonl"
    everything = {"min_freq": 0.0, "max_freq": 1.0, "min_examples": 1}
    assert label.label_catalog(str(catalog), str(out2), 100, workers=2, filters=everything) == 19
