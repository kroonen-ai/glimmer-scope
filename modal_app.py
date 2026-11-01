"""Glimmer Scope on Modal.

    modal run modal_app.py::download            # once: weights into the volume (CPU)
    modal run modal_app.py::download_data       # once: training and held-out text shards
    modal run modal_app.py::smoke               # load, measure, check hooks (~10 min GPU)
    modal run --detach modal_app.py::train      # train an SAE (hours)
    modal run modal_app.py::catalog --run layer24-64k-k64
    modal run modal_app.py::steer --run layer24-64k-k64 --latent 123 --prompt "Describe your ideal weekend."

GPU type comes from GLIMMER_GPU (default H200: 141 GB fits the 60 GB model,
the SAE with Adam state, and a 13 GB activation buffer on one card).
"""

import os

import modal

from scope.config import MODEL_ID, OUTPUT_DIR, WEIGHTS_DIR

GPU = os.environ.get("GLIMMER_GPU", "H200")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8",
        "transformers==5.21.0",
        "accelerate",
        "datasets",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "numpy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("scope")
)

app = modal.App("glimmer-scope", image=image)
weights = modal.Volume.from_name("glimmer-weights", create_if_missing=True)
scope_volume = modal.Volume.from_name("glimmer-scope", create_if_missing=True)
VOLUMES = {"/weights": weights, OUTPUT_DIR: scope_volume}


def run_name(layer: int, n_latents: int, k: int) -> str:
    return f"layer{layer}-{n_latents // 1024}k-k{k}"


@app.function(volumes=VOLUMES, cpu=8, memory=16384, timeout=3 * 3600)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        local_dir=WEIGHTS_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"],
    )
    weights.commit()
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(WEIGHTS_DIR) for f in fs)
    print(f"weights in volume: {total / 1e9:.1f} GB")


@app.function(volumes=VOLUMES, cpu=4, timeout=3600)
def download_data():
    from huggingface_hub import hf_hub_download

    from scope.data import DATA_DIR, SHARDS

    for kind, spec in SHARDS.items():
        for split in ("train", "catalog"):
            for path in spec[split]:
                local = hf_hub_download(spec["repo"], path, repo_type="dataset", local_dir=f"{DATA_DIR}/{kind}/_hf")
                target = f"{DATA_DIR}/{kind}/{path.rsplit('/', 1)[-1]}"
                os.replace(local, target)
                print(f"{target}: {os.path.getsize(target) / 1e9:.2f} GB")
    scope_volume.commit()


@app.function(volumes=VOLUMES, gpu=GPU, timeout=3600)
def smoke(layer: int = 24):
    import time

    import torch

    from scope import data, model as glimmer

    t = time.time()
    model, tokenizer = glimmer.load()
    print(f"loaded in {time.time() - t:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU")
    print("bos", tokenizer.bos_token_id, "eos", tokenizer.eos_token_id)

    batches = data.token_batches(tokenizer, 1024, 16, 0.3, seed=0)
    ids, mask = next(batches)
    ids, mask = ids.cuda(), mask.cuda()
    for probe in (12, layer, 36):
        acts = glimmer.residual(model, ids, mask, probe)
        norms = acts.float().norm(dim=-1)
        keep = glimmer.keep_mask(acts, mask)
        kept = norms[keep]
        print(
            f"layer {probe}: pos0 norm {norms[:, 0].mean():.1f} | median {kept.median():.1f} "
            f"| p99 {kept.quantile(0.99):.1f} | max {norms.max():.1f} | kept {keep.float().mean():.3f}"
        )

    for _ in range(2):
        ids, mask = next(batches)
        ids, mask = ids.cuda(), mask.cuda()
        torch.cuda.synchronize()
        t = time.time()
        glimmer.residual(model, ids, mask, layer)
        torch.cuda.synchronize()
        print(f"layer {layer} forward: {ids.numel() / (time.time() - t):.0f} tokens/s")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Describe your ideal weekend in three sentences."}],
        add_generation_prompt=True,
        tokenize=False,
        reasoning_strength="low",
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    median = glimmer.residual(model, inputs.input_ids, inputs.attention_mask, layer)[0, 1:].float().norm(dim=-1).median()
    direction = torch.randn(model.config.text_config.hidden_size, device="cuda")
    direction /= direction.norm()

    def generate(label, vector=None):
        context = glimmer.steering(model, layer, vector) if vector is not None else None
        if context:
            context.__enter__()
        try:
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        finally:
            if context:
                context.__exit__(None, None, None)
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
        print(f"--- {label}\n{text}\n")

    generate("baseline")
    generate("random direction x0.5 median norm", direction * 0.5 * median)
    generate("random direction x2 median norm", direction * 2 * median)
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


@app.function(volumes=VOLUMES, gpu=GPU, timeout=24 * 3600)
def train(layer: int = 24, n_latents: int = 65536, k: int = 64, total_tokens: int = 100_000_000, tag: str = "",
          buffer_tokens: int = 1 << 19):
    """``buffer_tokens`` is the GPU shuffle buffer: 512k tokens fits a 96 GB card (1M ran out of memory on refill)."""
    from scope import model as glimmer
    from scope.config import SAEConfig, TrainConfig
    from scope.train import train as run_training

    model, tokenizer = glimmer.load()
    sae_cfg = SAEConfig(layer=layer, n_latents=n_latents, k=k)
    cfg = TrainConfig(total_tokens=total_tokens, buffer_tokens=buffer_tokens)
    out = f"{OUTPUT_DIR}/{run_name(layer, n_latents, k)}{'-' + tag if tag else ''}"
    final = run_training(model, tokenizer, sae_cfg, cfg, out, commit=scope_volume.commit)
    print(f"saved {final}")


@app.function(volumes=VOLUMES, gpu=GPU, timeout=6 * 3600)
def catalog(run: str, n_tokens: int = 5_000_000, checkpoint: str = "final", traces: bool = False):
    """``traces`` also keeps every firing in {run}/traces/ (scope.interpret.build_catalog). The same
    seed gives the same tokens, so a rerun writes its catalog beside an existing one, never over it."""
    from scope import model as glimmer
    from scope.interpret import build_catalog
    from scope.sae import TopKSAE

    base = f"{OUTPUT_DIR}/{run}"
    out = f"{base}/catalog.jsonl.gz"
    if traces and os.path.exists(out):
        out = f"{base}/catalog-traces.jsonl.gz"
    model, tokenizer = glimmer.load()
    sae = TopKSAE.load(f"{base}/{checkpoint}", device="cuda")
    stats = build_catalog(model, tokenizer, sae, n_tokens, out, traces_dir=f"{base}/traces" if traces else None)
    scope_volume.commit()
    print(stats)


@app.function(volumes=VOLUMES, gpu=GPU, timeout=3600)
def steer(run: str, latent: int, prompt: str = "Describe your ideal weekend.", strengths: str = "0,2,4,8", checkpoint: str = "final"):
    """Generate with one latent added at several multiples of its typical peak activation."""
    import gzip
    import json

    from scope import model as glimmer
    from scope.sae import TopKSAE

    model, tokenizer = glimmer.load()
    sae = TopKSAE.load(f"{OUTPUT_DIR}/{run}/{checkpoint}", device="cuda")
    peak = 1.0
    catalog_path = f"{OUTPUT_DIR}/{run}/catalog.jsonl.gz"
    if os.path.exists(catalog_path):
        with gzip.open(catalog_path, "rt") as f:
            for line in f:
                row = json.loads(line)
                if row["latent"] == latent:
                    peak = row["max_act"]
                    break
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, reasoning_strength="low"
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    for s in (float(x) for x in strengths.split(",")):
        vector = sae.steering_vector(latent, s * peak).to("cuda")
        with glimmer.steering(model, sae.cfg.layer, vector):
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(f"--- latent {latent} x{s} (peak {peak:.2f})\n{tokenizer.decode(out[0, inputs.input_ids.shape[1]:])}\n")


# GLIMMER_META_SECRET picks which Modal secret holds META_API_KEY (default "meta")
meta_secret = modal.Secret.from_name(os.environ.get("GLIMMER_META_SECRET", "meta"))


@app.function(volumes=VOLUMES, secrets=[meta_secret], cpu=2, timeout=8 * 3600)
def label(run: str, limit: int = 20000, workers: int = 16, out_name: str = "labels.jsonl", everything: bool = False,
          skip_screened: bool = False):
    """Muse Spark names the catalog's features; resumes where it stopped.

    ``out_name`` writes to another file (merged into labels.jsonl later), so a second run
    never shares a file with one still going; features named in either file are skipped.
    ``everything`` drops the firing-rate filter: every feature that fired at all gets a try.
    ``skip_screened`` leaves out features whose examples the judge's screen would stop, so nothing
    sensitive goes to Spark (they are labeled locally, scripts/label_split.py).
    Features a request could not name are logged with the reason in label-failures.jsonl.
    """
    import json

    from scope.label import label_catalog

    base = f"{OUTPUT_DIR}/{run}"
    out = f"{base}/{out_name}"
    done = set()
    for path in {f"{base}/labels.jsonl", out}:
        if os.path.exists(path):
            done |= {json.loads(line)["latent"] for line in open(path)}
    if skip_screened:
        import gzip

        from scope.label import describe
        from scope.safety import screen

        with gzip.open(f"{base}/catalog.jsonl.gz", "rt") as f:
            screened = {row["latent"] for row in map(json.loads, f) if screen(describe(row))}
        print(f"{len(screened - done)} screened features left for local labeling", flush=True)
        done |= screened
    filters = {"min_freq": 0.0, "max_freq": 1.0, "min_examples": 1} if everything else None
    written = label_catalog(f"{base}/catalog.jsonl.gz", out, limit, workers=workers, done=done,
                            failures_path=f"{base}/label-failures.jsonl", filters=filters)
    scope_volume.commit()
    print(f"labeled {written} now, {len(done) + written} in total")


@app.function(volumes=VOLUMES, secrets=[meta_secret], gpu=GPU, timeout=12 * 3600)
def sweep(
    run: str,
    n_features: int = 1000,
    min_steer: int = 2,
    kinds: str = "concept,unclear",
    strengths: str = "0.2,0.35,0.5,0.7",
    checkpoint: str = "final",
    candidates: str = "labels",
    prompt_set: str = "everyday",
    rows_per_batch: int = 128,
    judge_workers: int = 32,
):
    """Steer with the most promising labeled features and have Muse Spark score them.

    Generation is memory-bound, so 128 rows a batch costs little more per step than 32 and
    roughly doubles features per GPU hour; greedy decoding keeps the texts the same.

    ``candidates="catalog"`` sweeps every feature that fired, whatever its kind and labeled or not.
    ``candidates="personas"`` sweeps the features ``persona_scan`` flagged instead, judged on
    ``prompt_set`` prompts from scope.archetype.EVAL_SETS (``social`` suits them) into
    steerability-personas.jsonl.
    """
    import gzip
    import json
    import random

    from scope import model as glimmer
    from scope.archetype import EVAL_SETS
    from scope.sae import TopKSAE
    from scope.steerability import contexts_for, sweep as run_sweep

    base = f"{OUTPUT_DIR}/{run}"
    # the catalog mode writes where the labels mode does: one file of scores per layer
    out = f"{base}/steerability.jsonl" if candidates in ("labels", "catalog") else f"{base}/steerability-{candidates}.jsonl"
    scored = set()
    if os.path.exists(out):
        scored = {json.loads(line)["latent"] for line in open(out)}
    # features whose replies are already kept without a score are never steered again
    for suffix in ("-screened.jsonl", "-unjudged.jsonl", "-filtered.jsonl"):
        kept = out.replace(".jsonl", suffix)
        if os.path.exists(kept):
            for line in open(kept):
                row = json.loads(line)
                if "steered" in row:
                    scored.add(row["latent"])
    if candidates == "catalog":
        # every feature that fired, labeled or not; one without a label keeps its replies for later
        labels = {}
        if os.path.exists(f"{base}/labels.jsonl"):
            labels = {r["latent"]: r for r in map(json.loads, open(f"{base}/labels.jsonl"))}
        features = []
        with gzip.open(f"{base}/catalog.jsonl.gz", "rt") as f:
            for line in f:
                latent = json.loads(line)["latent"]
                if latent not in scored:
                    r = labels.get(latent, {})
                    features.append({"latent": latent, "label": r.get("label", ""), "kind": r.get("kind", "unlabeled"),
                                     "steer": r.get("steer", -1)})
    elif candidates == "labels":
        wanted = set(kinds.split(","))
        labels = [json.loads(line) for line in open(f"{base}/labels.jsonl")]
        features = [
            f for f in labels if f["kind"] in wanted and f["steer"] >= min_steer and f["latent"] not in scored
        ]
    else:
        rows = [json.loads(line) for line in open(f"{base}/{candidates}.jsonl")]
        features = [
            {**r, "label": f"{r['label']} (could act as: {r['persona']})" if r.get("persona") else r["label"]}
            for r in rows
            if r["latent"] not in scored
        ]
    # A seeded shuffle, then by guessed steerability, so the sample spans every firing rate.
    random.Random(0).shuffle(features)
    features.sort(key=lambda f: -f["steer"])
    features = features[:n_features]
    picked = {f["latent"] for f in features}
    contexts = {}
    with gzip.open(f"{base}/catalog.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["latent"] in picked:
                contexts[row["latent"]] = contexts_for(row)
    print(f"sweeping {len(features)} features", flush=True)
    model, tokenizer = glimmer.load()
    sae = TopKSAE.load(f"{base}/{checkpoint}", device="cuda")
    written = run_sweep(
        model, tokenizer, sae, features, contexts, out, strengths=tuple(float(s) for s in strengths.split(",")),
        prompts=EVAL_SETS[prompt_set], rows_per_batch=rows_per_batch, judge_workers=judge_workers,
    )
    scope_volume.commit()
    print(f"scored {written}")


@app.function(volumes=VOLUMES)
def top(run: str, n: int = 40, min_fluency: int = 2, file: str = "steerability.jsonl"):
    """The best-steering features found so far."""
    import json

    rows = [json.loads(line) for line in open(f"{OUTPUT_DIR}/{run}/{file}")]
    rows.sort(key=lambda r: -r["score"])
    print(f"{len(rows)} features scored; top {n}:")
    for r in rows[:n]:
        sample = r["samples"].get(str(r["best_strength"]), [""])[0] if r["best_strength"] is not None else ""
        print(f"{r['latent']:>6}  {r['score']:.2f}  x{r['best_strength']}  {r['label']}  ->  {r['observed']}")
        print(f"        {' '.join(sample.split())[:160]}")


@app.function(volumes=VOLUMES, secrets=[meta_secret], gpu=GPU, timeout=4 * 3600, max_containers=5)
def archetype(
    run: str,
    name: str,
    n_pairs: int = 24,
    top: int = 10,
    strengths: str = "0.2,0.35,0.5,0.7",
    checkpoint: str = "final",
):
    """Rank the features behind an archetype and test steering with them, their blend, and a mean-difference vector."""
    import gzip
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import save_file

    from scope import model as glimmer
    from scope.archetype import (
        ARCHETYPES,
        EVAL_SETS,
        PROMPT_SETS,
        answer_activations,
        blend_direction,
        detector_accuracy,
        everyday_pairs,
        glimmer_pairs,
        mean_difference_direction,
        rank_features,
        slug,
        spark_pairs,
        split_pairs,
    )
    from scope.sae import TopKSAE
    from scope.steerability import contexts_for, sweep as run_sweep

    scope_volume.reload()
    if name not in ARCHETYPES:
        raise SystemExit(f"unknown archetype {name!r}; add it to scope/archetype.py")
    spec = ARCHETYPES[name]
    base = Path(f"{OUTPUT_DIR}/{run}")
    out = base / "archetypes" / slug(name)
    out.mkdir(parents=True, exist_ok=True)

    model = tokenizer = None
    pairs_path = out / "pairs.json"
    if pairs_path.exists():
        pairs = json.loads(pairs_path.read_text())
    elif spec["prompts"] == "everyday":
        pairs = everyday_pairs(name, spec, n_pairs)
    else:
        prompts = PROMPT_SETS[spec["prompts"]][:n_pairs]
        pairs = spark_pairs(name, spec, prompts)
        missing = [p for p in prompts if p not in {q["prompt"] for q in pairs}]
        if missing:
            print(f"{name}: Muse Spark wrote {len(pairs)} pairs, Glimmer writes the other {len(missing)}", flush=True)
            model, tokenizer = glimmer.load()
            pairs += glimmer_pairs(model, tokenizer, 24, name, spec, missing)
    if not pairs_path.exists():
        pairs_path.write_text(json.dumps(pairs, indent=2))
        scope_volume.commit()
    sources = {s: sum(p.get("source") == s for p in pairs) for s in ("muse-spark", "glimmer")}
    print(f"{len(pairs)} pairs for {name} {sources}", flush=True)

    if model is None:
        model, tokenizer = glimmer.load()
    sae = TopKSAE.load(base / checkpoint, device="cuda")
    train_pairs, test_pairs = split_pairs(pairs)

    def activations(rows):
        feats_a, feats_n, resid_a, resid_n = [], [], [], []
        for pair in rows:
            fa, ra = answer_activations(model, tokenizer, sae, pair["prompt"], pair["archetype"])
            fn, rn = answer_activations(model, tokenizer, sae, pair["prompt"], pair["neutral"])
            feats_a.append(fa), feats_n.append(fn), resid_a.append(ra), resid_n.append(rn)
        stack = lambda xs: torch.stack(xs) if xs else torch.empty(0)  # noqa: E731
        return stack(feats_a), stack(feats_n), stack(resid_a), stack(resid_n)

    fa, fn, ra, rn = activations(train_pairs)
    ranked = rank_features(fa, fn, top)
    blend = blend_direction(sae, ranked).detach()
    mean_difference = mean_difference_direction(ra, rn)

    # the held-out third: does the top feature, or the mean-difference direction, tell the sides apart?
    detector = {"held_out_pairs": len(test_pairs)}
    if test_pairs:
        ta, tn, tra, trn = activations(test_pairs)
        detector["top_feature"] = detector_accuracy(ta[:, ranked[0]["latent"]], tn[:, ranked[0]["latent"]]) if ranked else None
        detector["mean_difference"] = detector_accuracy(tra @ mean_difference, trn @ mean_difference)

    save_file(
        {"blend": blend.float().cpu().contiguous(), "mean_difference": mean_difference.float().cpu().contiguous()},
        str(out / f"directions-{checkpoint}.safetensors"),
        metadata={"layer": str(sae.cfg.layer)},
    )

    labels = {}
    if (base / "labels.jsonl").exists():
        labels = {r["latent"]: r["label"] for r in map(json.loads, open(base / "labels.jsonl"))}
    contexts = {}
    wanted = {r["latent"] for r in ranked}
    if (base / "catalog.jsonl.gz").exists():
        with gzip.open(base / "catalog.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                if row["latent"] in wanted:
                    contexts[row["latent"]] = contexts_for(row)
    for r in ranked:
        r["feature_label"] = labels.get(r["latent"], "")

    target = f"the archetype '{name}': {spec['description']}"
    candidates = [{"latent": r["latent"], "label": target} for r in ranked]
    candidates.append({"name": "blend", "label": target, "direction": blend})
    candidates.append({"name": "mean-difference", "label": target, "direction": mean_difference})
    results_path = out / f"steerability-{checkpoint}.jsonl"
    results_path.unlink(missing_ok=True)
    run_sweep(
        model, tokenizer, sae, candidates, contexts, str(results_path),
        strengths=tuple(float(s) for s in strengths.split(",")),
        prompts=EVAL_SETS[spec["prompts"]], max_new_tokens=100,
    )
    results = {(r["name"] or r["latent"]): r for r in map(json.loads, open(results_path))}
    for r in ranked:
        r["steer"] = {k: results.get(r["latent"], {}).get(k) for k in ("score", "best_strength", "observed")}
    keys = ("score", "best_strength", "observed", "samples")
    summary = {
        "name": name,
        "group": spec["group"],
        "description": spec["description"],
        "checkpoint": checkpoint,
        "pairs": len(pairs),
        "sources": sources,
        "detector": detector,
        "features": ranked,
        "blend": {k: results.get("blend", {}).get(k) for k in keys},
        "mean_difference": {k: results.get("mean-difference", {}).get(k) for k in keys},
    }
    (out / f"ranking-{checkpoint}.json").write_text(json.dumps(summary, indent=2))
    scope_volume.commit()

    lines = [f"\n== {name} ({spec['group']}, {checkpoint}) pairs {sources} detector {detector}"]
    for r in ranked:
        s = r["steer"]
        lines.append(
            f"{r['latent']:>6}  diff {r['mean_diff']:.3f}  wins {r['consistency']:.0%}  steer {s['score']}  "
            f"x{s['best_strength']}  {r['feature_label'][:40]}  ->  {s['observed']}"
        )
    for key in ("blend", "mean_difference"):
        s = summary[key]
        lines.append(f"{key:>15}: steer {s['score']} x{s['best_strength']} -> {s['observed']}")
        sample = (s.get("samples") or {}).get(str(s["best_strength"]), [""])[0] if s["best_strength"] else ""
        if sample:
            lines.append(f"                 {' '.join(sample.split())[:220]}")
    report = "\n".join(lines)
    print(report, flush=True)
    return report


@app.local_entrypoint()
def archetypes(run: str = "layer24-64k-k64", names: str = "all", skip_done: bool = True):
    """Run many archetypes from one app (five GPUs at a time), and print each summary as it lands."""
    from scope.personas import ARCHETYPES, slug

    chosen = list(ARCHETYPES) if names == "all" else [n.strip() for n in names.split(",")]
    if skip_done:
        done = set()
        for entry in scope_volume.listdir(f"{run}/archetypes", recursive=True):
            if entry.path.endswith("/directions-final.safetensors"):
                done.add(entry.path.split("/")[-2])
        chosen = [n for n in chosen if slug(n) not in done]
    print(f"running {len(chosen)} archetypes: {', '.join(chosen)}", flush=True)
    calls = {n: archetype.spawn(run, n) for n in chosen}
    for n, call in calls.items():
        try:
            print(call.get(), flush=True)
        except Exception as error:
            print(f"\n== {n} FAILED: {error}", flush=True)


@app.function(volumes=VOLUMES, secrets=[meta_secret], cpu=2, timeout=4 * 3600)
def persona_scan(run: str, batch: int = 80, workers: int = 16):
    """Muse Spark flags the labeled features that could give Glimmer a persona."""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from scope.archetype import scan_batch

    base = f"{OUTPUT_DIR}/{run}"
    rows = [json.loads(line) for line in open(f"{base}/labels.jsonl")]
    rows = [r for r in rows if r["kind"] in ("concept", "unclear") and r["label"]]
    batches = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    picked = []
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(scan_batch, b) for b in batches]):
            try:
                picked += future.result()
            except Exception as error:
                print(f"scan batch failed: {error}", flush=True)
    with open(f"{base}/personas.jsonl", "w") as out:
        for row in sorted(picked, key=lambda r: r["latent"]):
            out.write(json.dumps(row) + "\n")
    scope_volume.commit()
    print(f"{len(picked)} persona candidates from {len(rows)} labeled features")


@app.function(volumes=VOLUMES, gpu=GPU, timeout=2 * 3600)
def rank_pairs(run: str, pairs: list[dict], top: int = 40, checkpoint: str = "final") -> dict:
    """Rank features for caller-supplied {prompt, archetype, neutral} pairs and return the directions.

    Nothing is written to the volume: the pairs arrive as arguments and the ranking, the held-out
    detector check, and the blend and mean-difference directions come back as the return value.
    """
    import json
    from pathlib import Path

    import torch

    from scope import model as glimmer
    from scope.archetype import (
        answer_activations,
        blend_direction,
        detector_accuracy,
        mean_difference_direction,
        rank_features,
        split_pairs,
    )
    from scope.sae import TopKSAE

    base = Path(f"{OUTPUT_DIR}/{run}")
    model, tokenizer = glimmer.load()
    sae = TopKSAE.load(base / checkpoint, device="cuda")
    pairs = [p for p in pairs if p.get("prompt") and p.get("archetype") and p.get("neutral")]
    train_pairs, test_pairs = split_pairs(pairs)

    def activations(rows):
        out = {"fa": [], "fn": [], "ra": [], "rn": []}
        with torch.no_grad():
            for pair in rows:
                fa, ra = answer_activations(model, tokenizer, sae, pair["prompt"], pair["archetype"])
                fn, rn = answer_activations(model, tokenizer, sae, pair["prompt"], pair["neutral"])
                out["fa"].append(fa), out["fn"].append(fn), out["ra"].append(ra), out["rn"].append(rn)
        return [torch.stack(out[k]) for k in ("fa", "fn", "ra", "rn")]

    fa, fn, ra, rn = activations(train_pairs)
    ranked = rank_features(fa, fn, top)
    blend = blend_direction(sae, ranked).detach().float().cpu()
    mean_difference = mean_difference_direction(ra, rn).detach().float().cpu()
    detector = {"train_pairs": len(train_pairs), "held_out_pairs": len(test_pairs)}
    if test_pairs:
        ta, tn, tra, trn = activations(test_pairs)
        detector["top_feature"] = detector_accuracy(ta[:, ranked[0]["latent"]], tn[:, ranked[0]["latent"]])
        detector["mean_difference"] = detector_accuracy(tra.cpu() @ mean_difference, trn.cpu() @ mean_difference)
    labels = {}
    if (base / "labels.jsonl").exists():
        labels = {r["latent"]: r["label"] for r in map(json.loads, open(base / "labels.jsonl"))}
    for r in ranked:
        r["feature_label"] = labels.get(r["latent"], "")
    return {
        "layer": sae.cfg.layer,
        "detector": detector,
        "features": ranked,
        "blend": blend.tolist(),
        "mean_difference": mean_difference.tolist(),
    }


@app.function(volumes=VOLUMES, gpu=GPU, timeout=2 * 3600)
def crosslayer(runs: str, n_tokens: int = 5_000_000, top: int = 8, checkpoint: str = "final"):
    """Match features between neighboring layers' SAEs by what they fire on (scope.crosslayer)."""
    import numpy as np

    from scope import model as glimmer
    from scope.crosslayer import cofire
    from scope.sae import TopKSAE

    import torch

    model, tokenizer = glimmer.load()
    # One neighboring pair at a time: a pair's 64k x 64k sums are 17 GB, and beside Glimmer's ~60 GB
    # a 96 GB card holds one pair, not three (2026-11-01). The pairs land in one file, as before.
    names = runs.split(",")
    result = {}
    for lo, hi in zip(names, names[1:]):
        saes = [TopKSAE.load(f"{OUTPUT_DIR}/{run}/{checkpoint}", device="cuda") for run in (lo, hi)]
        pair = cofire(model, tokenizer, saes, n_tokens, top=top)
        del saes
        torch.cuda.empty_cache()
        layers = sorted(set(result.get("layers", [])) | set(pair["layers"]))
        tokens = min(result.get("tokens", pair["tokens"]), pair["tokens"])
        result.update({key: value for key, value in pair.items() if key not in ("layers", "tokens")})
        result["layers"], result["tokens"] = layers, tokens
        print(f"pair {lo} / {hi}: done", flush=True)
    os.makedirs(f"{OUTPUT_DIR}/crosslayer", exist_ok=True)
    path = f"{OUTPUT_DIR}/crosslayer/{'-'.join(f'L{layer}' for layer in result['layers'])}.npz"
    np.savez_compressed(path, **result)
    scope_volume.commit()
    for key, value in result.items():
        if key.startswith("up_cos"):
            best = value[:, 0]
            print(f"{key}: best partner cosine median {np.median(best):.3f}, >=0.5 for {(best >= 0.5).mean():.1%}")
    print(f"saved {path} from {result['tokens']} tokens")


@app.function(volumes=VOLUMES, timeout=600)
def directions(run: str, latents: str, checkpoint: str = "final", name: str = "features"):
    """Save the unit decoder directions of some features, for control vectors outside Modal."""
    import torch
    from safetensors.torch import save_file

    from scope.sae import TopKSAE

    base = f"{OUTPUT_DIR}/{run}"
    sae = TopKSAE.load(f"{base}/{checkpoint}", device="cpu")
    picked = [int(l) for l in latents.split(",")]
    rows = {str(l): sae.W_dec[l].detach().float().contiguous() for l in picked}
    os.makedirs(f"{base}/directions", exist_ok=True)
    path = f"{base}/directions/{name}-{checkpoint}.safetensors"
    save_file(rows, path, metadata={"layer": str(sae.cfg.layer), "input_scale": str(sae.cfg.input_scale)})
    scope_volume.commit()
    print(f"saved {len(rows)} directions to {path}; norms {[round(float(torch.norm(v)), 3) for v in rows.values()]}")


@app.function(volumes=VOLUMES, secrets=[meta_secret], cpu=2, timeout=3600)
def retarget(run: str, workers: int = 16, checkpoint: str = "final"):
    """Re-judge every archetype's steering for the archetype itself, and pick each one's best direction."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from scope.archetype import ARCHETYPES, slug, target_judge
    from scope.steerability import Filtered, Screened, stop_judging

    base = Path(f"{OUTPUT_DIR}/{run}/archetypes")
    stop = []  # set at the first filter rejection: it is recorded as a strike and nothing more is sent
    jobs = []
    for name, spec in ARCHETYPES.items():
        path = base / slug(name) / f"steerability-{checkpoint}.jsonl"
        if path.exists():
            jobs += [(name, spec, json.loads(line)) for line in open(path)]

    def run_one(job):
        name, spec, entry = job
        candidate = entry.get("name") or entry.get("latent")
        if stop:
            return name, {"candidate": candidate, "error": "not sent: stopped after a filter rejection"}
        try:
            return name, target_judge(name, spec, entry)
        except Screened as screened:
            return name, {"candidate": candidate, "error": str(screened)}
        except Filtered as filtered:
            stop.append(str(filtered))
            stop_judging(str(filtered))
            print(f"JUDGING OFF: Meta's content filter rejected {filtered}; nothing more is sent", flush=True)
            return name, {"candidate": candidate, "error": "Meta's content filter rejected it; retarget stopped"}
        except Exception as error:
            return name, {"candidate": entry.get("name") or entry.get("latent"), "error": str(error)}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run_one, jobs))
    summary = {}
    for name in ARCHETYPES:
        rows = [r for n, r in results if n == name]
        if not rows:
            continue
        (base / slug(name) / f"target-{checkpoint}.json").write_text(json.dumps(rows, indent=2))
        scored = sorted((r for r in rows if "target_score" in r), key=lambda r: -r["target_score"])
        summary[name] = {"best": scored[0] if scored else None, "all": rows}
        best = scored[0] if scored else {}
        print(f"{name:18} best {str(best.get('candidate')):16} {best.get('target_score')} x{best.get('best_strength')}  {best.get('verdict', '')}")
    (base / f"targets-{checkpoint}.json").write_text(json.dumps(summary, indent=2))
    scope_volume.commit()


@app.function(secrets=[meta_secret], cpu=2, timeout=3600)
def judge_many(jobs: list[dict], workers: int = 8) -> list[dict]:
    """Muse Spark grades steered texts generated elsewhere (a local Q4 model), with the sweep's judge.

    Each job is {"feature": {latent, name, label}, "contexts": [...], "baseline": [...],
    "steered": {strength: [...]}}; a job the judge could not grade comes back with its "error".
    """
    from concurrent.futures import ThreadPoolExecutor

    from scope.steerability import judge

    def one(job):
        try:
            steered = {float(s): answers for s, answers in job["steered"].items()}
            return judge(job["feature"], job["contexts"], job["baseline"], steered)
        except Exception as error:
            return {"latent": job["feature"].get("latent"), "name": job["feature"].get("name"), "error": str(error)[:400]}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


@app.local_entrypoint()
def judge_file(path: str, out: str):
    """Grade a local jobs file with judge_many and write the verdicts next to it."""
    import json

    jobs = [json.loads(line) for line in open(path)]
    results = judge_many.remote(jobs)
    with open(out, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"judged {sum('error' not in r for r in results)} of {len(jobs)}, {sum('error' in r for r in results)} failed")


@app.function(secrets=[meta_secret], cpu=1, timeout=3600)
def ask_many(prompts: list[str], workers: int = 8) -> list[dict]:
    """Muse Spark answers prompts written elsewhere: each comes back as {"reply"} or {"error"}."""
    from concurrent.futures import ThreadPoolExecutor

    from scope.llm import chat

    def one(prompt):
        try:
            return {"reply": chat(prompt)}
        except Exception as error:
            return {"error": str(error)[:400]}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, prompts))


@app.local_entrypoint()
def ask_file(path: str, out: str):
    """Answer a local jsonl of {"id", "prompt"} with ask_many; writes {"id", "reply" | "error"} lines."""
    import json

    jobs = [json.loads(line) for line in open(path)]
    results = ask_many.remote([job["prompt"] for job in jobs])
    with open(out, "w") as f:
        for job, row in zip(jobs, results):
            f.write(json.dumps({"id": job["id"], **row}) + "\n")
    print(f"answered {sum('reply' in r for r in results)} of {len(jobs)}, {sum('error' in r for r in results)} failed")
