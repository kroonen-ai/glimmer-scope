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
def train(layer: int = 24, n_latents: int = 65536, k: int = 64, total_tokens: int = 100_000_000, tag: str = ""):
    from scope import model as glimmer
    from scope.config import SAEConfig, TrainConfig
    from scope.train import train as run_training

    model, tokenizer = glimmer.load()
    sae_cfg = SAEConfig(layer=layer, n_latents=n_latents, k=k)
    cfg = TrainConfig(total_tokens=total_tokens)
    out = f"{OUTPUT_DIR}/{run_name(layer, n_latents, k)}{'-' + tag if tag else ''}"
    final = run_training(model, tokenizer, sae_cfg, cfg, out, commit=scope_volume.commit)
    print(f"saved {final}")


@app.function(volumes=VOLUMES, gpu=GPU, timeout=6 * 3600)
def catalog(run: str, n_tokens: int = 5_000_000, checkpoint: str = "final"):
    from scope import model as glimmer
    from scope.interpret import build_catalog
    from scope.sae import TopKSAE

    model, tokenizer = glimmer.load()
    sae = TopKSAE.load(f"{OUTPUT_DIR}/{run}/{checkpoint}", device="cuda")
    stats = build_catalog(model, tokenizer, sae, n_tokens, f"{OUTPUT_DIR}/{run}/catalog.jsonl.gz")
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
