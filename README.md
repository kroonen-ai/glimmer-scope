# Glimmer Scope

Find the features inside Muse Glimmer 30B that can steer it, and rank them by
how well they steer.

A sparse autoencoder (SAE) trained on one layer of Glimmer's residual stream
splits the model's internal state into tens of thousands of features, most of
which stand for one recognizable thing. Adding a feature's direction back into
that layer during generation pushes the model toward it. Most features steer
poorly or break the text, so the pipeline scores every candidate and keeps the
ones that work.

## Pipeline

| Stage | What it does | Where |
| --- | --- | --- |
| `download` | Glimmer's BF16 weights (60 GB) into a Modal volume | Modal CPU |
| `smoke` | loads the model, measures residual norms at layers 12/24/36, tokens per second, and checks that a steering hook changes generation | Modal GPU |
| `train` | streams web text (FineWeb-Edu) and chat (UltraChat, in Glimmer's own template) through the model up to the chosen layer, and trains a TopK SAE on the activations | Modal GPU, hours |
| `catalog` | for every latent: firing rate, mean and peak activation, and the 16 contexts where it fires hardest | Modal GPU |
| `label` | Muse Spark 1.3 Contributor names each feature from its top contexts, with a kind and a guess at how steerable it is | Modal CPU + Meta API |
| `sweep` | each candidate is added at several strengths (fractions of the layer's median residual norm) to four neutral prompts, answering straight in the user channel; Muse Spark rates concept and fluency per strength and says what the steering actually did; the score is the best concept x fluency / 9 | Modal GPU + Meta API |
| `top` | the best-steering features so far, with a sample | Modal |
| `archetype` | Muse Spark writes matched archetype/plain answer pairs; features are ranked by how consistently they fire more on the archetype side, then each, their blend, and the plain mean-difference vector are steered and judged | Modal GPU + Meta API |
| `directions` | saves chosen features' unit decoder directions for use outside Modal | Modal |

Defaults: layer 24 of 52 (0-indexed), 65,536 latents, k = 64, 100M training
tokens. The forward pass stops right after the SAE's layer, so collection costs
about half a full forward.

## Run

```bash
modal run modal_app.py::download
modal run modal_app.py::smoke
modal run --detach modal_app.py::train --layer 24 --total-tokens 100000000
modal run modal_app.py::catalog --run layer24-64k-k64
modal run modal_app.py::label --run layer24-64k-k64 --limit 20000
modal run modal_app.py::sweep --run layer24-64k-k64 --n-features 1000
modal run modal_app.py::top --run layer24-64k-k64
modal run modal_app.py::steer --run layer24-64k-k64 --latent 1234 --strengths 0,4,8,16
```

`GLIMMER_GPU` picks the card (default `H200`). Outputs land in the
`glimmer-scope` Modal volume under the run name. Labeling and judging read
`META_API_KEY` from the Modal secret `meta`; `JUDGE_MODEL` overrides the model.

## Steering a quantized Glimmer locally

llama.cpp control vectors add a `direction.N` tensor to the output of layer N
at every token, which is the same place the steering hook works. `scope.cvec`
turns saved directions into control-vector files scaled by the layer's median
residual norm (256.6 at layer 24 in BF16), so `--control-vector-scaled
file.gguf:0.6` matches strength 0.6 in a sweep. The directions carry over to
the Q4 GGUF without retuning.

```bash
modal run modal_app.py::directions --run layer24-64k-k64 --latents 20791,34885
modal volume get glimmer-scope layer24-64k-k64/directions/features-final.safetensors .
uv run --with gguf --with numpy --with safetensors python -m scope.cvec \
  features-final.safetensors cvec/ --median 256.6 --name 20791=godfather --name 34885=noir
llama-server -m muse-glimmer-30B-kquant-17gb.gguf --control-vector-scaled cvec/noir.gguf:0.55
```

Several `--control-vector-scaled` flags add up, so personas combine.
`llama-swap/glimmer-personas.yaml` serves a set of them as separate models,
and `scripts/q4test.py` and `scripts/convo.py` compare strengths and hold a
multi-turn chat. Steering compounds over a conversation, because the model
reads its own steered replies back, so chat strengths sit a little below the
single-reply best.

## Glimmer Studio: live sliders

`llama.cpp/0001-server-per-request-steering.patch` (against llama.cpp
c0fd786) teaches llama-server to mix control vectors per request:

- `--steering-dir DIR` loads every `.gguf` control vector in `DIR`, named by
  file stem; an optional `steering.json` there adds a label, description,
  source and suggested strength for each.
- Requests take `"steering": {"noir": 0.55, "british": 0.3}` (or an array of
  `{name, strength}`) on `/v1/chat/completions` and `/completion`; unknown
  names are a 400. Vectors passed with `--control-vector` stay underneath.
- `GET /steering` lists the vectors.
- A slot whose steering changes drops its cached prompt, and the RAM prompt
  cache is off, so no reply reuses KV computed under another mix.

`studio/index.html` is the slider UI: one dial per vector, streaming chat,
the mix shown on every reply, and an optional unsteered reply side by side.
Serve it with `--path studio/` and open the server's root, or through
llama-swap at `/upstream/glimmer-studio/` (see `llama-swap/`).
`scripts/live_test.py` checks the server end to end.

## Tests

```bash
uv run --python 3.12 --with torch --with numpy --with safetensors --with gguf --with pytest pytest -q tests
```

## Notes

- Glimmer's decoder layers return the residual stream as a plain tensor, so a
  forward hook on `model.model.language_model.layers[i]` both reads and steers
  it.
- The first token of each row and attention-sink outliers (norm above 10x the
  median) are left out of training.
- Activations are scaled so their mean squared norm equals the model width; the
  scale is saved with the SAE, and `steering_vector` converts back to raw units.
- Chat template reasoning is set with the `reasoning_strength` template
  argument (`low`, `medium`, `high`).
