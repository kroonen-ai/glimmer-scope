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
| demo (next) | a steering server and a feature panel | Modal / local |

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

## Tests

```bash
uv run --python 3.12 --with torch --with numpy --with safetensors --with pytest pytest -q tests
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
