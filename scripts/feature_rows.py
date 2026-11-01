"""Unit decoder rows of a layer's steerable features, for persona_map.py's MAP_FEATURES.

    python scripts/feature_rows.py RUN OUT.safetensors [min_score]
"""
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_map import decoder  # noqa: E402

run, out = Path(sys.argv[1]).expanduser(), sys.argv[2]
floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.66
latents = sorted({r["latent"] for name in ("steerability.jsonl", "steerability-personas.jsonl") if (run / name).exists()
                  for r in map(json.loads, open(run / name)) if r["score"] >= floor})
W = decoder(run / "final" / "sae.safetensors")
rows = {str(l): (np.asarray(W[l], dtype=np.float32) / np.linalg.norm(W[l])).astype(np.float32) for l in latents}
save_file(rows, out)
print(f"{len(rows)} steerable features (score >= {floor}) -> {out}")
