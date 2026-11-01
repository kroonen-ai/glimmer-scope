"""Merge cross-layer match files from overlapping chunks into one, as layer_atlas.py reads it.

modal_app.py::crosslayer matches neighboring layers within the SAEs it loads; nine 64k SAEs do not
fit beside Glimmer on a 96 GB card, so the run goes in overlapping chunks (12-21, 21-30, 30-36)
and this joins their up/down partner arrays into L12-L15-...-L36.npz.

    python scripts/merge_crosslayer.py OUT_DIR CHUNK.npz [CHUNK.npz ...]
"""
import sys
from pathlib import Path

import numpy as np

out_dir, chunks = Path(sys.argv[1]), [np.load(p) for p in sys.argv[2:]]
merged, layers = {}, set()
for chunk in chunks:
    layers |= {int(x) for x in chunk["layers"]}
    for key in chunk.files:
        if key.startswith(("up_", "down_")):
            merged[key] = chunk[key]
merged["layers"] = np.array(sorted(layers))
merged["tokens"] = min(int(c["tokens"]) for c in chunks)
path = out_dir / f"{'-'.join(f'L{l}' for l in sorted(layers))}.npz"
np.savez_compressed(path, **merged)
print(f"merged {len(chunks)} chunks -> {path} ({len(merged) - 2} partner arrays)")
