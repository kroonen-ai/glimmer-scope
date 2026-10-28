"""Write steering directions as llama.cpp control vectors.

llama.cpp adds a control vector's ``direction.N`` tensor to the output of
layer N (0-indexed) at every token, the same place the steering hook adds it
here. Each file holds one unit direction times the layer's median residual
norm, so ``--control-vector-scaled file.gguf:0.5`` steers at the strength a
sweep calls 0.5. Several files passed together are summed.

    python -m scope.cvec directions.safetensors out/ --median 256.6 --name 20791=godfather
"""

import argparse
import json
from pathlib import Path

import numpy as np


def write_control_vector(path, layer: int, vector: np.ndarray, model_hint: str = "muse-glimmer") -> None:
    import gguf

    writer = gguf.GGUFWriter(str(path), "controlvector")
    writer.add_string("controlvector.model_hint", model_hint)
    writer.add_int32("controlvector.layer_count", 1)
    writer.add_tensor(f"direction.{layer}", np.ascontiguousarray(vector, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directions", help="safetensors of unit directions, from modal_app.py::directions or ::archetype")
    parser.add_argument("out_dir")
    parser.add_argument("--median", type=float, required=True, help="median residual norm at the layer")
    parser.add_argument("--name", action="append", default=[], help="KEY=NAME for the file name")
    args = parser.parse_args()

    names = dict(n.split("=", 1) for n in args.name)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    with safe_open(args.directions, "np") as f:
        layer = int(f.metadata()["layer"])
        for key in f.keys():
            direction = f.get_tensor(key).astype(np.float32)
            vector = direction / np.linalg.norm(direction) * args.median
            path = out / f"{names.get(key, key)}.gguf"
            write_control_vector(path, layer, vector)
            written[key] = path.name
    print(json.dumps({"layer": layer, "median": args.median, "files": written}, indent=2))


if __name__ == "__main__":
    main()
