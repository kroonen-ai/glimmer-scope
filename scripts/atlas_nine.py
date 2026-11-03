"""The nine-layer atlas from the layer maps: every feature that fired, lined up layer to layer.

Each layer keeps its own map layout (feature_map.py, every fired feature). The layers are lined up
by a similarity transform (rotation or mirror, scale, shift: Procrustes) fitted on mutual best
partners between neighboring layers from the merged cross-layer file, chained outward from the
middle layer, so a subject sits in about the same spot on every layer. Each point carries its best
partners one layer down and one up. Labels and steering grades come from the runs as they are now.
Writes results/layer-atlas.json (a manifest) and one results/atlas/L<n>.json per layer, since one
file with every point would pass the 25 MiB limit for a static file.

    python scripts/atlas_nine.py [runs dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_site import MAPS, RESULTS, RUNS, categories, grades  # noqa: E402

MIN_ALIGN = 0.3  # mutual best partners at least this alike anchor the alignment
MIN_LINK = 0.15  # partner cosine worth showing
PARTNERS = 4


def procrustes(src: np.ndarray, dst: np.ndarray):
    """Scale, orthogonal map (mirror allowed) and shift that best take src onto dst."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, s, vt = np.linalg.svd(a.T @ b)
    rot = u @ vt
    scale = s.sum() / (a ** 2).sum()
    return lambda xy: scale * (xy - ms) @ rot + md


def main():
    runs = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else RUNS
    layers = sorted(MAPS)
    maps = {layer: json.loads((RESULTS / MAPS[layer]).read_text()) for layer in layers}
    wanted = {f"up_idx_L{lo}_L{hi}" for lo, hi in zip(layers, layers[1:])}
    cross = next((c for c in map(np.load, sorted((runs / "crosslayer").glob("*.npz"))) if wanted <= set(c.files)), None)
    if cross is None:
        raise SystemExit(f"no cross-layer file covers {sorted(wanted)} in {runs / 'crosslayer'}")

    pos = {layer: {p["id"]: n for n, p in enumerate(maps[layer]["points"])} for layer in layers}
    xy = {layer: np.array([[p["x"], p["y"]] for p in maps[layer]["points"]], dtype=np.float64) for layer in layers}

    def anchors(lo, hi):
        up_idx, up_cos = cross[f"up_idx_L{lo}_L{hi}"], cross[f"up_cos_L{lo}_L{hi}"]
        down_idx = cross[f"down_idx_L{hi}_L{lo}"]
        pairs = [(pos[lo][a], pos[hi][int(up_idx[a, 0])]) for a in pos[lo]
                 if up_cos[a, 0] >= MIN_ALIGN and int(down_idx[int(up_idx[a, 0]), 0]) == a and int(up_idx[a, 0]) in pos[hi]]
        return np.array(pairs)

    # line the layers up from the middle outward, each onto its already placed neighbor
    mid = len(layers) // 2
    placed = {layers[mid]: xy[layers[mid]]}
    for i in range(mid + 1, len(layers)):
        lo, hi = layers[i - 1], layers[i]
        pairs = anchors(lo, hi)
        placed[hi] = procrustes(xy[hi][pairs[:, 1]], placed[lo][pairs[:, 0]])(xy[hi])
        print(f"L{lo}-L{hi}: {len(pairs)} anchors", flush=True)
    for i in range(mid - 1, -1, -1):
        lo, hi = layers[i], layers[i + 1]
        pairs = anchors(lo, hi)
        placed[lo] = procrustes(xy[lo][pairs[:, 0]], placed[hi][pairs[:, 1]])(xy[lo])
        print(f"L{lo}-L{hi}: {len(pairs)} anchors", flush=True)

    out_dir = RESULTS / "atlas"
    out_dir.mkdir(exist_ok=True)
    files = []
    for li, layer in enumerate(layers):
        run = runs / f"layer{layer}-64k-k64"
        labels = {}
        if (run / "labels.jsonl").exists():
            labels = {r["latent"]: r for r in map(json.loads, open(run / "labels.jsonl"))}
        graded, cats = grades(run), categories(layer)

        def links(key_idx, key_cos, other):
            if key_idx not in cross:
                return {}
            idx, cos = cross[key_idx], cross[key_cos]
            return {lat: row for lat in pos[layer]
                    if (row := [[int(j), round(float(c), 2)] for j, c in zip(idx[lat, :PARTNERS], cos[lat, :PARTNERS])
                                if c >= MIN_LINK and int(j) in pos[other]])}

        up = links(f"up_idx_L{layer}_L{layers[li + 1]}", f"up_cos_L{layer}_L{layers[li + 1]}", layers[li + 1]) if li + 1 < len(layers) else {}
        down = links(f"down_idx_L{layer}_L{layers[li - 1]}", f"down_cos_L{layer}_L{layers[li - 1]}", layers[li - 1]) if li else {}
        points = []
        for p, (x, y) in zip(maps[layer]["points"], placed[layer]):
            lat, r = p["id"], labels.get(p["id"])
            labeled = bool(r and r.get("label"))
            q = {"i": lat, "x": round(float(x), 2), "y": round(float(y), 2),
                 "l": r["label"] if labeled else "unlabeled", "k": (r.get("kind") or "unclear") if labeled else "unlabeled",
                 "r": p["region"]}
            if lat in graded:
                q["s"] = graded[lat][0]
                if graded[lat][1]:
                    q["j"] = 1
            if lat in cats:
                q["f"] = cats[lat]
            if lat in up:
                q["u"] = up[lat]
            if lat in down:
                q["d"] = down[lat]
            points.append(q)
        regions = []
        for reg in maps[layer]["regions"]:
            members = placed[layer][[pos[layer][p["id"]] for p in maps[layer]["points"] if p["region"] == reg["id"]]]
            regions.append({"id": reg["id"], "name": reg["name"], "size": int(len(members)),
                            "x": round(float(np.median(members[:, 0])), 2) if len(members) else 0,
                            "y": round(float(np.median(members[:, 1])), 2) if len(members) else 0})
        data = {"layer": layer, "points": points, "regions": regions,
                "tested": sum(1 for q in points if "s" in q), "labeled": sum(1 for q in points if q["l"] != "unlabeled")}
        (out_dir / f"L{layer}.json").write_text(json.dumps(data, separators=(",", ":")))
        files.append(f"atlas/L{layer}.json")
        print(f"layer {layer}: {len(points)} points, {len(up)} with links up, {len(down)} down", flush=True)
    manifest = {"layers": layers, "files": files, "maps": [], "names": {}, "tokens": int(cross["tokens"])}
    (RESULTS / "layer-atlas.json").write_text(json.dumps(manifest))
    print(f"wrote {RESULTS / 'layer-atlas.json'} and {len(files)} layer files")


if __name__ == "__main__":
    main()
