"""Map Glimmer's personas: how similar the archetype directions are, and what they share.

    python scripts/persona_map.py ARCHETYPES_DIR MAP_FEATURES.safetensors OUT_DIR [LABELS.jsonl]

ARCHETYPES_DIR holds <slug>/directions-final.safetensors and <slug>/ranking-final.json
from modal_app.py::archetype; MAP_FEATURES holds SAE decoder rows from ::directions.
Writes persona-map.png (similarity heatmap and a 2D layout) and persona-map.json. FEATURES names
layer 24's features; for another layer pass its labels.jsonl so the features get their own names.
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

GROUP_COLORS = {"companion": "#d9480f", "anime": "#ae3ec9", "character": "#1971c2", "research": "#495057"}

# Directions worth naming in the checks, by latent.
FEATURES = {
    "52372": "generic stylized voice A", "62673": "generic stylized voice B", "62496": "generic stylized voice C",
    "15507": "archaic poetic diction", "41386": "supportive questioning",
    "642": "consult-a-professional disclaimer", "1857": "medical advice disclaimer", "2366": "medical disclaimer boilerplate",
    "417": "accuracy disclaimer", "1008": "AI denial of capability",
    "18659": "hedging words (likely)", "15463": "uncertain dialogue hedges", "14178": "hedging: can't conclude",
    "18574": "romantic intimacy and affection", "22998": "romantic physical affection", "12766": "jealousy and envy",
    "16910": "helpful AI assistant system prompt", "17444": "helpful AI assistant phrase",
    "16215": "irony markers", "23289": "anime and manga culture",
    "34885": "noir (single feature)", "20791": "Godfather", "26734": "WWII Britain", "18334": "waking up disoriented",
    "22313": "listening and support", "19672": "negative self-talk",
}

CHECKS = [
    ("girlfriend", "yandere"), ("girlfriend", "best friend"), ("friendly", "cozy companion"), ("friendly", "best friend"),
    ("kuudere", "cold"), ("sycophant", "contrarian"), ("sycophant", "friendly"), ("tsundere", "witty friend"),
    ("manipulator", "yandere"), ("manipulator", "sycophant"), ("hype coach", "genki"), ("hype coach", "shonen hero"),
    ("noir detective", "stoic philosopher"), ("grandparent", "wise mentor"), ("dandere", "cozy companion"),
]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> None:
    arch_dir, features_path, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scope.personas import ARCHETYPES, slug

    by_slug = {slug(name): name for name in ARCHETYPES}
    names, groups, vectors = [], [], []
    for path in sorted(arch_dir.glob("*/directions-final.safetensors")):
        name = by_slug.get(path.parent.name, path.parent.name)
        names.append(name)
        groups.append(ARCHETYPES.get(name, {}).get("group", "character"))
        vectors.append(unit(load_file(str(path))["mean_difference"].astype(np.float64)))
    V = np.stack(vectors)
    sim = V @ V.T
    feats = {k: unit(v.astype(np.float64)) for k, v in load_file(features_path).items()}
    names_of = FEATURES
    if len(sys.argv) > 4:
        names_of = {str(r["latent"]): r["label"] for r in map(json.loads, open(sys.argv[4]))}

    # order by average-linkage clustering on cosine distance
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "average"))

    # 2D layout: classical MDS on cosine distance
    n = len(names)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dist**2) @ J
    w, U = np.linalg.eigh(B)
    top = np.argsort(w)[::-1][:2]
    xy = U[:, top] * np.sqrt(np.maximum(w[top], 0))

    # what the personas share: first principal direction of all of them
    mean = V.mean(0)
    shared = unit(mean)
    share = V @ shared
    companions = [i for i, g in enumerate(groups) if g == "companion"]
    pc_var = None
    if len(companions) >= 3:
        s = np.linalg.svd(V[companions], compute_uv=False)
        pc_var = float(s[0] ** 2 / (s**2).sum())

    report = {
        "archetypes": names,
        "pairs": {f"{a} ~ {b}": round(float(sim[names.index(a), names.index(b)]), 3)
                  for a, b in CHECKS if a in names and b in names},
        "most_similar": sorted(
            ({"a": names[i], "b": names[j], "cos": round(float(sim[i, j]), 3)} for i in range(n) for j in range(i + 1, n)),
            key=lambda r: -r["cos"],
        )[:12],
        "least_similar": sorted(
            ({"a": names[i], "b": names[j], "cos": round(float(sim[i, j]), 3)} for i in range(n) for j in range(i + 1, n)),
            key=lambda r: r["cos"],
        )[:8],
        "shared_direction": {
            "cos_with_each": {names[i]: round(float(share[i]), 3) for i in np.argsort(-share)},
            "companions_first_component_variance": round(pc_var, 3) if pc_var is not None else None,
            "closest_features": sorted(
                ({"feature": k, "label": names_of.get(k, ""), "cos": round(float(shared @ f), 3)} for k, f in feats.items()),
                key=lambda r: -abs(r["cos"]),
            )[:8],
        },
        "features_vs_archetypes": {
            f"{k} {names_of.get(k, '')}": dict(
                sorted(((names[i], round(float(V[i] @ f), 3)) for i in range(n)), key=lambda kv: -abs(kv[1]))[:5]
            )
            for k, f in feats.items()
        },
    }
    (out_dir / "persona-map.json").write_text(json.dumps(report, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1.15, 1]})
    ordered = [names[i] for i in order]
    im = ax1.imshow(sim[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(n), ordered, rotation=70, ha="right", fontsize=9)
    ax1.set_yticks(range(n), ordered, fontsize=9)
    for tick, i in zip(ax1.get_xticklabels(), order):
        tick.set_color(GROUP_COLORS.get(groups[i], "black"))
    for tick, i in zip(ax1.get_yticklabels(), order):
        tick.set_color(GROUP_COLORS.get(groups[i], "black"))
    for a in range(n):
        for b in range(n):
            v = sim[order[a], order[b]]
            if a != b and abs(v) >= 0.5:
                ax1.text(b, a, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.7 else "black")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02, label="cosine similarity of mean-difference directions")
    ax1.set_title("How alike Glimmer's personas are at layer 24")

    for i, name in enumerate(names):
        ax2.scatter(xy[i, 0], xy[i, 1], s=120, color=GROUP_COLORS.get(groups[i], "black"), alpha=0.85)
        ax2.annotate(name, xy[i], textcoords="offset points", xytext=(6, 4), fontsize=10)
    for group, color in GROUP_COLORS.items():
        ax2.scatter([], [], color=color, label=group)
    ax2.legend(loc="best", frameon=False)
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.set_title("Persona map (distances = 1 - cosine)")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.suptitle("Muse Glimmer 30B: persona directions from matched replies, layer 24 residual stream", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "persona-map.png", dpi=130)
    print(json.dumps({k: report[k] for k in ("pairs", "most_similar", "least_similar")}, indent=1))
    print(json.dumps(report["shared_direction"], indent=1))


if __name__ == "__main__":
    main()
