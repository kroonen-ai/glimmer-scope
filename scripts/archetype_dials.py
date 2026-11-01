"""Studio dials from one layer's archetypes, like layer 24's persona dials.

For each archetype, modal_app.py::retarget picked the best direction (the mean difference, the blend
of top features, or one SAE feature) and the strength where it plays the archetype best. Each becomes
a dial named <slug>-L<layer>, in the archetype's Studio group (companion, anime, character, or the
private research group), scaled by the layer's median residual norm like every other dial. An
archetype retarget could not score gets no dial and is listed with the reason.

    python scripts/archetype_dials.py RUN --median 176.5 --out RUN/dials-archetypes
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_map import decoder  # noqa: E402
from scope.cvec import write_control_vector  # noqa: E402
from scope.personas import slug  # noqa: E402

# Studio groups, as layer 24's archetype dials have them; dark and deceptive ones stay private.
GROUPS = {
    "pirate captain": "character", "hype coach": "character", "stoic philosopher": "character", "noir detective": "character",
    "wise mentor": "companion", "best friend": "companion", "girlfriend": "companion", "friendly": "companion",
    "cozy companion": "companion", "witty friend": "companion", "big sibling": "companion", "grandparent": "companion",
    "tsundere": "anime", "genki": "anime", "chuunibyou": "anime", "kuudere": "anime", "dandere": "anime",
    "catgirl": "anime", "shonen hero": "anime", "butler": "anime",
    "sycophant": "research", "contrarian": "research", "manipulator": "research", "yandere": "research",
    "gaslighter": "research", "confabulator": "research", "cold": "research", "preachy": "research",
}
KINDS = {"mean_difference": "mean difference", "blend": "blend of top features"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--median", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.expanduser()
    layer = json.loads((run / "final" / "cfg.json").read_text())["layer"]
    targets = json.loads((run / "archetypes" / "targets-final.json").read_text())
    W = decoder(run / "final" / "sae.safetensors")
    args.out.mkdir(parents=True, exist_ok=True)
    entries, missing = {}, []
    for name, group in GROUPS.items():
        best = (targets.get(name) or {}).get("best")
        if not best or not best.get("best_strength"):
            missing.append(f"{name}: {'not run' if name not in targets else 'no direction played the archetype'}")
            continue
        candidate = best["candidate"]
        if isinstance(candidate, int) or str(candidate).isdigit():
            direction = np.asarray(W[int(candidate)], dtype=np.float32)
            how = f"SAE feature {candidate} (best for {name})"
        else:
            candidate = str(candidate).replace("-", "_")  # retarget writes mean-difference
            direction = load_file(str(run / "archetypes" / slug(name) / "directions-final.safetensors"))[candidate]
            how = f"{name} archetype, {KINDS.get(candidate, candidate)}"
        dial = f"{slug(name)}-L{layer}"
        write_control_vector(args.out / f"{dial}.gguf", layer, direction / np.linalg.norm(direction) * args.median)
        entries[dial] = {"label": name[:1].upper() + name[1:], "description": best.get("verdict") or "",
                         "default": best["best_strength"], "layer": layer, "group": group,
                         "source": f"layer {layer} {how}, target score {best.get('target_score')}"}
    (args.out / "entries.json").write_text(json.dumps(entries, indent=2))
    (args.out / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    groups = {}
    for e in entries.values():
        groups[e["group"]] = groups.get(e["group"], 0) + 1
    print(f"layer {layer}: {len(entries)} archetype dials {groups}; {len(missing)} without one" + (f": {'; '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
