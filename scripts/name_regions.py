"""Muse Spark names the regions of the shareable maps, from the labels that stay in them.

The keyword names feature_map.py gives ("seo · spammy · essay") say which words are common, not
what a region is about. This asks Spark for a 2 to 4 word name per region, from a sample of the
labels left after public_site.py's dual-use filter, so a name never draws on a dropped feature.

    python scripts/name_regions.py jobs results/region-jobs.jsonl
    modal run modal_app.py::ask_file --path results/region-jobs.jsonl --out results/region-replies.jsonl
    python scripts/name_regions.py apply results/region-replies.jsonl     # adds to results/region-names.json

`jobs` leaves out regions already named, so a rerun only asks for what failed.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from public_site import NAMES, public_data  # noqa: E402
from scope.llm import json_from  # noqa: E402
from scope.safety import screen  # noqa: E402

PER_PROMPT = 10
SAMPLE = 30
PROMPT = """These are regions of a map of the features of a sparse autoencoder trained on a language model. \
Each feature has a short label saying what it responds to, and each region groups features that write similar \
things into the model.

Give every region below a short name, 2 to 4 words, that says what most of its features have in common, the way \
you would label an area on a map. Plain words, lowercase except proper nouns, no quotes. If a region is a mix, \
name its main theme. Two regions can share a theme but should not get the same name.

{regions}

Answer with only a JSON object from region number to name, like {{"1": "...", "2": "..."}}."""


def regions():
    """(key, size, labels) for every region with members, on the atlas and the flat maps."""
    atlas, maps, _ = public_data()
    sources = [("atlas", m["layer"], m["regions"], m["points"], "r", "l") for m in atlas["maps"]]
    sources += [("map", layer, d["regions"], d["points"], "region", "label") for layer, d in maps.items()]
    for prefix, layer, regs, points, rkey, lkey in sources:
        members = {}
        for p in points:
            members.setdefault(p[rkey], []).append(p[lkey])
        for reg in regs:
            if members.get(reg["id"]):
                yield f"{prefix}/{layer}/{reg['id']}", members[reg["id"]]


def jobs(path: Path):
    done = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    todo = [(key, labels) for key, labels in regions() if key not in done]
    rng = random.Random(0)
    with open(path, "w") as f:
        for start in range(0, len(todo), PER_PROMPT):
            batch = todo[start:start + PER_PROMPT]
            blocks = []
            for n, (key, labels) in enumerate(batch, 1):
                # labels the judge screen would stop stay out of the sample, so the prompt can be sent
                labels = [label for label in labels if not screen(label)] or labels
                sample = rng.sample(labels, min(SAMPLE, len(labels)))
                blocks.append(f"Region {n} ({len(labels)} features): " + "; ".join(sample))
            f.write(json.dumps({"id": [key for key, _ in batch], "prompt": PROMPT.format(regions="\n\n".join(blocks))}) + "\n")
    print(f"{len(todo)} regions to name in {-(-len(todo) // PER_PROMPT)} prompts -> {path}")


def apply(path: Path):
    names = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    failed = 0
    for row in map(json.loads, open(path)):
        keys = row["id"]
        try:
            answer = json_from(row["reply"])
            for n, key in enumerate(keys, 1):
                name = str(answer.get(str(n), "")).strip().strip("\"'.").strip()
                if name and len(name) <= 48:
                    names[key] = name
                else:
                    failed += 1
        except (KeyError, ValueError, AttributeError):
            print(f"no names for {keys[0]}..: {row.get('error') or row.get('reply', '')[:200]!r}")
            failed += len(keys)
    NAMES.write_text(json.dumps(names, indent=1, sort_keys=True))
    print(f"{len(names)} regions named, {failed} not -> {NAMES}")


if __name__ == "__main__":
    {"jobs": jobs, "apply": apply}[sys.argv[1]](Path(sys.argv[2]))
