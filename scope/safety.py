"""Safety patterns, kept out of this repository.

Glimmer Scope uses keyword patterns in three places: to screen what it sends to Muse Spark (Meta's
content filter counts every rejected request against the API key), to flag safety-relevant features
so the maps can hide or color them, and to keep dark or explicit personas out of the open dial
groups. The lists themselves are private. Point GLIMMER_SAFETY_PATTERNS at a JSON file of your own:

    {"categories": {"<category>": "<regex>", ...}, "extra": "<regex>", "dark": "<regex>",
     "screen": "<regex>", "private_personas": "<regex>"}

Without it nothing is screened or flagged, and everything else works the same.
"""
import json
import os
import re
from pathlib import Path

NEVER = re.compile(r"(?!x)x")  # matches nothing


def _load() -> dict:
    path = os.environ.get("GLIMMER_SAFETY_PATTERNS")
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


_PATTERNS = _load()


def _compile(pattern: str | None) -> re.Pattern:
    return re.compile(pattern, re.I) if pattern else NEVER


# category name -> pattern, for flag_safety.py and the maps' safety coloring
CATEGORIES: dict[str, str] = _PATTERNS.get("categories", {})
EXTRA: str = _PATTERNS.get("extra", "")
FLAGGED = _compile("|".join(f"(?:{p})" for p in [*CATEGORIES.values(), EXTRA] if p))


def flagged(text: str) -> bool:
    return bool(FLAGGED.search(text or ""))


# dark or explicit subjects, kept out of anything shared
DARK = _compile(_PATTERNS.get("dark"))
# what Meta's content filter reacts to, matched before a judge request is sent
SCREEN = _compile(_PATTERNS.get("screen"))
# personas kept out of the open dial groups
PRIVATE_PERSONAS = _compile(_PATTERNS.get("private_personas"))


def screen(text: str) -> str | None:
    """The first sensitive term in a prompt, or None when it is safe to send to Muse Spark."""
    hit = SCREEN.search(text or "")
    return hit.group(0) if hit else None
