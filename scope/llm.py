"""Muse Spark on the Meta Model API, for labeling and judging features.

The API is OpenAI-compatible Chat Completions. Muse models reason before they
answer and the token budget covers both, so requests leave room for it; they
reject stop sequences and n > 1, and reasoning effort takes named levels only.
"""

import json
import os
import random
import re
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.meta.ai/v1/chat/completions"
MODEL = os.environ.get("JUDGE_MODEL", "muse-spark-1.3-contributor")


def chat(prompt: str, max_tokens: int = 4000, effort: str = "low", retries: int = 12) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "reasoning_effort": effort,
        }
    ).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {os.environ['META_API_KEY']}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
            return data["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as error:
            if error.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries - 1:
                raise RuntimeError(f"Meta API {error.code}: {error.read()[:300]!r}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(min(60, 2**attempt) + random.random())
    raise RuntimeError("unreachable")


def json_from(text: str):
    """The first JSON object or array in a reply, fenced or not."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    return json.JSONDecoder().raw_decode(text[min(starts) :])[0]
