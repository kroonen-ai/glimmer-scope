"""Answer a jsonl of {"id", "prompt"} with Muse Spark on this machine, like modal_app.py::ask_file.

For machines where META_API_KEY is set (the AWS box reads it from SSM). Each prompt is screened first
(scope.safety.screen); a screened one comes back with "error" and the reason. The first content-filter
rejection is recorded as a strike (scope.steerability.STRIKES) and nothing more is sent.

    python scripts/ask_jobs.py JOBS.jsonl REPLIES.jsonl [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.llm import chat  # noqa: E402
from scope.safety import screen  # noqa: E402
from scope.steerability import STRIKE_BUDGET, STRIKES, strikes  # noqa: E402

STOP: list = []


def one(job):
    if STOP or strikes() >= STRIKE_BUDGET:
        return {"id": job["id"], "error": "not sent: filter strikes"}
    if term := screen(job["prompt"]):
        return {"id": job["id"], "error": f"screened: mentions {term!r}"}
    try:
        return {"id": job["id"], "reply": chat(job["prompt"])}
    except RuntimeError as error:
        if "content_policy" in str(error):
            STOP.append(job["id"])
            with open(STRIKES, "a") as strike_log:
                strike_log.write(f"ask_jobs {job['id']}\n")
        return {"id": job["id"], "error": str(error)[:300]}


def main():
    jobs = [json.loads(line) for line in open(sys.argv[1])]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, jobs))
    with open(sys.argv[2], "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"answered {sum('reply' in r for r in rows)} of {len(rows)}, {sum('error' in r for r in rows)} not")


if __name__ == "__main__":
    main()
