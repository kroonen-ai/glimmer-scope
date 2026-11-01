"""Grade a jobs file (from q4_check.py) with Muse Spark on this machine, like modal_app.py::judge_file.

Runs where META_API_KEY is set (the AWS box reads it from SSM through aws/run.sh). A job whose
feature label matches the safety patterns is never sent to Spark: it comes back with an error saying
so, and q4_check.py --apply leaves that dial as it is.

    python scripts/judge_jobs.py JOBS.jsonl GRADED.jsonl [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.safety import flagged  # noqa: E402
from scope.steerability import judge  # noqa: E402


def one(job):
    feature = job["feature"]
    if flagged(feature.get("label", "")):
        return {"latent": feature.get("latent"), "name": feature.get("name"), "error": "safety-flagged: not sent to Muse Spark"}
    try:
        steered = {float(s): answers for s, answers in job["steered"].items()}
        return judge(feature, job["contexts"], job["baseline"], steered)
    except Exception as error:
        return {"latent": feature.get("latent"), "name": feature.get("name"), "error": str(error)[:400]}


def main():
    jobs = [json.loads(line) for line in open(sys.argv[1])]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, jobs))
    with open(sys.argv[2], "w") as out:
        for row in results:
            out.write(json.dumps(row) + "\n")
    print(f"judged {sum('error' not in r for r in results)} of {len(jobs)}, {sum('error' in r for r in results)} not")


if __name__ == "__main__":
    main()
