"""Grade a jobs file (from q4_check.py) with Muse Spark on this machine, like modal_app.py::judge_file.

Runs where META_API_KEY is set (the AWS box reads it from SSM through aws/run.sh). Every request is
screened first (scope.safety.screen), and the first content-filter rejection is recorded as a strike
and stops all further sending; the dials not graded are left as they are by q4_check.py --apply.

    python scripts/judge_jobs.py JOBS.jsonl GRADED.jsonl [workers]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope.steerability import STRIKES, Filtered, Screened, judge  # noqa: E402


STOP: list = []  # set at the first filter rejection; later jobs are not sent


def one(job):
    feature = job["feature"]
    if STOP:
        return {"latent": feature.get("latent"), "name": feature.get("name"), "error": "not sent: stopped after a filter rejection"}
    try:
        steered = {float(s): answers for s, answers in job["steered"].items()}
        return judge(feature, job["contexts"], job["baseline"], steered)
    except Screened as screened:
        return {"latent": feature.get("latent"), "name": feature.get("name"), "error": json.loads(str(screened))["reason"]}
    except Filtered:
        # every rejection counts against the key: record the strike and send nothing more
        STOP.append(feature.get("label", ""))
        with open(STRIKES, "a") as strike_log:
            strike_log.write(feature.get("label", "") + "\n")
        return {"latent": feature.get("latent"), "name": feature.get("name"), "error": "Meta's content filter rejected it; grading stopped"}
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
