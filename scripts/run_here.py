"""Run one of modal_app's functions on this machine instead of on Modal.

For a GPU box laid out like the Modal containers: /scope holds the runs and data, /weights the
base model. Volume commits become no-ops (files are already on local disk) and secrets come from
the environment (META_API_KEY for labeling and judging). Arguments are the function's own, in
the same --kebab-case modal run uses.

    python scripts/run_here.py sweep --run layer36-64k-k64 --n-features 9000
    python scripts/run_here.py sweep --run layer36-64k-k64 --dry-run
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import modal_app  # noqa: E402


class LocalVolume:
    def commit(self):
        pass

    def reload(self):
        pass


def parse(fn, argv: list[str]) -> dict:
    params = inspect.signature(fn).parameters
    kwargs, i = {}, 0
    while i < len(argv):
        key = argv[i].removeprefix("--").replace("-", "_")
        if key not in params:
            raise SystemExit(f"{fn.__name__} has no argument {argv[i]}; it takes {', '.join(params)}")
        kind = params[key].annotation
        if kind is bool:
            kwargs[key], i = True, i + 1
            continue
        kwargs[key] = kind(argv[i + 1]) if kind in (int, float, str) else argv[i + 1]
        i += 2
    missing = [n for n, p in params.items() if p.default is inspect.Parameter.empty and n not in kwargs]
    if missing:
        raise SystemExit(f"{fn.__name__} needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
    return kwargs


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    fn = getattr(modal_app, sys.argv[1])._raw_f_
    argv = sys.argv[2:]
    dry = "--dry-run" in argv
    kwargs = parse(fn, [a for a in argv if a != "--dry-run"])
    print(f"{fn.__name__}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})", flush=True)
    if dry:
        return
    modal_app.scope_volume = LocalVolume()
    modal_app.weights = LocalVolume()
    fn(**kwargs)


if __name__ == "__main__":
    main()
