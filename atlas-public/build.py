"""Build the public Glimmer Scope atlas: R2 files for the maps, a D1 table for search, and the page.

    GLIMMER_SAFETY_PATTERNS=private/safety-patterns.json python atlas-public/build.py

Reads the nine-layer atlas the site builds (results/site/atlas/L*.json) and keeps only what proves the work
without giving anything away: where each feature sits, its kind and region, its label, and its links to the
neighboring layers. Left out, before anything is uploaded:

- steering: no score, no strength, nothing that says whether a feature steers;
- vectors: no decoder rows, activations or norms (the maps only carry 2D positions);
- safety: every feature in an upsetting category, flagged, or matching the private patterns is left out with
  everything that would point at it (its links, its share of a region), and so is any region name that matches.

Writes atlas-public/out/: r2/ (atlas.json and one file per layer), d1/features.sql and site/ (the page).
"""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GLIMMER_SAFETY_PATTERNS", str(ROOT / "private" / "safety-patterns.json"))
from scope import safety  # noqa: E402
import home  # noqa: E402  (atlas-public/home.py)

# more terms for the public atlas only (gore, remains, graphic harm), kept private like the patterns above
EXTRA_FILE = Path(os.environ.get("GLIMMER_ATLAS_EXTRA", ROOT / "private" / "atlas-public-extra.json"))
_extra = json.loads(EXTRA_FILE.read_text()) if EXTRA_FILE.exists() else None
EXTRA = re.compile("|".join(f"(?:{p})" for p in _extra["patterns"]), re.I) if _extra else None
# and private names the page itself must never carry
PAGE_WORDS = re.compile("|".join(_extra["page_words"]), re.I) if _extra else None
# the report carries its author's byline, the only private word it may have
REPORT_WORDS = re.compile("|".join(w for w in _extra["page_words"] if "robin" not in w), re.I) if _extra else None

SRC = ROOT / "results" / "site"
OUT = ROOT / "atlas-public" / "out"
KEEP = ("i", "x", "y", "k", "r", "u", "d")  # id, position, kind, region, links up and down
LINKS = {
    "code": "https://github.com/kroonen-ai/glimmer-scope",
    "space": "https://huggingface.co/spaces/rob-x-ai/glimmer-scope",
    "saes": "https://huggingface.co/kroonen-ai/glimmer-scope-saes",
}


def unsafe(text: str) -> bool:
    return bool(text) and bool(safety.FLAGGED.search(text) or safety.DARK.search(text) or safety.screen(text)
                               or EXTRA.search(text))


def leave_out(m: dict) -> set:
    return {p["i"] for p in m["points"] if p.get("h") or p.get("f") or unsafe(p.get("l", ""))}


def clean_layer(m: dict, out: set, out_up: set, out_down: set) -> tuple[dict, dict]:
    # a feature on an upsetting topic is left out entirely: no point, no position, no region count and no link
    # to it from the layers next to it (u links go one layer deeper, d one layer back), so nothing marks
    # which feature ids those are
    points, sizes = [], {}
    for p in m["points"]:
        if p["i"] in out:
            continue
        q = {k: p[k] for k in KEEP if k in p}
        q["l"] = p.get("l", "")
        for k, gone in (("u", out_up), ("d", out_down)):
            if k in q:
                q[k] = [[j, round(c, 2)] for j, c in q[k] if j not in gone]
                if not q[k]:
                    del q[k]
        points.append(q)
        sizes[q.get("r")] = sizes.get(q.get("r"), 0) + 1
    regions = []
    for r in m["regions"]:
        if not sizes.get(r["id"]):
            continue
        name = "" if r.get("h") or unsafe(r.get("name", "")) else r.get("name", "")
        regions.append({"id": r["id"], "name": name, "size": sizes[r["id"]], "x": r["x"], "y": r["y"]})
    stats = {"layer": m["layer"], "features": len(points), "left_out": len(out), "regions": len(regions)}
    return {"layer": m["layer"], "points": points, "regions": regions, "labeled": len(points)}, stats


def sql_text(v) -> str:
    return "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"


def write_sql(layers: list[dict], path: Path):
    with path.open("w") as f:
        f.write("DROP TABLE IF EXISTS features;\nDROP TABLE IF EXISTS regions;\nDROP TABLE IF EXISTS features_fts;\n")
        f.write("CREATE TABLE features (layer INTEGER, id INTEGER, label TEXT, kind TEXT, region INTEGER, "
                "up TEXT, down TEXT, PRIMARY KEY (layer, id));\n")
        f.write("CREATE TABLE regions (layer INTEGER, id INTEGER, name TEXT, size INTEGER, PRIMARY KEY (layer, id));\n")
        for m in layers:
            rows = []
            for p in m["points"]:
                rows.append(f"({m['layer']},{p['i']},{sql_text(p.get('l'))},{sql_text(p.get('k'))},{p.get('r', 'NULL')},"
                            f"{sql_text(json.dumps(p['u']) if 'u' in p else None)},{sql_text(json.dumps(p['d']) if 'd' in p else None)})")
            for i in range(0, len(rows), 400):
                f.write("INSERT INTO features VALUES " + ",".join(rows[i:i + 400]) + ";\n")
            regs = [f"({m['layer']},{r['id']},{sql_text(r['name'] or None)},{r['size']})" for r in m["regions"]]
            f.write("INSERT INTO regions VALUES " + ",".join(regs) + ";\n")
        # full-text search over the public labels only
        f.write("CREATE VIRTUAL TABLE features_fts USING fts5(label, layer UNINDEXED, id UNINDEXED);\n")
        f.write("INSERT INTO features_fts (label, layer, id) SELECT label, layer, id FROM features WHERE label IS NOT NULL AND label != '';\n")


LOADER_CSS = """  .loading { position: fixed; inset: 0; display: grid; place-items: center; color: var(--muted); pointer-events: none;
    transition: opacity .45s ease; }
  .loading.gone { opacity: 0; }
  .ld { display: grid; justify-items: center; gap: 12px; text-align: center; padding: 0 16px; }
  .ld-stack { width: 220px; height: auto; overflow: visible; }
  .ld-stack .lay { transform-box: fill-box; transform-origin: center; }
  .ld-stack polygon { fill: var(--panel); stroke: var(--line); stroke-width: 1.5; transition: fill .4s ease;
    animation: ld-wave 1.8s ease-in-out infinite; animation-delay: calc(var(--i) * -0.2s); }
  .ld-stack circle { opacity: 0; transition: opacity .35s ease; transition-delay: calc(var(--j) * 25ms); }
  .ld-stack .lay.done { animation: ld-land .5s cubic-bezier(.2,.8,.2,1) both; }
  .ld-stack .lay.done polygon { fill: var(--accent-tint); stroke: var(--accent); animation: none; }
  .ld-stack .lay.done circle { opacity: .9; }
  .ld-stack text { font-size: 10px; font-weight: 600; fill: var(--muted); font-variant-numeric: tabular-nums; transition: fill .4s ease; }
  .ld-stack text.done { fill: var(--accent-ink); }
  @keyframes ld-wave { 0%, 100% { stroke: var(--line); } 50% { stroke: var(--accent); } }
  @keyframes ld-land { from { transform: translateY(-7px); } to { transform: none; } }
  .ld-title { color: var(--ink); font-weight: 600; font-size: 15px; margin-top: 4px; }
  .ld-bar { width: 220px; max-width: 70vw; height: 4px; border-radius: 99px; background: var(--line); overflow: hidden; }
  .ld-bar i { display: block; height: 100%; background: var(--accent); transform: scaleX(0); transform-origin: left; transition: transform .25s ease-out; }
  .ld-sub { font-size: 12px; font-variant-numeric: tabular-nums; min-height: 1.3em; }
  @media (prefers-reduced-motion: reduce) {
    .ld-stack polygon, .ld-stack .lay.done { animation: none; } .ld-stack circle, .ld-bar i, .loading { transition: none; }
  }
"""
DOT_COLORS = ("#0064e0", "#0064e0", "#0064e0", "#8b5cf6", "#14a38b", "#f29a38")  # the atlas kinds, mostly concept


def loader_html(layers: list[int]) -> str:
    # one sheet per layer, the deeper ones higher, like the stacked view; as a layer's map arrives its sheet
    # lights up and a scatter of features lands on it
    import random
    rnd = random.Random(7)
    groups, labels = [], []
    for i, layer in enumerate(layers):
        y = 22 + (len(layers) - 1 - i) * 24
        dots = []
        for j in range(16):
            while True:
                u, v = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
                if abs(u) + abs(v) <= 0.8:
                    break
            dots.append(f'<circle style="--j:{j}" cx="{125 + u * 68:.1f}" cy="{y + v * 10:.1f}" r="1.7" fill="{rnd.choice(DOT_COLORS)}"/>')
        groups.append(f'<g class="lay" style="--i:{i}"><polygon points="57,{y} 125,{y - 10} 193,{y} 125,{y + 10}"/>{"".join(dots)}</g>')
        labels.append(f'<text x="44" y="{y + 3.5}" text-anchor="end">L{layer}</text>')
    h = 22 + (len(layers) - 1) * 24 + 14
    return (f'<div class="loading" id="loading" role="status" aria-live="polite"><div class="ld">'
            f'<svg class="ld-stack" viewBox="0 0 210 {h}" aria-hidden="true">{"".join(groups)}{"".join(labels)}</svg>'
            f'<div class="ld-title">Loading the atlas</div><div class="ld-bar"><i id="ld-fill"></i></div>'
            f'<div class="ld-sub" id="ld-sub">Fetching {len(layers)} layer maps</div></div></div>')


LOADER_JS = """const loading = document.getElementById("loading");
const ldFill = document.getElementById("ld-fill"), ldSub = document.getElementById("ld-sub");
const sheets = [...loading.querySelectorAll(".lay")], tags = [...loading.querySelectorAll("text")];
// a file read as it streams in, so the bar moves with the bytes
async function getJSON(url, onBytes) {
  const r = await fetch(url); if (!r.ok) throw new Error(r.status);
  if (!onBytes || !r.body) return r.json();
  const reader = r.body.getReader(), parts = [];
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); onBytes(value.length); }
  return new Response(new Blob(parts)).json();
}
let DATA = null;
try {
  DATA = await getJSON(SRC);
  const n = DATA.files.length, total = (DATA.bytes || []).reduce((a, b) => a + b, 0);
  let got = 0, done = 0;
  const mb = (b) => Math.round(b / 1e6);
  const show = () => {
    ldFill.style.transform = `scaleX(${Math.min(1, total ? got / total : done / n)})`;
    ldSub.textContent = `${done} of ${n} layers` + (total ? ` · ${mb(got)} of ${mb(total)} MB` : "");
  };
  DATA.maps = await Promise.all(DATA.files.map(async (f, i) => {
    const m = await getJSON(f, (b) => { got += b; show(); });
    done++; sheets[i]?.classList.add("done"); tags[i]?.classList.add("done"); show();
    return m;
  }));
  ldSub.textContent = "Drawing the maps";
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
}
catch (e) { loading.querySelector(".ld-title").textContent = "The atlas didn't load. Reload the page to try again."; throw e; }
loading.classList.add("gone");
setTimeout(() => loading.remove(), 500);
"""


def page(stats: list[dict]) -> str:
    from public_site import THEME_EARLY, THEME_TOGGLE, brand

    total = sum(s["features"] for s in stats)
    labeled = total + sum(s["left_out"] for s in stats)

    t = (ROOT / "scripts" / "layer_atlas_template.html").read_text()

    def rep(a, b, count=1):
        nonlocal t
        assert t.count(a) >= 1, a[:60]
        t = t.replace(a, b, count)

    # nothing about steering or safety on the public page
    rep('    <option value="score">Color: steering</option>\n', "")
    rep('    <option value="safety">Color: safety-relevant</option>\n', "")
    t = re.sub(r"for \(const m of M\) \{ SAFE\.apply\(m\.points, \[\"l\"\]\); SAFE\.apply\(m\.regions, \[\"name\"\], \"\"\); \}\n", "", t)
    t = re.sub(r"const hiddenName = .*?\n", "const hiddenName = () => false;\n", t, count=1)
    t = re.sub(r"const safetyOpt = .*?\nSAFE\.onChange\(.*?\);\n", "", t, count=1, flags=re.S)
    rep("· ${steerWord(p.s)}` +\n    (p.f ? ` · flagged: ${esc(p.f)}` : \"\") + `</div>`;", "` + `</div>`;")
    rep(" · #${p.i} · ${steerWord(p.s)}</span>", " · #${p.i}</span>")
    rep(" || (b.s ?? -1) - (a.s ?? -1)", "")
    t = re.sub(r" · \$\{m\.tested\.toLocaleString\(\)\} steering-test[a-z]*", "", t)
    # and none of the code behind those options either, so the page source says nothing about them
    for pat in (r"const scoreColor = .*?\n", r"const SAFETY = \{.*?\};\n", r"// the judge's score.*?\n",
                r"function steerWord\(s\) \{.*?\n\}\n", r'  if \(mode === "score"\) return .*?\n',
                r'  if \(mode === "safety"\) return .*?\n'):
        t, n = re.subn(pat, "", t, count=1, flags=re.S)
        assert n == 1, pat
    t, n = re.subn(r'  if \(mode === "score"\) \{\n.*?\n  \} else if \(mode === "kind"\)', '  if (mode === "kind")', t,
                   count=1, flags=re.S)
    assert n == 1
    rep("const hiddenName = () => false;\n", "")
    t = re.sub(r"const SAFE = window\.SAFE \|\| .*?\n", "", t, count=1)
    rep("esc(hiddenName(M[li].layer, j) ? SAFE.TEXT : (DATA.names", "esc((DATA.names")
    # what the page is, with the size of the work, where people first look
    rep("`Same subject, same place on every layer.",
        f"`{len(stats)} sparse autoencoders on Muse Glimmer 30B, layers 12 to 36: {labeled:,} features labeled by "
        f"Muse Spark 1.3, {total:,} shown here (the ones on upsetting topics are left out). Same subject, same place on every layer.")
    # search goes to D1 through the worker, across all nine layers
    t = t.replace("qEl.addEventListener(\"input\", search);",
                  "let searchTimer = null;\n"
                  "qEl.addEventListener(\"input\", () => { clearTimeout(searchTimer); searchTimer = setTimeout(remoteSearch, 180); });\n"
                  "async function remoteSearch() {\n"
                  "  const q = qEl.value.trim();\n"
                  "  if (q.length < 2 || /^\\d+$/.test(q)) return search();\n"
                  "  try {\n"
                  "    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`); if (!r.ok) throw new Error(r.status);\n"
                  "    const hits = (await r.json()).results || [];\n"
                  "    const li = new Map(M.map((m, i) => [m.layer, i]));\n"
                  "    search(hits.map((h) => byId[li.get(h.layer)]?.get(h.id)).filter(Boolean));\n"
                  "  } catch (e) { search(); }\n"
                  "}")
    t = t.replace("function search() {\n  const q = qEl.value.trim().toLowerCase();",
                  "function search(remote) {\n  const q = qEl.value.trim().toLowerCase();")
    t = t.replace("  found = all.filter((p) => p.l.toLowerCase().includes(q) || String(p.i) === q)",
                  "  found = (remote || all.filter((p) => p.l.toLowerCase().includes(q) || String(p.i) === q))")
    # a loading animation: the nine layers fill in as their maps arrive
    rep("  .loading { position: fixed; inset: 0; display: grid; place-items: center; color: var(--muted); }\n", LOADER_CSS)
    rep('<div class="loading" id="loading">Loading the atlas…</div>', loader_html([st["layer"] for st in stats]))
    load_start = t.index("const loading = document.getElementById(\"loading\");")
    load_end = t.index("loading.remove();\n", load_start) + len("loading.remove();\n")
    t = t[:load_start] + LOADER_JS + t[load_end:]
    links = '<a href="/">Home</a>' + "".join(f'<a href="{u}" target="_blank" rel="noopener">{n}</a>' for n, u in
                    [("Code", LINKS["code"]), ("Try steering", LINKS["space"]), ("The 9 SAEs", LINKS["saes"])])
    head = (f'<div class="bar-head">{brand().replace('href="./"', 'href="/"')}<nav class="site-nav public-nav">{links}</nav>'
            f'<span class="toggles">{THEME_TOGGLE}</span></div><script src="/theme.js"></script>')
    t = (t.replace("<!--THEME-->", '<link rel="icon" href="/favicon.svg" type="image/svg+xml">'
                   '<link rel="stylesheet" href="/theme.css">' + THEME_EARLY)
          .replace("<!--HEADER-->", "").replace("<!--BARHEAD-->", head)
          .replace("/*SRC*/null", json.dumps("/data/atlas.json"))
          .replace("<title>Glimmer Layer Atlas</title>", "<title>Glimmer Scope · nine-layer atlas</title>"))
    return t


# words that only the private build may contain: anything about steering results or the safety categories
PRIVATE_WORDS = re.compile(r"SAFETY|\bSAFE\b|safe-toggle|data-safe|scoreColor|steerWord|steers|\"score\"|"
                           r"\"safety\"|\bp\.[sf]\b")


CODE_WORDS = re.compile(r"SAFETY|\bSAFE\b|safe-toggle|data-safe|scoreColor|steerWord|\"score\"|\"safety\"")
ASSETS = Path(__file__).resolve().parent / "assets"
PAPER = ROOT / "paper"


def write_home(layers: list[dict], stats: list[dict]):
    """The home at /, its stylesheet and hero script, the icon, the Mono face, the screenshots and the report."""
    from public_site import THEME_EARLY, THEME_TOGGLE, brand

    site = OUT / "site"
    report = PAPER / "glimmer-scope-report.html"
    has_paper = report.exists()
    data = home.stack_data(layers)
    (site / "index.html").write_text(home.home_html(stats, data, brand().replace('href="./"', 'href="/"'), THEME_EARLY,
                                                    THEME_TOGGLE, has_paper))
    (site / "home.css").write_text(home.HOME_CSS)
    (site / "stack.js").write_text(home.STACK_JS)
    mark = (ROOT / "site" / "brand" / "mark.svg.html").read_text().strip()
    mark = mark.replace('<svg class="brand-mark"', '<svg xmlns="http://www.w3.org/2000/svg"', 1).replace(' aria-hidden="true"', "")
    mark = mark.replace(">", "><style>.k{fill:#0a0a0c}.b{fill:#dcdce0}.c{fill:#2fae62}"
                        "@media (prefers-color-scheme: dark){.k{fill:#f2f2f5}.b{fill:#3a3a44}.c{fill:#9fe8b8}}</style>", 1)
    (site / "favicon.svg").write_text(mark)
    if (ASSETS / "GeistMono-latin.woff2").exists():
        shutil.copy2(ASSETS / "GeistMono-latin.woff2", site / "brand" / "GeistMono-latin.woff2")
    (site / "img").mkdir(exist_ok=True)
    for f in ASSETS.glob("*.jpg"):
        shutil.copy2(f, site / "img" / f.name)
    if (ASSETS / "og.png").exists():
        shutil.copy2(ASSETS / "og.png", site / "og.png")
    if has_paper:  # the report, as a page at /paper/ and a PDF at the root
        (site / "paper").mkdir(exist_ok=True)
        for f in PAPER.iterdir():
            if f.is_file() and f.suffix in (".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".woff2", ".pdf"):
                shutil.copy2(f, site / "paper" / ("index.html" if f == report else f.name))
            elif f.is_dir() and f.name in ("img", "figures", "fig", "assets"):
                shutil.copytree(f, site / "paper" / f.name)
        pdf = PAPER / "glimmer-scope-report.pdf"
        if pdf.exists():
            shutil.copy2(pdf, site / "glimmer-scope-report.pdf")


def main():
    if not safety.SCREEN.pattern or safety.SCREEN.pattern == "(?!x)x":
        raise SystemExit("the private safety patterns didn't load: set GLIMMER_SAFETY_PATTERNS")
    if EXTRA is None:
        raise SystemExit(f"the extra atlas terms are missing: {EXTRA_FILE}")
    for d in ("r2", "d1", "site"):  # emptied in place, so a running `wrangler dev` keeps serving the folder
        (OUT / d).mkdir(parents=True, exist_ok=True)
        for child in (OUT / d).iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    index = json.loads((SRC / "atlas.json").read_text())
    layers, stats = [], []
    src = [json.loads((SRC / f).read_text()) for f in index["files"]]
    outs = [leave_out(m) for m in src]
    for li, m in enumerate(src):
        m, st = clean_layer(m, outs[li], outs[li + 1] if li + 1 < len(src) else set(), outs[li - 1] if li else set())
        layers.append(m)
        stats.append(st)
        (OUT / "r2" / f"L{m['layer']}.json").write_text(json.dumps(m, separators=(",", ":")))
    version = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((OUT / "r2").glob("L*.json")))).hexdigest()[:10]
    public_index = {"layers": index["layers"], "files": [f"/data/L{m['layer']}.json?v={version}" for m in layers],
                    "bytes": [(OUT / "r2" / f"L{m['layer']}.json").stat().st_size for m in layers],
                    "maps": [], "names": {}, "tokens": index["tokens"], "hn": {}}
    (OUT / "r2" / "atlas.json").write_text(json.dumps(public_index, separators=(",", ":")))
    write_sql(layers, OUT / "d1" / "features.sql")
    shutil.copytree(SRC / "brand", OUT / "site" / "brand")
    (OUT / "site" / "atlas").mkdir()
    (OUT / "site" / "atlas" / "index.html").write_text(page(stats).replace('"/data/atlas.json"', f'"/data/atlas.json?v={version}"'))
    write_home(layers, stats)
    # the light/dark switch only: the safe-browsing switch and its notes stay with the private site
    js = (SRC / "safe.js").read_text()
    (OUT / "site" / "theme.js").write_text(js[js.index("// Light or dark"):])
    css = (SRC / "theme.css").read_text()
    css = re.sub(r"^(\.safe-toggle|:root\[data-safe).*?\}\n", "", css, flags=re.M | re.S)
    (OUT / "site" / "theme.css").write_text(css)
    # nothing left in the output that the public must not see
    allowed_point, allowed_region = set(KEEP) | {"l"}, {"id", "name", "size", "x", "y"}
    for p in (OUT / "r2").glob("L*.json"):
        m = json.loads(p.read_text())
        assert set(m) <= {"layer", "points", "regions", "labeled"}, (p, set(m))
        for q in m["points"]:
            assert set(q) <= allowed_point, (p, set(q) - allowed_point)
            assert not unsafe(q.get("l", "")), (p, q["i"])
        for r in m["regions"]:
            assert set(r) <= allowed_region and not unsafe(r["name"]), (p, r)
    # every link lands on a feature that is shown on that neighbor layer: none points at one left out
    shown = [{q["i"] for q in m["points"]} for m in layers]
    for li, m in enumerate(layers):
        for q in m["points"]:
            for k, nb in (("u", li + 1), ("d", li - 1)):
                for j, _ in q.get(k, []):
                    assert 0 <= nb < len(layers) and j in shown[nb], (m["layer"], q["i"], k, j)
        assert not shown[li] & outs[li]
    for f in OUT.joinpath("site").rglob("*"):
        if f.suffix not in (".html", ".js", ".css", ".svg", ".json", ".txt"):
            continue
        text = f.read_text()
        # the report may talk about features that steer; the code words and private names stay out everywhere
        report = "paper" in f.relative_to(OUT / "site").parts
        hit = (CODE_WORDS if report else PRIVATE_WORDS).search(text) or (REPORT_WORDS if report else PAGE_WORDS).search(text)
        assert not hit, (str(f.relative_to(OUT / "site")), hit.group(0))
    total = sum(s["features"] for s in stats)
    left_out = sum(s["left_out"] for s in stats)
    print(f"{len(layers)} layers, {total:,} features shown, {left_out:,} left out")
    for s in stats:
        print(f"  layer {s['layer']}: {s['features']:,} features, {s['left_out']:,} left out, {s['regions']} regions")
    print("sizes:", {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted((OUT / "r2").iterdir())},
          "| sql", f"{(OUT / 'd1' / 'features.sql').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
