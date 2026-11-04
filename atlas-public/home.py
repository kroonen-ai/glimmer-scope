"""The project home at scope.kroonen.ai: what Glimmer Scope is, what it found, and where to try it.

build.py calls home_html() with the cleaned public layers, so the features drawn in the hero come only
from what the atlas itself publishes. The hero's stack is drawn by stack.js from a few hundred sampled
points per layer, packed as bytes: the home never loads the full maps.
"""
import base64
import html
import random

SPACE = "https://huggingface.co/spaces/rob-x-ai/glimmer-scope"
CODE = "https://github.com/kroonen-ai/glimmer-scope"
SAES = "https://huggingface.co/kroonen-ai/glimmer-scope-saes"
VIDEO_ID = "6krusrrYQls"
VIDEO = f"https://youtu.be/{VIDEO_ID}"
SITE = "https://scope.kroonen.ai"
KINDS = ("concept", "syntax", "token", "format", "unclear")
PER_LAYER = 800
THREAD_START = (24, 24544)  # the Ocean feature, the one the video and the Space open with
# share of tested features that steer cleanly, by Muse Spark's grades (scripts/public_site.py, rechecked
# against the runs: Spark-graded tests only, as in the README)
CLEAN_SHARE = {12: 4.2, 15: 2.5, 18: 3.0, 21: 11.4, 24: 10.5, 27: 9.7, 30: 4.2, 33: 1.5, 36: 1.1}


def stack_data(layers: list[dict]) -> dict:
    xs = sorted(p["x"] for m in layers for p in m["points"])
    ys = sorted(p["y"] for m in layers for p in m["points"])
    q = lambda a, f: a[int(f * (len(a) - 1))]
    x0, x1, y0, y1 = q(xs, 0.004), q(xs, 0.996), q(ys, 0.004), q(ys, 0.996)
    byte = lambda v, a, b: max(0, min(255, round((v - a) / (b - a) * 255)))
    packed, counts = bytearray(), []
    for m in layers:
        pts = random.Random(m["layer"]).sample(m["points"], min(PER_LAYER, len(m["points"])))
        for p in pts:
            k = KINDS.index(p.get("k")) if p.get("k") in KINDS else 4
            packed += bytes((byte(p["x"], x0, x1), byte(p["y"], y0, y1), k))
        counts.append(len(pts))
    # the thread: the strongest partner on each next layer while it stays at 0.3 or more, like the atlas
    order = [m["layer"] for m in layers]
    by = [{p["i"]: p for p in m["points"]} for m in layers]
    li = order.index(THREAD_START[0])
    start = by[li][THREAD_START[1]]
    chain = [(li, start)]
    for key, step in (("u", 1), ("d", -1)):
        cur, i = start, li
        while 0 <= i + step < len(layers) and cur.get(key):
            j, c = max(cur[key], key=lambda t: t[1])
            if c < 0.3 or j not in by[i + step]:
                break
            i += step
            cur = by[i][j]
            chain.append((i, cur))
    chain.sort(key=lambda t: t[0])
    thread = [{"li": i, "layer": order[i], "id": p["i"], "label": p["l"],
               "x": byte(p["x"], x0, x1), "y": byte(p["y"], y0, y1)} for i, p in chain]
    return {"layers": order, "n": counts, "pts": base64.b64encode(bytes(packed)).decode(), "thread": thread,
            "origin": THREAD_START[0]}


def clean_share_chart() -> str:
    # one row per layer, the deepest on top like the stack; the middle band is the finding
    rows = sorted(CLEAN_SHARE.items(), reverse=True)
    top, row_h, bar_h, label_w, value_w, width = 8, 30, 12, 46, 52, 560
    plot_w = width - label_w - value_w
    vmax = 12.0
    h = top * 2 + row_h * len(rows)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-labelledby="chart-title chart-desc">',
             '<title id="chart-title">Share of tested features that steer cleanly, per layer</title>',
             '<desc id="chart-desc">' + ", ".join(f"layer {l}: {v}%" for l, v in rows) + "</desc>"]
    band_rows = [i for i, (l, _) in enumerate(rows) if 21 <= l <= 27]
    by0 = top + band_rows[0] * row_h + 2
    by1 = top + (band_rows[-1] + 1) * row_h - 2
    parts.append(f'<rect class="band" x="0" y="{by0}" width="{width}" height="{by1 - by0}" rx="8"/>')
    for g in (0, 4, 8, 12):
        x = label_w + plot_w * g / vmax
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{h - top}"/>')
    for i, (layer, v) in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        w = max(4.0, plot_w * v / vmax)
        mid = 21 <= layer <= 27
        parts.append(f'<g class="row{" mid" if mid else ""}"><title>Layer {layer}: {v}% steer cleanly</title>'
                     f'<rect class="hit" x="0" y="{top + i * row_h}" width="{width}" height="{row_h}"/>'
                     f'<text class="lab" x="0" y="{y + bar_h / 2 + 4:.1f}">L{layer}</text>'
                     f'<path class="bar" d="M{label_w},{y:.1f} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h{-(w - 4):.1f} z"/>'
                     f'<text class="val" x="{label_w + w + 8:.1f}" y="{y + bar_h / 2 + 4:.1f}">{v}%</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def home_html(stats: list[dict], data: dict, brand_html: str, theme_early: str, theme_toggle: str,
              has_paper: bool) -> str:
    import json

    shown = sum(s["features"] for s in stats)
    labeled = shown + sum(s["left_out"] for s in stats)
    first, last = data["thread"][0], data["thread"][-1]
    origin = next(t for t in data["thread"] if t["layer"] == data["origin"])
    caption = (f"Each sheet is one layer of Glimmer with {data['n'][0]:,} of its features, colored by kind. "
               f"The line is one feature's thread: <b>{esc(origin['label'])}</b> (layer {origin['layer']}, "
               f"#{origin['id']}), from {esc(first['label'].lower())} at layer {first['layer']} "
               f"to {esc(last['label'].lower())} at layer {last['layer']}.")
    ext = 'target="_blank" rel="noopener"'
    report_nav = '<a href="/paper/">Report</a>' if has_paper else ""
    report_link = ('<a class="text-link" href="/paper/">Read the report</a>'
                   f'<a class="text-link" href="https://doi.org/10.5281/zenodo.23952447" {ext}>DOI 10.5281/zenodo.23952447</a>') if has_paper else ""
    readout = [("Model", "Muse Glimmer 30B"), ("Sparse autoencoders", "9, layers 12 to 36"),
               ("Features per layer", "65,536"), ("Labeled by Muse Spark", f"{labeled:,}"),
               ("Steering tests", "170,000+"), ("Through all nine layers", "43%")]
    readout_html = "".join(f'<div class="cell"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>'
                           for k, v in readout)
    steps = [
        ("Train", "A TopK sparse autoencoder on each of nine layers of Glimmer's residual stream: 65,536 features, "
                  "64 active per token, 100M tokens of web and chat text.", "scope/train.py"),
        ("Catalog", "For every feature, where it fires hardest over 5M tokens.", "scope/interpret.py"),
        ("Label", "Muse Spark 1.3 names every feature that fires, from those examples.", "scope/label.py"),
        ("Test steering", "The feature is turned up while Glimmer answers neutral questions, and Spark judges "
                          "whether the answers move to its subject and still read well.", "scope/steerability.py"),
        ("Link the layers", "Features on neighboring layers are matched by how they fire together, so a concept "
                            "can be followed from layer 12 to 36.", "scope/crosslayer.py"),
        ("Map and demo", "Maps of every layer, the nine-layer atlas, and a live demo on the full 30B model.",
         "scripts/atlas_nine.py"),
    ]
    steps_html = "".join(f'<li><span class="n">{i}</span><h3>{esc(t)}</h3><p>{esc(d)}</p>'
                         f'<a class="code" href="{CODE}/blob/main/{f}" {ext}>{esc(f)}</a></li>'
                         for i, (t, d, f) in enumerate(steps, 1))

    def shot(name: str, alt: str) -> str:
        return (f'<img class="only-light" src="/img/{name}-light.jpg" alt="{esc(alt)}" width="1280" height="800" loading="lazy" decoding="async">'
                f'<img class="only-dark" src="/img/{name}-dark.jpg" alt="{esc(alt)}" width="1280" height="800" loading="lazy" decoding="async">')

    desc = ("Sparse autoencoders on nine layers of Muse Glimmer 30B: every feature labeled, tested for steering, "
            "and followed from layer to layer. Try it live, or explore the nine-layer atlas.")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glimmer Scope</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:type" content="website">
<meta property="og:title" content="Glimmer Scope: see inside Muse Glimmer 30B">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:url" content="{SITE}/">
<meta property="og:image" content="{SITE}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Glimmer Scope: see inside Muse Glimmer 30B">
<meta name="twitter:description" content="{esc(desc)}">
<meta name="twitter:image" content="{SITE}/og.png">
<meta name="theme-color" content="#0b1118" media="(prefers-color-scheme: dark)">
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preload" href="/brand/Geist-Variable.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="/theme.css">
<link rel="stylesheet" href="/home.css">
{theme_early}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <div class="wrap top-in">
    {brand_html}
    <nav class="nav" aria-label="Glimmer Scope">
      <a href="/atlas/">Atlas</a><a href="{SPACE}" {ext}>Live demo</a><a href="#watch">Video</a>{report_nav}<a href="{CODE}" {ext}>Code</a><a href="{SAES}" {ext}>SAEs</a>
    </nav>
    <span class="toggles">{theme_toggle}</span>
  </div>
</header>
<main id="main">
<section class="hero">
  <div class="wrap hero-in">
    <div class="hero-copy">
      <p class="eyebrow">Muse Glimmer 30B · 9 SAEs · layers 12 to 36</p>
      <h1>See inside Muse&nbsp;Glimmer&nbsp;30B.</h1>
      <p class="lede">Sparse autoencoders on nine of its layers split what the model computes into features you can
        read. Every feature that fires is labeled, tested for steering, and followed from layer to layer.</p>
      <div class="cta">
        <a class="btn primary" href="{SPACE}" {ext}>Try it live <span aria-hidden="true">↗</span></a>
        <a class="btn" href="/atlas/">Explore the atlas</a>
      </div>
      <p class="more">{report_link}<a class="text-link" href="{CODE}" {ext}>Code on GitHub</a><a class="text-link" href="{SAES}" {ext}>The 9 SAEs</a><a class="text-link" href="#watch">Watch the video</a></p>
    </div>
    <figure class="hero-fig">
      <canvas id="stack" role="img" aria-label="Nine layers of Muse Glimmer 30B stacked, deepest at the top, with sampled features on each and one feature's thread running through them"></canvas>
      <figcaption>{caption}</figcaption>
    </figure>
  </div>
</section>

<section class="readout" aria-label="The work in numbers">
  <div class="wrap readout-in">{readout_html}</div>
</section>

<section class="sec" id="watch">
  <div class="wrap two watch">
    <div>
      <p class="eyebrow">Watch</p>
      <h2>Two minutes inside Glimmer.</h2>
      <p>The atlas, Muse Spark naming a feature live through the Meta Model API, and Glimmer steered toward the ocean
        and a pirate captain, next to the answer it gives without the dials.</p>
      <p class="more"><a class="text-link" href="{VIDEO}" {ext}>Watch on YouTube</a></p>
    </div>
    <div class="video" id="video" data-id="{VIDEO_ID}">
      <button class="video-play" type="button" aria-label="Play the video: Glimmer Scope, steering Muse Glimmer 30B from the inside">
        <img src="/img/video-poster.jpg" alt="" width="1280" height="720" loading="lazy">
        <span class="play" aria-hidden="true"></span>
        <span class="dur" aria-hidden="true">2:14</span>
      </button>
    </div>
  </div>
</section>

<section class="sec" id="found">
  <div class="wrap two">
    <div>
      <p class="eyebrow">What it found</p>
      <h2>Steering peaks in the middle of the network.</h2>
      <p>About 10 to 11% of tested features steer cleanly at layers 21, 24 and 27, against 1 to 4% near the start
        and the end, by Muse Spark's grades.</p>
      <h3>There is a stable backbone.</h3>
      <p>43% of features belong to a thread that runs through all nine layers, and 31% live on one layer only.
        Shallow layers keep their concepts (62% of layer 12 runs all the way), deep ones add their own (49% of
        layer 36 exists there only).</p>
      <h3>Characters live at every depth.</h3>
      <p>Archetypes from pirate captain to gaslighter show up as steering directions on every layer, and the most
        play cleanly at layer 24 (19 of 28).</p>
    </div>
    <figure class="chart-fig">
      <figcaption><b>Share of tested features that steer cleanly</b><span>per layer, deepest on top, Muse Spark's grades</span></figcaption>
      {clean_share_chart()}
    </figure>
  </div>
</section>

<section class="sec" id="try">
  <div class="wrap">
    <p class="eyebrow">Try it</p>
    <h2>Two ways in.</h2>
    <div class="cards">
      <a class="card" href="{SPACE}" {ext}>
        <span class="shot">{shot("space", "The live demo: one question answered by Glimmer as it is and by Glimmer with a few features turned up")}</span>
        <span class="card-body"><b>Steer Glimmer live <span aria-hidden="true">↗</span></b>
          <span>Ask a question and get two answers from the real 30B model, one as it is and one with a few
            features turned up. Runs on a Hugging Face ZeroGPU Space.</span></span>
      </a>
      <a class="card" href="/atlas/">
        <span class="shot">{shot("atlas", "The atlas: nine layers stacked, with one feature's thread and its lineage")}</span>
        <span class="card-body"><b>Walk the atlas <span aria-hidden="true">→</span></b>
          <span>All nine layers with {shown:,} labeled features. Search a subject, click a feature, and follow
            its thread from layer 12 to 36.</span></span>
      </a>
    </div>
  </div>
</section>

<section class="sec" id="how">
  <div class="wrap">
    <p class="eyebrow">How it works</p>
    <h2>From open weights to features you can read.</h2>
    <ol class="steps">{steps_html}</ol>
  </div>
</section>

<section class="sec" id="filtering">
  <div class="wrap two narrow">
    <div>
      <p class="eyebrow">Safety</p>
      <h2>Some findings could be upsetting.</h2>
    </div>
    <div>
      <p>So we filter them. The SAEs on Hugging Face carry a content warning, and this atlas leaves out the
        features on upsetting topics entirely, with their place on the map and their links to other layers.
        No decoder vectors, activations or steering strengths are published here.</p>
    </div>
  </div>
</section>

<section class="sec built">
  <div class="wrap two narrow">
    <div>
      <p class="eyebrow">Built during the hackathon</p>
      <h2>October 28 to November 4, 2026.</h2>
    </div>
    <div>
      <p>Built by Kroonen AI for the Meta Global AI Developer Hackathon, with Muse Glimmer 30B (open weights) and
        Muse Spark 1.3 through the Meta Model API. Compute from Modal and AWS. Every step is in the
        <a href="{CODE}/commits/main" {ext}>commit history</a>.</p>
    </div>
  </div>
</section>
</main>

<footer class="foot">
  <div class="wrap foot-in">
    <span>Glimmer Scope by Kroonen AI · Apache 2.0</span>
    <nav aria-label="Everything Glimmer Scope">
      <a href="/atlas/">Atlas</a><a href="{SPACE}" {ext}>Live demo</a><a href="#watch">Video</a>{report_nav}<a href="{CODE}" {ext}>Code</a><a href="{SAES}" {ext}>SAEs</a><a href="https://huggingface.co/collections/kroonen-ai/glimmer-scope" {ext}>HF collection</a><a href="https://kroonen.ai" {ext}>kroonen.ai</a>
    </nav>
  </div>
</footer>
<script type="application/json" id="stack-data">{json.dumps(data, separators=(",", ":"))}</script>
<script src="/theme.js"></script>
<script src="/stack.js"></script>
<script>
  const qs = new URLSearchParams(location.search);
  if (qs.has("og")) document.documentElement.classList.add("og");
  glimmerStack(document.getElementById("stack"), JSON.parse(document.getElementById("stack-data").textContent),
               {{ still: qs.has("still") || qs.has("og") }});
  // the YouTube player loads only when someone presses play
  const video = document.getElementById("video");
  video.querySelector("button").addEventListener("click", () => {{
    const f = document.createElement("iframe");
    f.src = `https://www.youtube-nocookie.com/embed/${{video.dataset.id}}?autoplay=1&rel=0`;
    f.title = "Glimmer Scope: steering Muse Glimmer 30B from the inside";
    f.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
    f.allowFullscreen = true;
    video.replaceChildren(f);
  }});
</script>
</body>
</html>
"""


HOME_CSS = r"""/* scope.kroonen.ai home. Colors come from theme.css; this adds the type scale, layout and the hero. */
@font-face { font-family: "Geist Mono"; src: url("/brand/GeistMono-latin.woff2") format("woff2"); font-weight: 400 600; font-display: swap; }
:root { --mono: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace; --gutter: 16px; --max: 1180px; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 17px/1.6 var(--font); overflow-x: hidden; }
a { color: inherit; }
img { max-width: 100%; display: block; }
.wrap { max-width: var(--max); margin: 0 auto; padding: 0 var(--gutter); }
@media (min-width: 720px) { :root { --gutter: 32px; } }
.skip { position: absolute; left: -999px; top: 8px; background: var(--panel); padding: 8px 12px; border-radius: 8px; z-index: 10; }
.skip:focus { left: 8px; }
.nb { white-space: nowrap; }

/* header */
.top { position: sticky; top: 0; z-index: 5; background: color-mix(in srgb, var(--bg) 86%, transparent);
  backdrop-filter: saturate(1.4) blur(12px); -webkit-backdrop-filter: saturate(1.4) blur(12px); border-bottom: 1px solid var(--line); }
.top-in { display: flex; align-items: center; gap: 16px; height: 60px; }
.top .brand { flex: none; }
.nav { display: flex; gap: 2px; margin-left: auto; overflow-x: auto; scrollbar-width: none; }
.nav::-webkit-scrollbar { display: none; }
.nav a { text-decoration: none; font-size: 14px; font-weight: 500; padding: 6px 11px; border-radius: 999px; white-space: nowrap; }
.nav a:hover { background: var(--surface); }
.toggles { flex: none; }
.theme-toggle { width: 34px; height: 34px; border-radius: 999px; border: 1px solid var(--line); background: var(--panel);
  color: var(--ink); cursor: pointer; font-size: 15px; display: grid; place-items: center; }
@media (max-width: 760px) {
  .top-in { height: auto; padding-top: 10px; padding-bottom: 8px; flex-wrap: wrap; row-gap: 6px; }
  .nav { order: 3; width: 100%; margin-left: -10px; }
  .toggles { margin-left: auto; }
}

/* the hero: copy on the left, the stack on a dot graticule (the brand mark's grid) on the right */
.hero { position: relative; overflow: hidden; border-bottom: 1px solid var(--line); }
.hero::before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: radial-gradient(circle, color-mix(in srgb, var(--muted) 34%, transparent) 1px, transparent 1.4px);
  background-size: 24px 24px; background-position: 12px 12px;
  -webkit-mask-image: radial-gradient(ellipse 70% 80% at 70% 50%, #000 20%, transparent 75%);
          mask-image: radial-gradient(ellipse 70% 80% at 70% 50%, #000 20%, transparent 75%); opacity: .55; }
.hero-in { position: relative; display: grid; grid-template-columns: minmax(0, 6fr) minmax(0, 7fr); gap: 24px;
  align-items: center; min-height: min(640px, calc(100vh - 60px)); padding-top: 40px; padding-bottom: 40px; }
.eyebrow { font: 500 12px/1.4 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 14px; }
h1 { font-size: clamp(38px, 4.4vw, 56px); line-height: 1; letter-spacing: -0.045em; font-weight: 680; margin: 0 0 22px; text-wrap: balance; }
.lede { font-size: clamp(17px, 1.6vw, 20px); line-height: 1.55; color: var(--muted); max-width: 32em; margin: 0 0 28px; }
.cta { display: flex; flex-wrap: wrap; gap: 10px; }
.btn { display: inline-flex; align-items: center; gap: 8px; height: 46px; padding: 0 20px; border-radius: 999px;
  border: 1px solid var(--line); background: var(--panel); text-decoration: none; font-weight: 600; font-size: 16px;
  transition: border-color .15s ease, background .15s ease, transform .15s ease; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
.btn.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
.btn:active { transform: translateY(1px); }
.more { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 20px 0 0; font-size: 15px; }
.text-link { color: var(--muted); text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 4px; }
.text-link:hover { color: var(--ink); text-decoration-color: var(--accent); }
.hero-fig { margin: 0; }
#stack { width: 100%; height: min(560px, 62vh); display: block; touch-action: pan-y; }
.hero-fig figcaption { font-size: 13px; line-height: 1.5; color: var(--muted); max-width: 46em; margin: 6px 0 0 34px; }
.hero-fig figcaption b { color: var(--ink); font-weight: 600; }
@media (max-width: 900px) {
  .hero-in { grid-template-columns: 1fr; min-height: 0; padding-top: 32px; gap: 8px; }
  #stack { height: 380px; }
  .hero-fig figcaption { margin-left: 0; }
}

/* the readout: an instrument's status line */
.readout { border-bottom: 1px solid var(--line); background: var(--surface); }
.readout-in { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); }
.readout .cell { padding: 18px 16px 18px 0; display: flex; flex-direction: column; justify-content: space-between; gap: 6px; }
.readout .cell + .cell { padding-left: 16px; border-left: 1px solid var(--line); }
.readout .k { font: 500 11px/1.3 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.readout .v { font-size: 19px; font-weight: 640; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
@media (max-width: 980px) {
  .readout-in { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .readout .cell { padding: 14px 12px 14px 0; border-top: 1px solid var(--line); }
  .readout .cell:nth-child(-n+2) { border-top: 0; }
  .readout .cell + .cell { border-left: 0; padding-left: 0; }
  .readout .cell:nth-child(2n) { border-left: 1px solid var(--line); padding-left: 14px; }
}

/* sections */
.sec { padding: 88px 0; border-bottom: 1px solid var(--line); }
@media (max-width: 720px) { .sec { padding: 56px 0; } }
h2 { font-size: clamp(30px, 3.6vw, 44px); line-height: 1.06; letter-spacing: -0.035em; font-weight: 660; margin: 0 0 20px; max-width: 18em; }
h3 { font-size: 19px; letter-spacing: -0.015em; font-weight: 640; margin: 28px 0 6px; }
.sec p { color: var(--muted); margin: 0 0 12px; max-width: 36em; }
.sec p a { color: var(--ink); text-underline-offset: 3px; }
.two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 56px; align-items: start; }
.two.narrow { grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); }
@media (max-width: 860px) { .two, .two.narrow { grid-template-columns: 1fr; gap: 20px; } }

/* the depth chart */
.chart-fig { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 22px 22px 14px; }
.chart-fig figcaption { display: flex; flex-direction: column; gap: 2px; margin-bottom: 14px; }
.chart-fig figcaption b { font-size: 15px; font-weight: 640; }
.chart-fig figcaption span { font: 400 12px/1.4 var(--mono); color: var(--muted); }
.chart { width: 100%; height: auto; display: block; overflow: visible; }
.chart .band { fill: var(--accent-tint); }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .bar { fill: color-mix(in srgb, var(--muted) 42%, var(--panel)); transition: fill .15s ease; }
.chart .mid .bar { fill: var(--accent); }
.chart .lab { font: 500 12px var(--mono); fill: var(--muted); }
.chart .mid .lab { fill: var(--accent-ink); }
.chart .val { font: 600 12px var(--font); fill: var(--ink); font-variant-numeric: tabular-nums; }
.chart .hit { fill: transparent; }
.chart .row:hover .bar { fill: var(--accent-hover); }

/* the video: a poster first, the player only after a click */
.watch { align-items: center; }
.video { position: relative; aspect-ratio: 16 / 9; border-radius: 20px; overflow: hidden; border: 1px solid var(--line);
  background: #0b1118; box-shadow: var(--shadow); }
.video iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }
.video-play { position: absolute; inset: 0; padding: 0; border: 0; background: none; cursor: pointer; display: block; }
.video-play img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s ease; }
.video-play:hover img { transform: scale(1.015); }
.play { position: absolute; left: 80%; top: 52%; width: 76px; height: 76px; margin: -38px 0 0 -38px; border-radius: 50%;
  background: var(--accent); box-shadow: 0 10px 34px rgba(0, 0, 0, 0.35); transition: transform .2s ease; }
@media (max-width: 560px) { .play { width: 58px; height: 58px; margin: -29px 0 0 -29px; } .play::after { left: 23px !important; top: 17px !important; border-width: 12px 0 12px 18px !important; } }
.play::after { content: ""; position: absolute; left: 31px; top: 24px; border-style: solid; border-width: 14px 0 14px 22px;
  border-color: transparent transparent transparent #fff; }
.video-play:hover .play, .video-play:focus-visible .play { transform: scale(1.08); }
.dur { position: absolute; right: 14px; bottom: 12px; padding: 3px 8px; border-radius: 8px; font: 500 12px var(--mono);
  color: #fff; background: rgba(0, 0, 0, 0.6); }
@media (prefers-reduced-motion: reduce) { .video-play img, .play { transition: none; } .video-play:hover img { transform: none; } }

/* cards */
.cards { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin-top: 28px; }
@media (max-width: 760px) { .cards { grid-template-columns: 1fr; } }
.card { display: flex; flex-direction: column; text-decoration: none; border: 1px solid var(--line); border-radius: 20px;
  overflow: hidden; background: var(--panel); transition: border-color .15s ease, box-shadow .2s ease, transform .2s ease; }
.card:hover { border-color: color-mix(in srgb, var(--accent) 55%, var(--line)); box-shadow: var(--shadow); transform: translateY(-2px); }
.shot { display: block; aspect-ratio: 16 / 10; overflow: hidden; border-bottom: 1px solid var(--line); background: var(--surface); }
.shot img { width: 100%; height: 100%; object-fit: cover; object-position: top center; }
.card-body { display: flex; flex-direction: column; gap: 6px; padding: 18px 20px 22px; }
.card-body b { font-size: 19px; letter-spacing: -0.015em; font-weight: 640; }
.card-body span { color: var(--muted); font-size: 15px; line-height: 1.55; }
@media (prefers-reduced-motion: reduce) { .card, .btn { transition: none; } .card:hover { transform: none; } }

/* the pipeline: a real sequence, so it is numbered */
.steps { list-style: none; padding: 0; margin: 32px 0 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  border-top: 1px solid var(--line); }
.steps li { padding: 22px 24px 26px 0; border-bottom: 1px solid var(--line); display: flex; flex-direction: column; }
.steps li:not(:nth-child(3n + 1)) { padding-left: 24px; border-left: 1px solid var(--line); }
.steps .n { font: 500 12px var(--mono); color: var(--accent-ink); }
.steps h3 { margin: 8px 0 6px; }
.steps p { font-size: 15px; line-height: 1.55; margin: 0 0 12px; }
.steps .code { margin-top: auto; font: 400 12px var(--mono); color: var(--muted); text-decoration: none; }
.steps .code:hover { color: var(--accent-ink); text-decoration: underline; }
@media (max-width: 860px) {
  .steps { grid-template-columns: 1fr; }
  .steps li, .steps li:not(:nth-child(3n + 1)) { padding: 18px 0 20px; border-left: 0; }
}

.built { background: var(--surface); }
.foot { padding: 28px 0 40px; font-size: 14px; color: var(--muted); }
.foot-in { display: flex; flex-wrap: wrap; gap: 10px 24px; justify-content: space-between; align-items: center; }
.foot nav { display: flex; flex-wrap: wrap; gap: 4px 16px; }
.foot a { text-decoration: none; }
.foot a:hover { color: var(--ink); text-decoration: underline; }

/* light and dark screenshots, following the switch as well as the system */
.only-dark { display: none; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .only-dark { display: block; } :root:not([data-theme="light"]) .only-light { display: none; } }
:root[data-theme="dark"] .only-dark { display: block; } :root[data-theme="dark"] .only-light { display: none; }

/* ?og: the share card, 1200 x 630 */
html.og .top, html.og .readout, html.og .sec, html.og .foot, html.og .more, html.og .cta, html.og .hero-fig figcaption { display: none; }
html.og .hero-in { min-height: 630px; height: 630px; padding: 48px 40px 48px 64px; max-width: none; grid-template-columns: 520px minmax(0, 1fr); gap: 16px; }
html.og h1 { font-size: 62px; }
html.og .lede { font-size: 19px; }
html.og #stack { height: 560px; }
"""


STACK_JS = r"""// The home's hero: nine of Glimmer's layers stacked, deepest at the top, each with a sample of its real
// features colored by kind, and one feature's thread drawn through them. Data comes packed from the page
// (3 bytes per feature: x, y, kind), so the home never loads the full maps.
(() => {
  const KIND = ["#0064e0", "#8b5cf6", "#14a38b", "#f29a38", "#9aa6b4"];
  const KIND_DARK = ["#3d8bff", "#a685ff", "#2cc4a6", "#f5a95a", "#7d8a99"];
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const ease = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

  window.glimmerStack = function (canvas, data, opt = {}) {
    const ctx = canvas.getContext("2d");
    const raw = atob(data.pts), bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const NL = data.layers.length, starts = [];
    data.n.reduce((o, n) => (starts.push(o), o + n), 0);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)");
    const still = () => opt.still || reduce.matches;
    let W = 0, H = 0, dpr = 1, col = {}, dark = false, visible = true, raf = 0;
    let px = 0, py = 0, tpx = 0, tpy = 0;
    const t0 = performance.now();

    function theme() {
      const ink = css("--ink");
      dark = parseInt(ink.slice(1, 3), 16) > 128;
      col = { line: css("--line"), plane: css("--plane"), ink, muted: css("--muted"), accent: css("--accent"),
              accentInk: css("--accent-ink"), panel: css("--panel"), kinds: dark ? KIND_DARK : KIND };
    }
    function size() {
      const r = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = r.width; H = r.height;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }

    function frame(now) {
      const t = still() ? 1e9 : now - t0;
      px += (tpx - px) * 0.06; py += (tpy - py) * 0.06;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const labels = W > 520;
      const left = 34, right = labels ? Math.min(290, W * 0.4) : 8;
      const flat = 0.25 * (1 + 0.18 * py), gapK = 0.42;
      const R = Math.max(40, Math.min((W - left - right) / (2 * Math.SQRT2), (H - 28) / (2 * Math.SQRT2 * flat + (NL - 1) * gapK)));
      const spread = 0.16 + 0.84 * ease((t - 120) / 1400);
      const gap = R * gapK * spread;
      const cx = left + (W - left - right) / 2;
      const total = (NL - 1) * gap;
      const bottom = H / 2 + total / 2;
      const th = Math.PI / 4 + (still() ? 0 : 0.09 * Math.sin(t / 5200)) + 0.16 * px;
      const c = Math.cos(th), s = Math.sin(th);
      const proj = (x, y, i) => {
        const a = x / 127.5 - 1, b = y / 127.5 - 1;
        return [cx + (a * c - b * s) * R, bottom - i * gap + (a * s + b * c) * R * flat];
      };
      const dot = Math.max(1.1, R / 95);
      const show = ease((t - 60) / 900);
      for (let i = 0; i < NL; i++) {
        const cy = bottom - i * gap;
        // the sheet
        ctx.beginPath();
        for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          const X = cx + (a * c - b * s) * R, Y = cy + (a * s + b * c) * R * flat;
          ctx.lineTo(X, Y);
        }
        ctx.closePath();
        ctx.fillStyle = dark ? "rgba(19,29,40,0.72)" : "rgba(255,255,255,0.74)";
        ctx.fill();
        ctx.fillStyle = col.plane; ctx.fill();
        ctx.strokeStyle = data.layers[i] === data.origin ? col.accent : col.line;
        ctx.lineWidth = data.layers[i] === data.origin ? 1.4 : 1;
        ctx.stroke();
        // its features, gray ones first
        ctx.globalAlpha = 0.9 * show;
        for (const k of [4, 3, 2, 1, 0]) {
          ctx.fillStyle = col.kinds[k];
          for (let j = starts[i], e = starts[i] + data.n[i]; j < e; j++) {
            if (bytes[j * 3 + 2] !== k) continue;
            const [X, Y] = proj(bytes[j * 3], bytes[j * 3 + 1], i);
            ctx.fillRect(X - dot / 2, Y - dot / 2, dot, dot);
          }
        }
        ctx.globalAlpha = 1;
        // the layer's name on the left
        ctx.font = `500 11px "Geist Mono", ui-monospace, monospace`;
        ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillStyle = data.layers[i] === data.origin ? col.accentInk : col.muted;
        ctx.fillText(`L${data.layers[i]}`, cx - R * Math.SQRT2 - 10, cy);
      }
      // the thread, drawn upward once the sheets have spread
      const nodes = data.thread.map((n) => proj(n.x, n.y, n.li));
      const draw = ease((t - 900) / 1300);
      if (nodes.length > 1 && draw > 0) {
        const segs = nodes.length - 1, upto = draw * segs;
        ctx.save();
        ctx.strokeStyle = col.accent; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
        if (dark) { ctx.shadowColor = col.accent; ctx.shadowBlur = 10; }
        ctx.beginPath(); ctx.moveTo(nodes[0][0], nodes[0][1]);
        for (let k = 1; k <= segs; k++) {
          const f = Math.min(1, upto - (k - 1));
          if (f <= 0) break;
          const [x0, y0] = nodes[k - 1], [x1, y1] = nodes[k];
          ctx.lineTo(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f);
        }
        ctx.stroke(); ctx.restore();
        nodes.forEach(([X, Y], k) => {
          if (k > upto + 0.001) return;
          const origin = data.thread[k].layer === data.origin;
          ctx.beginPath(); ctx.arc(X, Y, origin ? 5.5 : 3.6, 0, 2 * Math.PI);
          ctx.fillStyle = col.accent; ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = col.panel; ctx.stroke();
          if (origin) { ctx.beginPath(); ctx.arc(X, Y, 10, 0, 2 * Math.PI); ctx.lineWidth = 1.2; ctx.strokeStyle = col.accent; ctx.stroke(); }
        });
        // a pulse running up the thread
        if (!still() && draw >= 1) {
          const u = ((t - 2200) / 3800) % 1, at = u * segs, k = Math.min(segs - 1, Math.floor(at)), f = at - k;
          const X = nodes[k][0] + (nodes[k + 1][0] - nodes[k][0]) * f, Y = nodes[k][1] + (nodes[k + 1][1] - nodes[k][1]) * f;
          const g = ctx.createRadialGradient(X, Y, 0, X, Y, 12);
          g.addColorStop(0, col.accent); g.addColorStop(1, "rgba(0,0,0,0)");
          ctx.globalAlpha = 0.55; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X, Y, 12, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1;
        }
        // what the thread is on each layer, on the right
        if (labels) {
          ctx.font = `400 12px "Geist", system-ui, sans-serif`; ctx.textAlign = "left";
          const lx = cx + R * Math.SQRT2 + 26, maxw = W - lx - 4;
          data.thread.forEach((n, k) => {
            if (k > upto + 0.001) return;
            const y = bottom - n.li * gap, origin = n.layer === data.origin;
            ctx.font = `${origin ? 600 : 400} 12px "Geist", system-ui, sans-serif`;
            ctx.fillStyle = origin ? col.ink : col.muted;
            let text = n.label;
            while (text.length > 8 && ctx.measureText(text).width > maxw) text = text.slice(0, -2);
            if (text !== n.label) text = text.replace(/\s+\S*$/, "") + "…";
            ctx.globalAlpha = ease((upto - k + 0.6) * 1.6);
            ctx.fillText(text, lx, y);
            ctx.globalAlpha = 0.5; ctx.strokeStyle = col.line; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(nodes[k][0] + 8, nodes[k][1]); ctx.lineTo(lx - 6, y); ctx.stroke();
            ctx.globalAlpha = 1;
          });
        }
      }
      if (!still() && visible) raf = requestAnimationFrame(frame);
      else raf = 0;
    }
    const kick = () => { if (!raf) raf = requestAnimationFrame(frame); };
    theme(); size();
    window.addEventListener("resize", () => { size(); kick(); });
    window.addEventListener("themechange", () => { theme(); kick(); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { theme(); kick(); });
    reduce.addEventListener("change", kick);
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      tpx = (e.clientX - r.left) / r.width * 2 - 1; tpy = (e.clientY - r.top) / r.height * 2 - 1;
      kick();
    });
    canvas.addEventListener("pointerleave", () => { tpx = 0; tpy = 0; kick(); });
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; if (visible) kick(); }).observe(canvas);
    document.addEventListener("visibilitychange", () => { visible = !document.hidden; if (visible) kick(); });
    (document.fonts ? document.fonts.ready : Promise.resolve()).then(kick);
    kick();
  };
})();
"""
