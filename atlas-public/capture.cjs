// Screenshots for the home's "Try it" cards and its share image, saved into assets/ (committed, so a build
// needs no browser). Run with the atlas served locally (wrangler dev on :8799) and the Space up:
//   node atlas-public/capture.cjs [space] [atlas] [og]
const path = require("path");
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
const OUT = path.join(__dirname, "assets");
const LOCAL = process.env.LOCAL || "http://127.0.0.1:8799";
const SPACE = "https://rob-x-ai-glimmer-scope.hf.space/";
const want = new Set(process.argv.slice(2).length ? process.argv.slice(2) : ["space", "atlas", "og"]);

(async () => {
  const b = await chromium.launch();
  for (const theme of ["light", "dark"]) {
    if (want.has("space")) {
      const p = await b.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1, colorScheme: theme });
      await p.goto(`${SPACE}?__theme=${theme}`, { waitUntil: "networkidle", timeout: 180000 });
      await p.waitForTimeout(6000);
      await p.screenshot({ path: path.join(OUT, `space-${theme}.jpg`), type: "jpeg", quality: 82 });
      await p.close();
    }
    if (want.has("atlas")) {
      const p = await b.newPage({ viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
      await p.addInitScript((t) => { try { localStorage.setItem("glimmer-theme", t); } catch (e) {} }, theme);
      await p.goto(`${LOCAL}/atlas/`, { waitUntil: "load", timeout: 180000 });
      await p.waitForSelector("#loading", { state: "detached", timeout: 180000 });
      await p.click("#q"); await p.keyboard.type("marine ocean and underwater environments"); await p.waitForTimeout(1500);
      const n = await p.$$eval("#results button", (bs) => bs.findIndex((x) => /layer 24 · #24544/.test(x.textContent)));
      await p.click(`#results button[data-n="${Math.max(0, n)}"]`); await p.waitForTimeout(1200);
      await p.click(".card .stackit"); await p.waitForTimeout(2500);
      await p.screenshot({ path: path.join(OUT, `atlas-${theme}.jpg`), type: "jpeg", quality: 82 });
      await p.close();
    }
  }
  if (want.has("og")) {
    const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
    await p.addInitScript(() => { try { localStorage.setItem("glimmer-theme", "dark"); } catch (e) {} });
    await p.goto(`${LOCAL}/?og`, { waitUntil: "load" });
    await p.waitForTimeout(1200);
    await p.screenshot({ path: path.join(OUT, "og.png"), clip: { x: 0, y: 0, width: 1200, height: 630 } });
    await p.close();
  }
  await b.close();
})();
