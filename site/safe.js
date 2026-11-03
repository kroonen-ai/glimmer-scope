// Safe browsing, on by default. Features about weapons, biological and chemical
// risk, cyberattacks, self-harm and other topics whose examples could be upsetting or misused keep
// their place on every map, but their labels, examples and region names stay hidden until it is
// turned off in the header. Nothing is removed from the data: the switch is for browsing and demos.
(() => {
  const KEY = "glimmer-safe-browsing", TEXT = "Hidden in safe browsing";
  let on = true;
  try { on = localStorage.getItem(KEY) !== "off"; } catch (e) { /* storage blocked: stay on */ }
  const listeners = [], swapped = [];
  // items marked h (at build time) show `text` in place of `keys` while safe browsing is on
  function swap(items, keys, text) {
    for (const it of items) {
      if (!it || !it.h) continue;
      for (const k of keys) {
        if (!(("_" + k) in it)) it["_" + k] = it[k];
        it[k] = on ? text : it["_" + k];
      }
    }
  }
  window.SAFE = {
    get on() { return on; },
    TEXT,
    hide: (it) => on && !!(it && it.h),
    apply(items, keys, text = TEXT) { swapped.push([items, keys, text]); swap(items, keys, text); },
    onChange(fn) { listeners.push(fn); },
  };
  function render() {
    document.documentElement.dataset.safe = on ? "on" : "off";
    for (const b of document.querySelectorAll(".safe-toggle")) {
      b.setAttribute("aria-pressed", String(on));
      const state = b.querySelector("b");
      if (state) state.textContent = on ? "on" : "off";
    }
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest(".safe-toggle");
    if (!b) return;
    on = !on;
    try { localStorage.setItem(KEY, on ? "on" : "off"); } catch (err) { /* not remembered */ }
    for (const [items, keys, text] of swapped) swap(items, keys, text);
    render();
    for (const fn of listeners) fn(on);
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render);
  else render();
})();

// Light or dark: follows the system until the switch picks one, then remembers it. Pages that draw
// on a canvas listen for "themechange" and redraw with the new colors.
(() => {
  const KEY = "glimmer-theme", root = document.documentElement;
  const systemDark = () => matchMedia("(prefers-color-scheme: dark)").matches;
  const current = () => root.dataset.theme || (systemDark() ? "dark" : "light");
  function label() {
    for (const b of document.querySelectorAll(".theme-toggle")) {
      const dark = current() === "dark";
      b.textContent = dark ? "☀" : "☾";
      b.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
      b.title = b.getAttribute("aria-label");
    }
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest(".theme-toggle");
    if (!b) return;
    root.dataset.theme = current() === "dark" ? "light" : "dark";
    try { localStorage.setItem(KEY, root.dataset.theme); } catch (err) { /* not remembered */ }
    label();
    window.dispatchEvent(new Event("themechange"));
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!root.dataset.theme) { label(); window.dispatchEvent(new Event("themechange")); }
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", label);
  else label();
})();

