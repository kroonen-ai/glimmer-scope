// The public Glimmer Scope atlas: the page from static assets, the nine layer maps from R2, and search and
// feature lookups from D1. Everything it can return was cleaned by build.py first: positions, kinds, regions,
// links between layers and public labels, nothing about vectors or steering.

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "public, max-age=300" };
const reply = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });

async function mapFile(env, key) {
  if (!/^(atlas|L\d{2})\.json$/.test(key)) return new Response("Not found", { status: 404 });
  const obj = await env.ATLAS.get(key);
  if (!obj) return new Response("Not found", { status: 404 });
  return new Response(obj.body, {
    // the layer files are versioned in their URLs; the small index isn't, so it stays fresh
    headers: { "content-type": "application/json", "cache-control": `public, max-age=${key === "atlas.json" ? 300 : 86400}`,
      etag: obj.httpEtag },
  });
}

// words only, each as a prefix: "ocean under" finds "marine ocean and underwater environments"
function ftsQuery(q) {
  const words = (q.toLowerCase().match(/[a-z0-9]+/g) || []).slice(0, 6);
  return words.length ? words.map((w) => `"${w}"*`).join(" ") : null;
}

async function search(env, url) {
  const match = ftsQuery(url.searchParams.get("q") || "");
  if (!match) return reply({ results: [] });
  const layer = Number(url.searchParams.get("layer")) || null;
  const sql = "SELECT layer, id, label FROM features_fts WHERE features_fts MATCH ?1" +
    (layer ? " AND layer = ?2" : "") + " ORDER BY rank LIMIT 40";
  const stmt = env.DB.prepare(sql);
  const { results } = await (layer ? stmt.bind(match, layer) : stmt.bind(match)).all();
  return reply({ results });
}

async function feature(env, layer, id) {
  const row = await env.DB.prepare("SELECT f.layer, f.id, f.label, f.kind, f.up, f.down, r.name AS region " +
    "FROM features f LEFT JOIN regions r ON r.layer = f.layer AND r.id = f.region WHERE f.layer = ?1 AND f.id = ?2")
    .bind(layer, id).first();
  if (!row) return reply({ error: "No feature with that layer and id." }, 404);
  const parse = (s) => (s ? JSON.parse(s) : []);
  return reply({ layer: row.layer, id: row.id, label: row.label, kind: row.kind, region: row.region,
    up: parse(row.up), down: parse(row.down) });
}

async function stats(env) {
  const { results } = await env.DB.prepare("SELECT layer, COUNT(*) AS features FROM features GROUP BY layer ORDER BY layer").all();
  return reply({ layers: results, features: results.reduce((n, r) => n + r.features, 0) });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") return new Response("Method not allowed", { status: 405 });
    try {
      if (url.pathname.startsWith("/data/")) return await mapFile(env, url.pathname.slice(6));
      if (url.pathname === "/api/search") return await search(env, url);
      if (url.pathname === "/api/stats") return await stats(env);
      const m = url.pathname.match(/^\/api\/feature\/(\d{1,2})\/(\d{1,6})$/);
      if (m) return await feature(env, Number(m[1]), Number(m[2]));
      if (url.pathname.startsWith("/api/")) return reply({ error: "Unknown endpoint." }, 404);
    } catch (e) {
      return reply({ error: "The atlas couldn't answer that just now." }, 500);
    }
    return env.ASSETS.fetch(request);
  },
};
