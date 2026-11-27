// Precompute EmbeddingGemma 2 embeddings for Preference Finder's "動画（Gemma 2）" mode.
// Runs in GitHub Actions (see .github/workflows/gemma-embed.yml), or locally with Node 20+ and ffmpeg on PATH.
//
// Same recipe as the browser (js/main.js, js/gemma-worker.js): 4 frames at the centres of 4 equal segments
// (short side 320 px) as one video + the thumbnail as a poster image, 70 vision tokens per image, q4 weights.
// When the frames cannot be read, the thumbnail alone is embedded (kind "t" instead of "v").
//
// Monsnode is read through the shared Cloudflare Worker (config.js). Monsnode answers 403 when the Worker runs
// outside Japan, so the Worker pins itself near AWS Tokyo ([placement] region in monsnodeWebPlayer/worker/wrangler.toml).
//
// Output (OUT_DIR, default ./data):
//   gemma.json         { version, model, dtype, frames, visionTokens, updatedAt,
//                        items: { [id]: [addedAt(ms), kind "v"|"t", scale, base64 int8 × 768] },
//                        meta:  { [id]: {title, thumbnail, pageUrl, redirectUrl, redirectId} },
//                        ranking: { fetchedAt, "24h": [ids…], "3d": [...], "7d": [...] } }
//   so the site can add these videos as candidates without reading monsnode itself.
//   gemma-status.json  what the last run did (counts, first errors), readable without signing in
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { env, AutoConfig, AutoProcessor, AutoModel, RawImage, RawVideo } from '@huggingface/transformers';

const run = promisify(execFile);
if (process.env.HF_CACHE_DIR) env.cacheDir = process.env.HF_CACHE_DIR;   // kept between Actions runs (actions/cache)
const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
const DTYPE = 'q4';
const VISION_TOKENS = 70;
const FRAMES = 4;
const FRAME_SIZE = 320;
const MAX_ITEMS = Number(process.env.MAX_ITEMS) || 6000;      // keep the newest this many
const NEW_PAGES = Number(process.env.NEW_PAGES) || 5;         // new-arrival list pages to read
const MAX_NEW = Number(process.env.MAX_NEW) || 1000;          // at most this many embeddings per run
const BUDGET_MS = (Number(process.env.TIME_BUDGET_MIN) || 300) * 60e3;   // stop in time to save (Actions: 6 h max)
const OUT = process.env.OUT_DIR || 'data';
const started = Date.now();
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function apiBase() {
  if (process.env.API_BASE) return process.env.API_BASE.replace(/\/+$/, '');
  const cfg = await readFile(new URL('../../config.js', import.meta.url), 'utf8');
  return cfg.match(/GPPL_API_BASE\s*=\s*"([^"]+)"/)[1].replace(/\/+$/, '');
}
const API = await apiBase();

const listErrors = [];   // shown in gemma-status.json (Actions logs need a sign-in to read)
async function api(path) {
  const res = await fetch(`${API}${path}`, { headers: { 'user-agent': 'Mozilla/5.0 (gppl-gemma-batch)' } });
  const text = await res.text();
  let data = {};
  try { data = JSON.parse(text); } catch { /* not JSON */ }
  if (!res.ok || data.error) throw new Error(data.error || `Worker HTTP ${res.status}: ${text.slice(0, 160).replace(/\s+/g, ' ')}`);
  return data;
}

const ranking = { fetchedAt: Date.now() };   // period → ids in ranking order (the site turns them into popularity)

/** Ranking videos first (they are what most people see), then the newest list pages. */
async function candidates() {
  const out = new Map();
  for (const [period, pages] of [['24h', [0]], ['3d', [0]], ['7d', [0, 1]]]) {
    ranking[period] = [];
    for (const page of pages) {
      try {
        const d = await api(`/search?ranking=1&period=${period}&page=${page}`);
        for (const v of d.videos || []) {
          if (!v.thumbnail || !v.redirectId) continue;
          if (!ranking[period].includes(v.id)) ranking[period].push(v.id);
          if (!out.has(v.id)) out.set(v.id, v);
        }
      } catch (error) { console.warn(`ranking ${period}/${page}: ${error.message}`); listErrors.push(`ranking ${period}/${page}: ${error.message}`); }
      await sleep(400);
    }
  }
  for (let page = 1; page <= NEW_PAGES; page++) {
    try {
      const d = await api(`/search?q=&page=${page}`);
      for (const v of d.videos || []) if (v.thumbnail && v.redirectId && !out.has(v.id)) out.set(v.id, v);
    } catch (error) { console.warn(`list page ${page}: ${error.message}`); listErrors.push(`list page ${page}: ${error.message}`); break; }
    await sleep(500);
  }
  return [...out.values()];
}

async function duration(url) {
  const { stdout } = await run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', url], { timeout: 30e3 });
  const d = Number.parseFloat(stdout);
  if (!Number.isFinite(d) || d <= 0) throw new Error('unknown duration');
  return d;
}

/** One frame at `t` seconds, short side FRAME_SIZE, as a RawImage (ffmpeg seeks with HTTP range requests). */
async function frameAt(url, t) {
  const scale = `scale='if(lt(iw,ih),${FRAME_SIZE},-2)':'if(lt(iw,ih),-2,${FRAME_SIZE})'`;
  const { stdout } = await run('ffmpeg', ['-v', 'error', '-ss', t.toFixed(3), '-i', url, '-frames:v', '1', '-vf', scale,
    '-f', 'image2', '-c:v', 'mjpeg', '-q:v', '3', 'pipe:1'], { timeout: 60e3, encoding: 'buffer', maxBuffer: 20e6 });
  if (!stdout.length) throw new Error(`no frame at ${t.toFixed(1)}s`);
  return RawImage.fromBlob(new Blob([stdout], { type: 'image/jpeg' }));
}

async function frames(video) {
  const { videoUrl } = await api(`/resolve?id=${encodeURIComponent(video.redirectId)}`);
  const d = await duration(videoUrl);
  const images = [];
  for (let i = 0; i < FRAMES; i++) images.push(await frameAt(videoUrl, d * (i + 0.5) / FRAMES));
  return new RawVideo(images, d);
}

async function poster(video) {
  const res = await fetch(video.thumbnail);
  if (!res.ok) throw new Error(`thumbnail ${res.status}`);
  return RawImage.fromBlob(await res.blob());
}

function quantize(vec) {
  let m = 0; for (const x of vec) m = Math.max(m, Math.abs(x));
  const scale = m / 127 || 1;
  const q = Int8Array.from(vec, x => Math.round(x / scale));
  return [scale, Buffer.from(q.buffer).toString('base64')];
}

// ---------- main ----------
await mkdir(OUT, { recursive: true });
let data = { items: {} };
try { data = JSON.parse(await readFile(`${OUT}/gemma.json`, 'utf8')); } catch { /* first run */ }
const items = data.items || {};
const meta = data.meta || {};
const status = { startedAt: new Date(started).toISOString(), api: API, candidates: 0, todo: 0, added: 0, upgraded: 0, thumbOnly: 0, failed: 0, errors: [], frameErrors: [] };

async function save(final = false) {
  const ids = Object.keys(items).sort((a, b) => items[b][0] - items[a][0]).slice(0, MAX_ITEMS);
  const kept = Object.fromEntries(ids.map(id => [id, items[id]]));
  const keptMeta = Object.fromEntries(ids.filter(id => meta[id]).map(id => [id, meta[id]]));
  const keptClip = Object.fromEntries(ids.filter(id => data.clip?.[id]).map(id => [id, data.clip[id]]));   // filled by clip.mjs
  const out = { version: 2, model: MODEL_ID, dtype: DTYPE, frames: FRAMES, frameSize: FRAME_SIZE, visionTokens: VISION_TOKENS,
    updatedAt: new Date().toISOString(), count: ids.length, items: kept, meta: keptMeta, ranking, clip: keptClip };
  await writeFile(`${OUT}/gemma.json`, JSON.stringify(out));
  await writeFile(`${OUT}/gemma-status.json`, JSON.stringify({ ...status, total: ids.length, final,
    elapsedMin: +((Date.now() - started) / 60e3).toFixed(1), updatedAt: out.updatedAt }, null, 1));
}

const list = await candidates();
status.candidates = list.length;
for (const v of list) meta[v.id] = { title: v.title, thumbnail: v.thumbnail, pageUrl: v.pageUrl, redirectUrl: v.redirectUrl, redirectId: v.redirectId };
status.listErrors = listErrors.slice(0, 6);
// new videos first, then ones that only have a thumbnail embedding (their frames are tried again)
const todo = [...list.filter(v => !items[v.id]), ...list.filter(v => items[v.id]?.[1] === 't')].slice(0, MAX_NEW);
status.todo = todo.length;
console.log(`candidates ${list.length}, to embed ${todo.length}, stored ${Object.keys(items).length}`);

const processor = await AutoProcessor.from_pretrained(MODEL_ID);
processor.image_processor.max_soft_tokens = VISION_TOKENS;
processor.video_processor.frame_processor.max_soft_tokens = VISION_TOKENS;
const config = await AutoConfig.from_pretrained(MODEL_ID);
delete config.audio_config;   // vision + text only
const model = await AutoModel.from_pretrained(MODEL_ID, { config, dtype: DTYPE, device: 'cpu' });
console.log(`model ready in ${((Date.now() - started) / 1e3).toFixed(0)} s`);

for (const [k, v] of todo.entries()) {
  if (Date.now() - started > BUDGET_MS) { console.log('time budget reached'); break; }
  const t0 = Date.now();
  try {
    const [clip, img] = await Promise.all([
      frames(v).catch(error => {
        const why = String(error.stderr || error.message).split('\n').filter(Boolean).slice(0, 2).join(' / ').slice(0, 240);
        console.log(`  ${v.id}: frames unavailable (${why})`);
        if (status.frameErrors.length < 8) status.frameErrors.push(`${v.id}: ${why}`);
        return null;
      }),
      poster(v).catch(() => null),
    ]);
    if (!clip && !img) throw new Error('no frames and no thumbnail');
    const had = items[v.id];
    if (had?.[1] === 't' && !clip) { console.log(`${k + 1}/${todo.length} ${v.id} still thumbnail only`); continue; }
    const inputs = await processor(null, img ? [img] : null, null, clip ? [clip] : null);
    const { sentence_embedding } = await model(inputs);
    const vec = Float32Array.from(sentence_embedding.data.subarray(0, sentence_embedding.dims.at(-1)));
    let n = 0; for (const x of vec) n += x * x; n = Math.sqrt(n) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= n;
    items[v.id] = [had?.[0] ?? Date.now(), clip ? 'v' : 't', ...quantize(vec)];
    if (had) status.upgraded++; else status.added++;
    if (!clip) status.thumbOnly++;
    console.log(`${k + 1}/${todo.length} ${v.id} ${clip ? 'video' : 'thumb'} ${((Date.now() - t0) / 1e3).toFixed(1)} s`);
  } catch (error) {
    status.failed++;
    if (status.errors.length < 10) status.errors.push(`${v.id}: ${String(error.message).slice(0, 200)}`);
    console.log(`${k + 1}/${todo.length} ${v.id} failed: ${error.message}`);
  }
  if ((status.added + status.upgraded) % 25 === 24) await save();
}
await save(true);
console.log(`done: added ${status.added} (thumbnail only ${status.thumbOnly}), failed ${status.failed}, stored ${Math.min(Object.keys(items).length, MAX_ITEMS)}`);
