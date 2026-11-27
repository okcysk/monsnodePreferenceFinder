// Precompute CLIP ViT-B/32 thumbnail embeddings (the site's tags / text search and its "サムネ" modes).
// Runs after embed.mjs in GitHub Actions and adds them to the same gemma.json:
//   clip: { [id]: [scale, base64 int8 × 512] }
// Same model, library version (transformers.js 3.5.1, aliased as "transformers3") and dtype (q8) as the browser's
// WASM setting (js/embed-worker.js), so the site uses these only when its backend is WASM; WebGPU (fp32) vectors
// differ noticeably (cosine 0.89–0.96) and are still computed in the browser.
// A separate process from embed.mjs: the two transformers.js versions bring different onnxruntime-node builds.
import { readFile, writeFile } from 'node:fs/promises';
import { env, AutoProcessor, CLIPVisionModelWithProjection, RawImage } from 'transformers3';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const OUT = process.env.OUT_DIR || 'data';
const BUDGET_MS = (Number(process.env.CLIP_BUDGET_MIN) || 30) * 60e3;
if (process.env.HF_CACHE_DIR) env.cacheDir = process.env.HF_CACHE_DIR;
const started = Date.now();

const file = `${OUT}/gemma.json`;
const data = JSON.parse(await readFile(file, 'utf8'));
const clip = data.clip || {};
const todo = Object.keys(data.items || {}).filter(id => data.meta?.[id]?.thumbnail && !clip[id]);
console.log(`CLIP: ${todo.length} thumbnails to embed (${Object.keys(clip).length} stored)`);

const processor = await AutoProcessor.from_pretrained(MODEL_ID);
const model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { dtype: 'q8', device: 'cpu' });

function quantize(vec) {
  let m = 0; for (const x of vec) m = Math.max(m, Math.abs(x));
  const scale = m / 127 || 1;
  return [scale, Buffer.from(Int8Array.from(vec, x => Math.round(x / scale)).buffer).toString('base64')];
}

let done = 0, failed = 0;
for (const id of todo) {
  if (Date.now() - started > BUDGET_MS) { console.log('CLIP: time budget reached'); break; }
  try {
    const res = await fetch(data.meta[id].thumbnail);
    if (!res.ok) throw new Error(`thumbnail ${res.status}`);
    const image = await RawImage.fromBlob(await res.blob());
    const { image_embeds } = await model(await processor([image]));   // one image per call, like the browser (q8)
    const vec = Float32Array.from(image_embeds.data.subarray(0, image_embeds.dims.at(-1)));
    let n = 0; for (const x of vec) n += x * x; n = Math.sqrt(n) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= n;
    clip[id] = quantize(vec);
    done++;
  } catch (error) {
    failed++;
    console.log(`CLIP ${id}: ${error.message}`);
  }
}
// keep only ids that are still in the file
data.clip = Object.fromEntries(Object.keys(data.items).filter(id => clip[id]).map(id => [id, clip[id]]));
await writeFile(file, JSON.stringify(data));
try {
  const status = JSON.parse(await readFile(`${OUT}/gemma-status.json`, 'utf8'));
  status.clip = { added: done, failed, total: Object.keys(data.clip).length, elapsedMin: +((Date.now() - started) / 60e3).toFixed(1) };
  await writeFile(`${OUT}/gemma-status.json`, JSON.stringify(status, null, 1));
} catch { /* no status file */ }
console.log(`CLIP: added ${done}, failed ${failed}, stored ${Object.keys(data.clip).length}`);
