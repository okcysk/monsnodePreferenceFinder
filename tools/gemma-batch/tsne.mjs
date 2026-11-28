// Precompute the map's t-SNE layout of the EmbeddingGemma 2 vectors (after embed.mjs), with the same code as the
// browser (js/pca.js → 48-D, js/tsne.js → 2-D), so the map can show it at once instead of running t-SNE itself.
//   tsne: { ids: [...], xy: [x0, y0, x1, y1, ...] (rounded), perplexity, iters, at }
// Only the newest MAX_TSNE videos (t-SNE is O(N²) per step). The map uses it when nearly all of its videos are in it.
import { readFile, writeFile } from 'node:fs/promises';
import { pcaProject } from '../../js/pca.js';
import { TSNE } from '../../js/tsne.js';

const OUT = process.env.OUT_DIR || 'data';
const MAX_TSNE = Number(process.env.MAX_TSNE) || 1500;
const ITERS = 1000;
const started = Date.now();

const file = `${OUT}/gemma.json`;
const data = JSON.parse(await readFile(file, 'utf8'));
const items = data.items || {};
const ids = Object.keys(items).sort((a, b) => items[b][0] - items[a][0]).slice(0, MAX_TSNE);

function dequantize(scale, b64) {
  const q = new Int8Array(Buffer.from(b64, 'base64'));
  const vec = Float32Array.from(q, x => x * scale);
  let n = 0; for (const x of vec) n += x * x; n = Math.sqrt(n) || 1;
  return vec.map(x => x / n);
}

if (ids.length >= 3) {
  const pca = pcaProject(ids.map(id => dequantize(items[id][2], items[id][3])), 48);
  const perplexity = Math.min(30, Math.max(5, Math.floor(ids.length / 4)));   // same rule as js/viz.js
  const tsne = new TSNE(pca, { perplexity });
  while (tsne.iter < ITERS) tsne.step(50);
  const xy = tsne.Y.flatMap(([x, y]) => [Math.round(x * 100) / 100, Math.round(y * 100) / 100]);
  data.tsne = { ids, xy, perplexity, iters: ITERS, at: new Date().toISOString() };
  await writeFile(file, JSON.stringify(data));
  try {
    const status = JSON.parse(await readFile(`${OUT}/gemma-status.json`, 'utf8'));
    status.tsne = { count: ids.length, elapsedMin: +((Date.now() - started) / 60e3).toFixed(1) };
    await writeFile(`${OUT}/gemma-status.json`, JSON.stringify(status, null, 1));
  } catch { /* no status file */ }
}
console.log(`t-SNE: ${ids.length} videos in ${((Date.now() - started) / 1e3).toFixed(0)} s`);
