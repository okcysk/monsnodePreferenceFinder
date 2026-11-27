// Embeddings precomputed daily by GitHub Actions (tools/gemma-batch: embed.mjs, clip.mjs), published on the
// orphan branch `gemma-data`. One JSON file:
//   items: { [id]: [addedAt, "v"|"t", scale, base64 int8 × 768] }   EmbeddingGemma 2 (q4, 4 frames + thumbnail)
//   meta:  { [id]: {title, thumbnail, pageUrl, redirectUrl, redirectId} }
//   ranking: { fetchedAt, "24h": [ids], "3d": [...], "7d": [...] }
//   clip:  { [id]: [scale, base64 int8 × 512] }                       CLIP thumbnail (q8, transformers.js 3.5.1)
// Against the browser's own vectors: Gemma cosine 0.97–0.99, CLIP 0.95–0.99. Gemma vectors share the browser's store;
// CLIP ones are used only for the map's tags / text search in the Gemma mode (not mixed into the "サムネ" modes).
export const GEMMA_DATA_URL = new URLSearchParams(location.search).get('gemmaData')   // ?gemmaData=… for testing
  || 'https://raw.githubusercontent.com/okcysk/monsnodePreferenceFinder/gemma-data/gemma.json';

let pending = null;
/** The file (or null when unavailable), fetched once per page. */
export const loadPrecomputed = () => pending ??= fetch(GEMMA_DATA_URL, { cache: 'no-cache' })
  .then(r => (r.ok ? r.json() : null)).catch(() => null);

/** int8 + scale → L2-normalised Float32Array */
export function dequantize(scale, b64) {
  const q = new Int8Array(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
  const vec = Float32Array.from(q, x => x * scale);
  let n = 0; for (const x of vec) n += x * x; n = Math.sqrt(n) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= n;
  return vec;
}
