// Ties PCA + GPPL + acquisition together. Shared by the extension page and tests.html.
import { fitWithModelSelection, medianDistance } from './gppl.js';
import { pcaProject } from './pca.js';
import { initialSelection, selectNext } from './acquisition.js';

// PCA only changes when the pool changes, so cache it per (embeddings map, pool size, dim).
let pcaCache = null;
function projectPool(embeddings, poolIds, pcaDim) {
  if (pcaCache && pcaCache.embeddings === embeddings && pcaCache.size === poolIds.length && pcaCache.dim === pcaDim
    && poolIds.every(id => pcaCache.Z.has(id))) return pcaCache;
  const projected = pcaProject(poolIds.map(id => embeddings.get(id)), pcaDim);
  const Z = new Map(poolIds.map((id, i) => [id, projected[i]]));
  pcaCache = { embeddings, size: poolIds.length, dim: pcaDim, projected, Z };
  return pcaCache;
}

/**
 * @param {object} p
 * @param {Map<string, ArrayLike<number>>} p.embeddings  id → raw embedding (pool)
 * @param {[string, string][]} p.pairs  (winner id, loser id)
 * @param {Set<string>} p.shown
 * @param {number} p.beta
 * @param {Set<string>} [p.excluded]  videos the user chose not to rate (kept for training, never selected)
 * @param {number} [p.pcaDim=48]
 * @param {() => number} [p.rng]
 */
export function computeRound({ embeddings, pairs, shown, beta, excluded = new Set(), pcaDim = 48, rng = Math.random }) {
  const poolIds = [...embeddings.keys()];
  const { projected, Z } = projectPool(embeddings, poolIds, pcaDim);

  const validPairs = pairs.filter(([w, l]) => Z.has(w) && Z.has(l));
  if (!validPairs.length) {
    const usable = poolIds.filter(id => !excluded.has(id));
    const fresh = usable.filter(id => !shown.has(id));
    return { best: null, selection: initialSelection(fresh.length >= 4 ? fresh : usable, Z, 4, rng), mu: null, sigma: null, model: null };
  }

  const trainIds = [...new Set(validPairs.flat())];
  const index = new Map(trainIds.map((id, i) => [id, i]));
  const X = trainIds.map(id => Z.get(id));
  const baseEll = medianDistance(projected);
  const model = fitWithModelSelection(X, validPairs.map(([w, l]) => [index.get(w), index.get(l)]), baseEll);

  const { mu: m, variance } = model.predict(poolIds.map(id => Z.get(id)));
  const mu = new Map(poolIds.map((id, i) => [id, m[i]]));
  const sigma = new Map(poolIds.map((id, i) => [id, Math.sqrt(variance[i])]));
  const { best, selection } = selectNext({ poolIds, Z, mu, sigma, shown, beta, excluded });
  return { best, selection, mu, sigma, model };
}

/** Top-n by posterior mean among shown videos (what the user has actually seen). */
export function topSeen(mu, shown, n = 10) {
  if (!mu) return [];
  return [...shown].filter(id => mu.has(id)).sort((a, b) => mu.get(b) - mu.get(a)).slice(0, n);
}
