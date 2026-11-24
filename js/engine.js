// Ties PCA + GPPL + acquisition together. Shared by the extension page and tests.html.
import { fitWithModelSelection, baseScale } from './gppl.js';
import { pcaProject } from './pca.js';
import { initialSelection, selectNext } from './acquisition.js';
import { popularityScores } from './popularity.js';

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

// Popularity scores only change with the pool or the ranking data.
let popCache = null;
function popularityFor(poolIds, Z, popularity) {
  if (!popularity?.size) return null;
  let sum = 0; for (const [id, s] of popularity) if (Z.has(id)) sum += s;
  const key = `${poolIds.length}|${popularity.size}|${sum.toFixed(6)}`;
  if (popCache && popCache.Z === Z && popCache.key === key) return popCache.result;
  const result = popularityScores(poolIds, Z, popularity);
  popCache = { Z, key, result };
  return result;
}

/**
 * Prior mean per pool video: popWeight · (popularity − average popularity).
 * @returns {{prior: Map<string, number>, pop: object} | null}
 */
export function priorMean(poolIds, Z, popularity, popWeight) {
  const pop = popularityFor(poolIds, Z, popularity);
  if (!pop) return null;
  let avg = 0; for (const id of poolIds) avg += pop.score.get(id) / poolIds.length;
  const prior = new Map(poolIds.map(id => [id, popWeight * (pop.score.get(id) - avg)]));
  return { prior, pop };
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
 * @param {Map<string, number>} [p.popularity]  id → ranking score (popularity.js rankScore) for ranked videos
 * @param {number} [p.popWeight=0]  strength of the popularity prior mean (0 = off)
 * @param {'rbf'|'linear'|'quad'} [p.kernel='rbf']  GP kernel (gppl.js KERNELS)
 */
export function computeRound({ embeddings, pairs, shown, beta, excluded = new Set(), pcaDim = 48, rng = Math.random, popularity = null, popWeight = 0, kernel = 'rbf' }) {
  const poolIds = [...embeddings.keys()];
  const { projected, Z } = projectPool(embeddings, poolIds, pcaDim);
  const pm = priorMean(poolIds, Z, popularity, popWeight);
  const prior = popWeight > 0 && pm ? pm.prior : null;
  const pop = pm?.pop ?? null;

  const validPairs = pairs.filter(([w, l]) => Z.has(w) && Z.has(l));
  if (!validPairs.length) {
    const usable = poolIds.filter(id => !excluded.has(id));
    const fresh = usable.filter(id => !shown.has(id));
    let cands = fresh.length >= 4 ? fresh : usable;
    // With a prior, spread the first four over the more promising part of the pool.
    if (prior && cands.length > 8) cands = cands.slice().sort((a, b) => prior.get(b) - prior.get(a)).slice(0, Math.max(8, Math.ceil(cands.length * 0.4)));
    return { best: null, selection: initialSelection(cands, Z, 4, rng), mu: null, sigma: null, model: null, prior, pop };
  }

  const trainIds = [...new Set(validPairs.flat())];
  const index = new Map(trainIds.map((id, i) => [id, i]));
  const X = trainIds.map(id => Z.get(id));
  const baseEll = baseScale(kernel, projected);
  const mean = prior ? trainIds.map(id => prior.get(id)) : null;
  const model = fitWithModelSelection(X, validPairs.map(([w, l]) => [index.get(w), index.get(l)]), baseEll, { mean, kernel });

  const { mu: m, variance } = model.predict(poolIds.map(id => Z.get(id)), prior ? poolIds.map(id => prior.get(id)) : null);
  const mu = new Map(poolIds.map((id, i) => [id, m[i]]));
  const sigma = new Map(poolIds.map((id, i) => [id, Math.sqrt(variance[i])]));
  const { best, selection } = selectNext({ poolIds, Z, mu, sigma, shown, beta, excluded });
  return { best, selection, mu, sigma, model, prior, pop };
}

/** Top-n by posterior mean among shown videos (what the user has actually seen). */
export function topSeen(mu, shown, n = 10) {
  if (!mu) return [];
  return [...shown].filter(id => mu.has(id)).sort((a, b) => mu.get(b) - mu.get(a)).slice(0, n);
}
