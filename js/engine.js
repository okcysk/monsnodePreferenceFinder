// Ties PCA + GPPL + acquisition together. Shared by the extension page and tests.html.
import { fitWithModelSelection, baseScale } from './gppl.js';
import { pcaProject, projectOnto } from './pca.js';
import { initialSelection, selectNext } from './acquisition.js';
import { popularityScores } from './popularity.js';

// PCA only changes when the pool changes. When the pool merely grows a little (embedding still running,
// a few more videos fetched), the new videos are projected onto the existing axes instead of recomputing,
// which is the slow part (≈ N² · iterations). A full recompute happens when the pool changes a lot.
let pcaCache = null;
const REUSE_GROWTH = 0.25;   // reuse the axes while new videos are at most 25% of the ones they were fitted on
function projectPool(embeddings, poolIds, pcaDim) {
  const c = pcaCache;
  if (c && c.dim === pcaDim && c.basis) {
    const same = c.size === poolIds.length && poolIds.every(id => c.Z.has(id) && c.vecs.get(id) === embeddings.get(id));
    if (same) return c;
    const known = new Set(poolIds.filter(id => c.Z.has(id) && c.vecs.get(id) === embeddings.get(id)));
    const added = poolIds.length - known.size;
    // reuse only if (almost) all fitted videos are still there with the same vectors and few were added
    if (added <= c.fitted * REUSE_GROWTH && known.size >= c.fitted * 0.9) {
      const Z = new Map(poolIds.map(id => [id, known.has(id) ? c.Z.get(id) : projectOnto(c.basis, embeddings.get(id))]));
      pcaCache = { ...c, size: poolIds.length, projected: poolIds.map(id => Z.get(id)), Z, vecs: new Map(poolIds.map(id => [id, embeddings.get(id)])) };
      return pcaCache;
    }
  }
  const projected = pcaProject(poolIds.map(id => embeddings.get(id)), pcaDim);
  const Z = new Map(poolIds.map((id, i) => [id, projected[i]]));
  pcaCache = { size: poolIds.length, fitted: poolIds.length, dim: pcaDim, projected, Z, basis: projected.basis,
    vecs: new Map(poolIds.map(id => [id, embeddings.get(id)])) };
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
