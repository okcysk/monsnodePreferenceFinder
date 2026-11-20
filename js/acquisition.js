// Choosing which 4 videos to show next.
import { sqDist } from './gppl.js';

function cosine(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return ab / (Math.sqrt(aa * bb) || 1);
}

/** Simple seeded PRNG so tests are reproducible. */
export function makeRng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

/**
 * First round: k-means++-style spread sampling, so the first four cover the space.
 * @param {string[]} ids
 * @param {Map<string, ArrayLike<number>>} Z
 */
export function initialSelection(ids, Z, k = 4, rng = Math.random) {
  if (ids.length <= k) return ids.slice();
  const chosen = [ids[Math.floor(rng() * ids.length)]];
  while (chosen.length < k) {
    const d = ids.map(id => chosen.includes(id) ? 0 : Math.min(...chosen.map(c => sqDist(Z.get(id), Z.get(c)))));
    const total = d.reduce((s, v) => s + v, 0);
    let r = rng() * total, pick = ids.find(id => !chosen.includes(id));
    for (let i = 0; i < ids.length; i++) { r -= d[i]; if (r <= 0 && d[i] > 0) { pick = ids[i]; break; } }
    chosen.push(pick);
  }
  return chosen;
}

/**
 * Keep the current best in slot 1, fill the other slots greedily by UCB = μ + β·σ
 * among videos not yet shown, skipping near-duplicates of what is already picked.
 *
 * @param {object} p
 * @param {string[]} p.poolIds     all candidate ids (embedded)
 * @param {Map<string, ArrayLike<number>>} p.Z
 * @param {Map<string, number>} p.mu
 * @param {Map<string, number>} p.sigma
 * @param {Set<string>} p.shown    ids already shown in some round
 * @param {number} p.beta
 * @param {Set<string>} [p.excluded]  ids the user chose not to rate: never shown again, never the best
 * @param {number} [p.k=4]
 * @param {number} [p.maxCos=0.98] diversity threshold in PCA space
 * @returns {{best: string, selection: string[]}}
 */
export function selectNext({ poolIds, Z, mu, sigma, shown, beta, excluded = new Set(), k = 4, maxCos = 0.98 }) {
  poolIds = poolIds.filter(id => !excluded.has(id));
  // Best = highest posterior mean among videos the user has actually seen (and rated).
  let best = null;
  for (const id of shown) if (!excluded.has(id) && mu.has(id) && (best === null || mu.get(id) > mu.get(best))) best = id;
  const selection = best ? [best] : [];
  const fresh = poolIds.filter(id => !shown.has(id));
  const pool = fresh.length >= k - selection.length ? fresh : poolIds.filter(id => id !== best);
  const scored = pool.map(id => ({ id, ucb: mu.get(id) + beta * sigma.get(id) })).sort((a, b) => b.ucb - a.ucb);
  for (const { id } of scored) {
    if (selection.length >= k) break;
    if (selection.includes(id)) continue;
    if (selection.some(s => cosine(Z.get(s), Z.get(id)) > maxCos)) continue;
    selection.push(id);
  }
  for (const { id } of scored) { if (selection.length >= k) break; if (!selection.includes(id)) selection.push(id); }
  return { best, selection };
}

export function nextBeta(beta) { return Math.max(0.5, beta * 0.9); }
