// Popularity prior from monsnode's ranking pages (?ranking=1&period=24h|3d|7d).
//   A: videos found in a ranking get a score from their period and rank.
//   B: a logistic regression on the PCA embedding ("ranked" vs "not ranked") spreads that
//      to every other candidate, shrunk by how well it predicts (cross-validated AUC).
// The result is a 0–1 popularity per video; engine.js turns it into the GP prior mean.

/** Ranking pages to read. Ranking pages start at page 0; 24h and 3d have one page. */
export const POP_PERIODS = [
  { period: '24h', pages: [0], weight: 1 },
  { period: '3d', pages: [0], weight: 0.85 },
  { period: '7d', pages: [0, 1], weight: 0.7 },
];

/** Score of a ranked video: higher for shorter periods and better ranks (0.35–1). */
export function rankScore(period, rank, count) {
  const w = POP_PERIODS.find(p => p.period === period)?.weight ?? 0.7;
  return w * (1 - 0.5 * rank / Math.max(1, count - 1));
}

/** Keep the best ranking a video has ever had. */
export function mergePop(old, next) {
  if (!old) return next;
  if (!next) return old;
  return next.s >= old.s ? { ...next } : { ...old, seenAt: next.seenAt };
}

const sigmoid = z => 1 / (1 + Math.exp(-z));

/**
 * L2-regularised, class-balanced logistic regression by gradient descent.
 * @param {ArrayLike<number>[]} X  rows (already roughly centred: PCA output)
 * @param {number[]} y  0/1
 */
export function fitLogistic(X, y, { lambda = 1e-2, iters = 300, lr = 0.5 } = {}) {
  const n = X.length, d = X[0].length;
  // standardise columns so one learning rate fits all
  const mean = new Float64Array(d), sd = new Float64Array(d);
  for (const x of X) for (let k = 0; k < d; k++) mean[k] += x[k] / n;
  for (const x of X) for (let k = 0; k < d; k++) { const t = x[k] - mean[k]; sd[k] += t * t / n; }
  for (let k = 0; k < d; k++) sd[k] = Math.sqrt(sd[k]) || 1;
  const Xs = X.map(x => Float64Array.from({ length: d }, (_, k) => (x[k] - mean[k]) / sd[k]));
  const pos = y.reduce((s, v) => s + v, 0), neg = n - pos;
  const cw = y.map(v => (v ? n / (2 * pos) : n / (2 * neg)));
  const w = new Float64Array(d); let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = new Float64Array(d); let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b; const x = Xs[i];
      for (let k = 0; k < d; k++) z += w[k] * x[k];
      const e = cw[i] * (sigmoid(z) - y[i]);
      gb += e;
      for (let k = 0; k < d; k++) gw[k] += e * x[k];
    }
    for (let k = 0; k < d; k++) w[k] -= lr * (gw[k] / n + lambda * w[k]);
    b -= lr * gb / n;
  }
  return x => { let z = b; for (let k = 0; k < d; k++) z += w[k] * (x[k] - mean[k]) / sd[k]; return sigmoid(z); };
}

/** Area under the ROC curve (ties count half). */
export function auc(scores, y) {
  const idx = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b]);
  let rankSum = 0, pos = 0;
  for (let r = 0; r < idx.length;) {
    let e = r; while (e + 1 < idx.length && scores[idx[e + 1]] === scores[idx[r]]) e++;
    const avg = (r + e) / 2 + 1;
    for (let k = r; k <= e; k++) if (y[idx[k]]) { rankSum += avg; pos++; }
    r = e + 1;
  }
  const neg = idx.length - pos;
  return pos && neg ? (rankSum - pos * (pos + 1) / 2) / (pos * neg) : 0.5;
}

/**
 * Popularity per pool video.
 * @param {string[]} poolIds
 * @param {Map<string, ArrayLike<number>>} Z  PCA features
 * @param {Map<string, number>} known  id → rankScore for videos seen in a ranking
 * @returns {{score: Map<string, number>, auc: number, shrink: number, known: number} | null}
 */
export function popularityScores(poolIds, Z, known, { folds = 5 } = {}) {
  const y = poolIds.map(id => (known.has(id) ? 1 : 0));
  const nPos = y.reduce((s, v) => s + v, 0);
  if (!nPos) return null;
  const X = poolIds.map(id => Z.get(id));
  let cvAuc = 0.5;
  if (nPos >= 10 && poolIds.length - nPos >= 10) {
    // out-of-fold predictions → honest AUC
    const oof = new Array(poolIds.length);
    for (let f = 0; f < folds; f++) {
      const tr = [], te = [];
      poolIds.forEach((_, i) => (i % folds === f ? te : tr).push(i));
      const model = fitLogistic(tr.map(i => X[i]), tr.map(i => y[i]), { iters: 150 });
      for (const i of te) oof[i] = model(X[i]);
    }
    cvAuc = auc(oof, y);
  }
  // shrink the guess for unranked videos toward their average by how much better than chance it is
  const shrink = Math.max(0, Math.min(1, (cvAuc - 0.5) / 0.25));
  const score = new Map();
  if (shrink > 0) {
    const model = fitLogistic(X, y);
    const p = poolIds.map((id, i) => (y[i] ? null : model(X[i])));
    const unk = p.filter(v => v !== null);
    const avg = unk.reduce((s, v) => s + v, 0) / (unk.length || 1);
    poolIds.forEach((id, i) => { if (!y[i]) score.set(id, 0.75 * (avg + shrink * (p[i] - avg))); });
  } else {
    poolIds.forEach((id, i) => { if (!y[i]) score.set(id, 0.375); });
  }
  poolIds.forEach((id, i) => { if (y[i]) score.set(id, 0.5 + 0.5 * known.get(id)); });
  return { score, auc: cvAuc, shrink, known: nPos };
}
