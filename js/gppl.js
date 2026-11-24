// Pairwise Gaussian-process preference learning (Chu & Ghahramani, 2005)
// with a Laplace approximation. Pure JS, no DOM: usable from pages, workers and tests.

const SQRT2 = Math.SQRT2;
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

/** erfc with ~1e-7 relative accuracy (Numerical Recipes erfcc). */
function erfc(x) {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

export function normPdf(x) { return Math.exp(-0.5 * x * x - LOG_SQRT_2PI); }
export function normCdf(x) { return 0.5 * erfc(-x / SQRT2); }

/** log Φ(z), stable for very negative z. */
export function logNormCdf(z) {
  if (z > -5) return Math.log(normCdf(z));
  // asymptotic: Φ(z) ≈ φ(z)/(-z) · (1 - 1/z² + 3/z⁴)
  const z2 = z * z;
  return -0.5 * z2 - LOG_SQRT_2PI - Math.log(-z) + Math.log(1 - 1 / z2 + 3 / (z2 * z2));
}

/** φ(z)/Φ(z) (inverse Mills ratio), stable for very negative z. */
export function millsRatio(z) {
  if (z > -5) return normPdf(z) / normCdf(z);
  return Math.exp(-0.5 * z * z - LOG_SQRT_2PI - logNormCdf(z));
}

/** Expand a best-first ordering into (winner, loser) pairs: 4 items → 6 pairs. */
export function rankingToPairs(order) {
  const pairs = [];
  for (let a = 0; a < order.length; a++)
    for (let b = a + 1; b < order.length; b++) pairs.push([order[a], order[b]]);
  return pairs;
}

export function sqDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

export function rbf(a, b, ell, sf2) { return sf2 * Math.exp(-sqDist(a, b) / (2 * ell * ell)); }

/** Median pairwise distance of (a sample of) vectors; the usual length-scale heuristic. */
export function medianDistance(vectors, maxSample = 200) {
  const v = vectors.length > maxSample ? vectors.filter((_, i) => i % Math.ceil(vectors.length / maxSample) === 0) : vectors;
  const d = [];
  for (let i = 0; i < v.length; i++) for (let j = i + 1; j < v.length; j++) d.push(Math.sqrt(sqDist(v[i], v[j])));
  if (!d.length) return 1;
  d.sort((x, y) => x - y);
  return d[Math.floor(d.length / 2)] || 1;
}

/** Cholesky of a symmetric positive-definite n×n row-major matrix. Adds jitter if needed. */
export function cholesky(A, n) {
  for (let jitter = 0, attempt = 0; attempt < 8; attempt++, jitter = jitter ? jitter * 10 : 1e-9) {
    const L = new Float64Array(n * n);
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * n + j] + (i === j ? jitter : 0);
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        if (i === j) {
          if (s <= 0) { ok = false; break; }
          L[i * n + i] = Math.sqrt(s);
        } else L[i * n + j] = s / L[j * n + j];
      }
    }
    if (ok) return L;
  }
  throw new Error('Matrix is not positive definite');
}

/** LU decomposition with partial pivoting (in place copy). Returns {LU, piv, logAbsDet}. */
function luDecompose(A, n) {
  const LU = Float64Array.from(A);
  const piv = new Int32Array(n);
  let logAbsDet = 0;
  for (let i = 0; i < n; i++) piv[i] = i;
  for (let k = 0; k < n; k++) {
    let p = k, max = Math.abs(LU[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(LU[i * n + k]); if (v > max) { max = v; p = i; } }
    if (max < 1e-300) throw new Error('Singular matrix');
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = LU[k * n + j]; LU[k * n + j] = LU[p * n + j]; LU[p * n + j] = t; }
      const t = piv[k]; piv[k] = piv[p]; piv[p] = t;
    }
    const d = LU[k * n + k];
    logAbsDet += Math.log(Math.abs(d));
    for (let i = k + 1; i < n; i++) {
      const f = (LU[i * n + k] /= d);
      if (f) for (let j = k + 1; j < n; j++) LU[i * n + j] -= f * LU[k * n + j];
    }
  }
  return { LU, piv, logAbsDet };
}

function luSolve({ LU, piv }, b, n) {
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = b[piv[i]];
  for (let i = 0; i < n; i++) { let s = x[i]; for (let j = 0; j < i; j++) s -= LU[i * n + j] * x[j]; x[i] = s; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; for (let j = i + 1; j < n; j++) s -= LU[i * n + j] * x[j]; x[i] = s / LU[i * n + i]; }
  return x;
}

/**
 * GPPL model.
 *   f = m + g,  g ~ GP(0, k),  k = sf2 · exp(-|x-x'|² / 2ℓ²)
 *   m = fixed prior mean per item (e.g. popularity; 0 when not given)
 *   P(i ≻ j | f) = Φ((f_i − f_j) / (√2 σn))
 * The Laplace approximation is done over g, so the prior mean only shifts the likelihood.
 */
export class GPPL {
  constructor({ ell = 1, sf2 = 1, sn = 0.5 } = {}) {
    this.ell = ell; this.sf2 = sf2; this.sn = sn;
    this.X = []; this.n = 0; this.f = null;
  }

  /**
   * @param {ArrayLike<number>[]} X  feature vector per item (items referenced by index)
   * @param {[number, number][]} pairs  (winner index, loser index)
   * @param {{mean?: ArrayLike<number>}} [opts]  prior mean m per item
   */
  fit(X, pairs, { maxIter = 50, tol = 1e-6, mean = null } = {}) {
    const n = X.length;
    this.X = X; this.n = n; this.pairs = pairs;
    const m0 = mean ? Float64Array.from(mean) : new Float64Array(n);
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      const v = rbf(X[i], X[j], this.ell, this.sf2) + (i === j ? 1e-6 : 0);
      K[i * n + j] = v; K[j * n + i] = v;
    }
    this.K = K;
    const c = SQRT2 * this.sn;
    let f = new Float64Array(n);
    let a = new Float64Array(n);
    let prevObj = -Infinity;

    const evalAt = (fv) => {
      const g = new Float64Array(n);
      const W = new Float64Array(n * n);
      let logLik = 0;
      for (const [i, j] of pairs) {
        const z = (fv[i] + m0[i] - fv[j] - m0[j]) / c;
        logLik += logNormCdf(z);
        const r = millsRatio(z);
        g[i] += r / c; g[j] -= r / c;
        const w = (r * r + z * r) / (c * c);
        W[i * n + i] += w; W[j * n + j] += w; W[i * n + j] -= w; W[j * n + i] -= w;
      }
      return { g, W, logLik };
    };

    for (let iter = 0; iter < maxIter; iter++) {
      const { g, W, logLik } = evalAt(f);
      // Newton step: f_new = K (I + W K)^{-1} (W f + g)
      const b = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = g[i]; for (let j = 0; j < n; j++) s += W[i * n + j] * f[j]; b[i] = s; }
      const IWK = matIWK(W, K, n);
      const lu = luDecompose(IWK, n);
      let aNew = luSolve(lu, b, n);
      let fNew = matVec(K, aNew, n);
      // simple step halving to guarantee ascent of Ψ(f) = log p(D|f) − ½ aᵀKa
      let obj = evalAt(fNew).logLik - 0.5 * dot(aNew, fNew);
      let step = 1;
      while (obj < prevObj - 1e-10 && step > 1e-3) {
        step /= 2;
        aNew = aNew.map((v, k) => a[k] + step * (v - a[k]));
        fNew = matVec(K, aNew, n);
        obj = evalAt(fNew).logLik - 0.5 * dot(aNew, fNew);
      }
      const delta = Math.abs(obj - prevObj);
      f = fNew; a = aNew; prevObj = obj;
      if (delta < tol) break;
    }
    const { g, W, logLik } = evalAt(f);
    const lu = luDecompose(matIWK(W, K, n), n);
    // M = (I + W K)^{-1} W, used for predictive variances: var* = k** − k*ᵀ M k*
    const M = new Float64Array(n * n);
    const col = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) col[i] = W[i * n + j];
      const x = luSolve(lu, col, n);
      for (let i = 0; i < n; i++) M[i * n + j] = x[i];
    }
    this.f = f; this.alpha = g; this.M = M;
    // Laplace log marginal likelihood: log p(D|f̂) − ½ f̂ᵀK⁻¹f̂ − ½ log|I + W K|
    this.logMarginal = logLik - 0.5 * dot(a, f) - 0.5 * lu.logAbsDet;
    return this;
  }

  /** Posterior mean and variance of the latent utility at new points (meanStar = their prior mean). */
  predict(Xstar, meanStar = null) {
    const n = this.n, m = Xstar.length;
    const mu = new Float64Array(m), variance = new Float64Array(m);
    const ks = new Float64Array(n), tmp = new Float64Array(n);
    for (let p = 0; p < m; p++) {
      for (let i = 0; i < n; i++) ks[i] = rbf(Xstar[p], this.X[i], this.ell, this.sf2);
      let mean = 0;
      for (let i = 0; i < n; i++) mean += ks[i] * this.alpha[i];
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += this.M[i * n + j] * ks[j]; tmp[i] = s; }
      mu[p] = mean + (meanStar ? meanStar[p] : 0);
      variance[p] = Math.max(1e-12, this.sf2 - dot(ks, tmp));
    }
    return { mu, variance };
  }
}

function matIWK(W, K, n) {
  const R = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const w = W[i * n + k];
      if (!w) continue;
      for (let j = 0; j < n; j++) R[i * n + j] += w * K[k * n + j];
    }
    R[i * n + i] += 1;
  }
  return R;
}

function matVec(A, x, n) {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += A[i * n + j] * x[j]; y[i] = s; }
  return y;
}

function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

/**
 * Fit with length-scale chosen by Laplace marginal likelihood over a small grid
 * around the median-distance heuristic (only once there is enough data).
 */
export function fitWithModelSelection(X, pairs, baseEll, { sf2 = 1, sn = 0.5, minPairsForSelection = 20, mean = null } = {}) {
  const grid = pairs.length >= minPairsForSelection ? [0.5, 0.75, 1, 1.5, 2] : [1];
  let best = null;
  for (const s of grid) {
    const model = new GPPL({ ell: baseEll * s, sf2, sn }).fit(X, pairs, { mean });
    if (!best || model.logMarginal > best.logMarginal) best = model;
  }
  return best;
}
