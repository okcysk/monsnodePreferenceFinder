// PCA via the Gram matrix (N items ≤ a few hundred, D = 512) + power iteration with deflation.

/**
 * @param {ArrayLike<number>[]} vectors  N × D
 * @param {number} k  target dimension
 * @returns {Float64Array[]}  N × k projections (component signs are arbitrary; distances are what matter).
 *   The array also carries `basis = {mean, comps}` (comps: k unit vectors in the input space) so that
 *   vectors added later can be projected with projectOnto() without recomputing.
 */
export function pcaProject(vectors, k = 48) {
  const N = vectors.length;
  if (!N) return [];
  const D = vectors[0].length;
  k = Math.max(1, Math.min(k, N - 1, D));
  if (N < 3) return vectors.map(v => Float64Array.from(v));

  const mean = new Float64Array(D);
  for (const v of vectors) for (let d = 0; d < D; d++) mean[d] += v[d] / N;
  const Xc = vectors.map(v => { const r = new Float64Array(D); for (let d = 0; d < D; d++) r[d] = v[d] - mean[d]; return r; });

  // Gram matrix G = Xc Xcᵀ (N × N). Its eigenvectors u with eigenvalues λ give projections u·√λ.
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
    let s = 0; const a = Xc[i], b = Xc[j];
    for (let d = 0; d < D; d++) s += a[d] * b[d];
    G[i * N + j] = s; G[j * N + i] = s;
  }

  const out = Array.from({ length: N }, () => new Float64Array(k));
  const comps = [];
  const u = new Float64Array(N), w = new Float64Array(N);
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) - 0.5;
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < N; i++) u[i] = rand();
    let lambda = 0;
    for (let iter = 0; iter < 120; iter++) {
      for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += G[i * N + j] * u[j]; w[i] = s; }
      let norm = 0; for (let i = 0; i < N; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      let diff = 0;
      for (let i = 0; i < N; i++) { const v = w[i] / norm; diff += Math.abs(v - u[i]); u[i] = v; }
      const prev = lambda; lambda = norm;
      if (diff < 1e-6 || Math.abs(lambda - prev) < 1e-8 * lambda) break;   // plenty for distances; much faster than 1e-9
    }
    if (lambda < 1e-10) break;
    const s = Math.sqrt(lambda);
    for (let i = 0; i < N; i++) out[i][c] = u[i] * s;
    // the same component in the input space: Xcᵀu / √λ (a unit vector)
    const comp = new Float64Array(D);
    for (let i = 0; i < N; i++) { const w = u[i] / s, x = Xc[i]; for (let d = 0; d < D; d++) comp[d] += w * x[d]; }
    comps.push(comp);
    // deflate
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) G[i * N + j] -= lambda * u[i] * u[j];
  }
  out.basis = { mean, comps };
  return out;
}

/** Project a vector onto a basis from pcaProject (same coordinates as its output, k values). */
export function projectOnto(basis, v) {
  const { mean, comps } = basis, k = comps.length, out = new Float64Array(k);
  for (let c = 0; c < k; c++) { const comp = comps[c]; let s = 0; for (let d = 0; d < comp.length; d++) s += (v[d] - mean[d]) * comp[d]; out[c] = s; }
  return out;
}
