// Exact t-SNE (van der Maaten & Hinton, 2008). O(N²) per step — fine for N ≤ ~800.
// Usage: const t = new TSNE(vectors, {perplexity}); repeat t.step(n); t.Y → N×2

export class TSNE {
  constructor(vectors, { perplexity = 30, learningRate = null, seed = 1 } = {}) {
    const N = this.N = vectors.length;
    this.iter = 0;
    this.lr = learningRate ?? Math.max(N / 16, 50);   // like sklearn's 'auto'; 200 overshoots for small N
    // pairwise squared distances
    const D = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let s = 0; const a = vectors[i], b = vectors[j];
      for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
      D[i * N + j] = D[j * N + i] = s;
    }
    // conditional probabilities with binary search on beta to match perplexity
    const P = new Float64Array(N * N);
    const target = Math.log(Math.min(perplexity, (N - 1) / 3));
    for (let i = 0; i < N; i++) {
      let beta = 1, lo = -Infinity, hi = Infinity;
      for (let t = 0; t < 50; t++) {
        let sum = 0, H = 0;
        for (let j = 0; j < N; j++) { if (j === i) continue; const p = Math.exp(-D[i * N + j] * beta); P[i * N + j] = p; sum += p; }
        sum = sum || 1e-12;
        for (let j = 0; j < N; j++) { if (j === i) continue; const p = P[i * N + j] / sum; P[i * N + j] = p; if (p > 1e-12) H -= p * Math.log(p); }
        if (Math.abs(H - target) < 1e-5) break;
        if (H > target) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
        else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
      }
    }
    // symmetrize
    this.P = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) this.P[i * N + j] = Math.max((P[i * N + j] + P[j * N + i]) / (2 * N), 1e-12);
    let s = seed >>> 0 || 1;
    const rand = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
    const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
    this.Y = Array.from({ length: N }, () => [gauss() * 1e-4, gauss() * 1e-4]);
    this.gains = Array.from({ length: N }, () => [1, 1]);
    this.vel = Array.from({ length: N }, () => [0, 0]);
  }

  step(n = 1) {
    const { N, P, Y } = this;
    const Q = new Float64Array(N * N);
    for (let k = 0; k < n; k++, this.iter++) {
      const exaggeration = this.iter < 100 ? 4 : 1;
      const momentum = this.iter < 250 ? 0.5 : 0.8;
      let sumQ = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
        const q = 1 / (1 + dx * dx + dy * dy);
        Q[i * N + j] = Q[j * N + i] = q; sumQ += 2 * q;
      }
      const grads = new Array(N);
      for (let i = 0; i < N; i++) {
        let gx = 0, gy = 0;
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const q = Q[i * N + j];
          const m = (exaggeration * P[i * N + j] - q / sumQ) * q;
          gx += m * (Y[i][0] - Y[j][0]); gy += m * (Y[i][1] - Y[j][1]);
        }
        grads[i] = [4 * gx, 4 * gy];
      }
      for (let i = 0; i < N; i++) {
        const g = grads[i];
        for (let d = 0; d < 2; d++) {
          const gain = this.gains[i][d] = Math.max(0.01, Math.sign(g[d]) === Math.sign(this.vel[i][d]) ? this.gains[i][d] * 0.8 : this.gains[i][d] + 0.2);
          this.vel[i][d] = momentum * this.vel[i][d] - this.lr * gain * g[d];
          Y[i][d] += this.vel[i][d];
        }
      }
      // re-center
      let mx = 0, my = 0;
      for (const p of Y) { mx += p[0] / N; my += p[1] / N; }
      for (const p of Y) { p[0] -= mx; p[1] -= my; }
    }
    return Y;
  }
}
