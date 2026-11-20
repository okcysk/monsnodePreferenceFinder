import { normCdf, logNormCdf, millsRatio, cholesky, rankingToPairs, GPPL } from './gppl.js';
import { computeRound } from './engine.js';
import { makeRng, nextBeta } from './acquisition.js';
import { pcaProject } from './pca.js';

const out = document.getElementById('out');
const results = [];
function log(line, cls = '') { const s = document.createElement('span'); s.className = cls; s.textContent = line + '\n'; out.append(s); }
function check(name, cond, detail = '') { results.push({ name, pass: !!cond }); log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`, cond ? 'ok' : 'ng'); }
const near = (a, b, eps) => Math.abs(a - b) < eps;

// --- unit tests ---
check('Φ(0)=0.5', near(normCdf(0), 0.5, 1e-7));
check('Φ(1.96)≈0.975', near(normCdf(1.96), 0.9750021, 1e-6));
check('logΦ(-10) finite & close', near(logNormCdf(-10), -53.231285, 1e-3), String(logNormCdf(-10)));
check('Mills ratio large-negative ≈ -z', near(millsRatio(-20), 20.0499, 1e-2), String(millsRatio(-20)));
{
  const A = [4, 2, 2, 3]; const L = cholesky(A, 2);
  check('Cholesky 2×2', near(L[0], 2, 1e-12) && near(L[2], 1, 1e-12) && near(L[3], Math.SQRT2, 1e-12));
}
{
  const p = rankingToPairs(['a', 'b', 'c', 'd']);
  check('ranking → 6 pairs', p.length === 6 && p[0].join() === 'a,b' && p[5].join() === 'c,d', JSON.stringify(p));
}
{
  const X = [[0], [1], [2]];
  const m = new GPPL({ ell: 1 }).fit(X, [[2, 0], [2, 1], [1, 0]]);
  const { mu, variance } = m.predict(X);
  check('posterior order follows comparisons', mu[2] > mu[1] && mu[1] > mu[0], Array.from(mu).map(v => v.toFixed(3)).join(' '));
  const far = m.predict([[50]]);
  check('variance shrinks near data, ≈ prior far away', variance[1] < 0.9 && near(far.variance[0], 1, 1e-6));
}
{
  const rng = makeRng(3);
  const vecs = Array.from({ length: 40 }, () => Array.from({ length: 10 }, () => rng()));
  const P = pcaProject(vecs, 9);
  // Full-rank PCA (k = D-1 is close): pairwise distances approximately preserved
  let maxErr = 0;
  const d = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
  for (let i = 0; i < 10; i++) maxErr = Math.max(maxErr, Math.abs(d(vecs[i], vecs[i + 1]) - d(P[i], P[i + 1])));
  check('PCA roughly preserves distances', maxErr < 0.25, maxErr.toFixed(3));
}

// --- synthetic user simulation ---
function simulate(seed, { N = 300, D = 32, rounds = 30, noise = 0.05 } = {}) {
  const rng = makeRng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
  const embeddings = new Map();
  for (let i = 0; i < N; i++) {
    const v = Array.from({ length: D }, gauss); const n = Math.hypot(...v);
    embeddings.set('v' + i, Float32Array.from(v.map(x => x / n)));
  }
  // Hidden utility: smooth function = sum of a few RBF bumps
  const centers = Array.from({ length: 3 }, () => embeddings.get('v' + Math.floor(rng() * N)));
  const weights = [1, 0.7, 0.5];
  const g = id => { const x = embeddings.get(id); return centers.reduce((s, c, k) => s + weights[k] * Math.exp(-x.reduce((a, v, i) => a + (v - c[i]) ** 2, 0) / 0.8), 0); };
  const ordered = [...embeddings.keys()].sort((a, b) => g(b) - g(a));
  const top5pct = new Set(ordered.slice(0, Math.ceil(N * 0.05)));

  const pairs = []; const shown = new Set(); let beta = 2;
  let hitRound = null; let best = null;
  for (let r = 1; r <= rounds; r++) {
    const res = computeRound({ embeddings, pairs, shown, beta, pcaDim: 24, rng });
    best = res.best;
    if (best && top5pct.has(best) && hitRound === null) hitRound = r;
    res.selection.forEach(id => shown.add(id));
    const order = res.selection.slice().sort((a, b) => (g(b) + noise * gauss()) - (g(a) + noise * gauss()));
    pairs.push(...rankingToPairs(order));
    beta = nextBeta(beta);
  }
  const final = computeRound({ embeddings, pairs, shown, beta, pcaDim: 24, rng });
  return { hitRound, finalBest: final.best, rankOfFinal: ordered.indexOf(final.best) + 1, shown: shown.size };
}

const t0 = performance.now();
const sims = [1, 2, 3, 4, 5].map(seed => simulate(seed));
const ms = performance.now() - t0;
sims.forEach((s, i) => log(`  sim${i + 1}: top5% reached at round ${s.hitRound ?? '—'}, final best true rank ${s.rankOfFinal}/300, shown ${s.shown}`));
const hits = sims.filter(s => s.hitRound !== null && s.hitRound <= 30).length;
check('synthetic user: best reaches top 5% within 30 rounds (≥4/5 runs)', hits >= 4, `${hits}/5, ${(ms / 5).toFixed(0)} ms per 30-round run`);

const failed = results.filter(r => !r.pass).length;
log(`\n${results.length - failed}/${results.length} passed`, failed ? 'ng' : 'ok');
window.__testResult = { passed: results.length - failed, failed, sims };
