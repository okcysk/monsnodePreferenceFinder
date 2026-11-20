// Embedding visualisation: 2-D map of the pool (t-SNE or PCA) drawn with thumbnails,
// coloured by the GPPL posterior mean, plus 512-D nearest neighbours for any video.
import * as db from './db.js';
import { pcaProject } from './pca.js';
import { TSNE } from './tsne.js';
import { computeRound } from './engine.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const ui = { proj: $('proj'), color: $('color'), size: $('size'), fit: $('fit'), status: $('status'), meter: $('meter-fill'), legend: $('legend'), tip: $('tip'), panel: $('panel') };

let items = [];        // [{id, video, vec, pca48, x, y, img, mu, sigma, shown}]
let view = { scale: 1, tx: 0, ty: 0 };
let hover = null, selected = null, tsneRun = 0;
let muMax = 1;

function setStatus(t, f = null) { ui.status.textContent = t; if (f !== null) ui.meter.style.width = `${Math.round(f * 100)}%`; }

async function load() {
  const videos = new Map((await db.getAll('videos')).map(v => [v.id, v]));
  const session = await db.kvGet('session') || { pairs: [], shown: [], broken: [] };
  const broken = new Set(session.broken || []);
  const mode = (await db.kvGet('settings'))?.embedMode === 'frames' ? 'frames' : 'thumb';
  const embs = (await db.getAll(mode === 'frames' ? 'frameEmbeddings' : 'embeddings')).filter(e => videos.has(e.id) && !broken.has(e.id));
  document.querySelector('.brand small').textContent = `CLIP 512D (${mode === 'frames' ? 'サムネ＋4フレーム' : 'サムネ'}) → PCA 48D → 2D`;
  if (embs.length < 3) { setStatus('埋め込み済みの動画がまだありません（3本以上必要）'); return; }

  const pool = new Map(embs.map(e => [e.id, e.vec]));
  const shown = new Set(session.shown || []);
  let mu = null, sigma = null;
  const pairs = (session.pairs || []).filter(([w, l]) => pool.has(w) && pool.has(l));
  if (pairs.length) ({ mu, sigma } = computeRound({ embeddings: pool, pairs, shown, beta: 0 }));

  const pca48 = pcaProject(embs.map(e => e.vec), 48);
  items = embs.map((e, i) => {
    const img = new Image();
    img.src = videos.get(e.id).thumbnail || '';
    img.onload = draw;
    return { id: e.id, video: videos.get(e.id), vec: e.vec, pca48: pca48[i], x: 0, y: 0, img, mu: mu?.get(e.id) ?? null, sigma: sigma?.get(e.id) ?? null, shown: shown.has(e.id) };
  });
  muMax = Math.max(1e-6, ...items.map(it => Math.abs(it.mu ?? 0)));
  setStatus(`${items.length} 本 · 512次元 → PCA ${pca48[0].length}次元 · 比較 ${pairs.length} 件`, 1);
  window.__viz = { count: items.length, dim: embs[0].vec.length, pcaDim: pca48[0].length, pairs: pairs.length };
  layout();
}

// ---------- layout ----------
function layout() {
  const run = ++tsneRun;
  if (ui.proj.value === 'pca') {
    const p2 = pcaProject(items.map(it => it.pca48), 2);
    items.forEach((it, i) => { it.x = p2[i][0]; it.y = p2[i][1] ?? 0; });
    fitView(); draw();
    setStatus(`${items.length} 本 · 512次元 → PCA ${items[0].pca48.length}次元 → 第1・第2主成分`, 1);
    return;
  }
  const tsne = new TSNE(items.map(it => it.pca48), { perplexity: Math.min(30, Math.max(5, Math.floor(items.length / 4))) });
  const total = 600;
  const tick = () => {
    if (run !== tsneRun) return;
    tsne.step(10);
    tsne.Y.forEach((p, i) => { items[i].x = p[0]; items[i].y = p[1]; });
    if (tsne.iter <= 120 || tsne.iter % 50 === 0) fitView();
    draw();
    setStatus(`t-SNE 計算中 ${tsne.iter}/${total}`, tsne.iter / total);
    if (tsne.iter < total) requestAnimationFrame(tick);
    else { fitView(); draw(); setStatus(`${items.length} 本 · 512次元 → PCA ${items[0].pca48.length}次元 → t-SNE 2次元`, 1); window.__vizDone = true; }
  };
  tick();
}

function fitView() {
  const w = canvas.clientWidth, h = canvas.clientHeight, pad = Number(ui.size.value);
  const xs = items.map(i => i.x), ys = items.map(i => i.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const s = Math.min((w - 2 * pad) / ((x1 - x0) || 1), (h - 2 * pad) / ((y1 - y0) || 1));
  view = { scale: s, tx: w / 2 - s * (x0 + x1) / 2, ty: h / 2 - s * (y0 + y1) / 2 };
}
const toScreen = it => [it.x * view.scale + view.tx, it.y * view.scale + view.ty];

// ---------- drawing ----------
function muColor(mu) {
  const t = Math.max(-1, Math.min(1, mu / muMax));
  const lerp = (a, b, u) => Math.round(a + (b - a) * u);
  const [c0, c1] = t < 0 ? [[85, 88, 98], [61, 123, 255]] : [[85, 88, 98], [255, 154, 61]];
  const u = Math.abs(t);
  return `rgb(${lerp(c0[0], c1[0], u)},${lerp(c0[1], c1[1], u)},${lerp(c0[2], c1[2], u)})`;
}

function draw() {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const size = Number(ui.size.value), mode = ui.color.value;
  const neighbours = selected ? new Set(selected.nn.map(n => n.it.id)) : null;
  // draw low-μ first so favourites sit on top
  const order = items.slice().sort((a, b) => (a.mu ?? -1e9) - (b.mu ?? -1e9));
  for (const it of order) {
    const [sx, sy] = toScreen(it);
    if (sx < -size || sy < -size || sx > w + size || sy > h + size) continue;
    const bw = size, bh = size * 0.66;
    ctx.globalAlpha = neighbours && !neighbours.has(it.id) && it !== selected.it ? 0.25 : 1;
    let border = '#2a2c33';
    if (mode === 'mu' && it.mu !== null) border = muColor(it.mu);
    if (mode === 'shown') border = it.shown ? '#c4f26a' : '#2a2c33';
    ctx.fillStyle = border;
    ctx.fillRect(sx - bw / 2 - 2, sy - bh / 2 - 2, bw + 4, bh + 4);
    if (it.img.complete && it.img.naturalWidth) ctx.drawImage(it.img, sx - bw / 2, sy - bh / 2, bw, bh);
    else { ctx.fillStyle = '#1a1d24'; ctx.fillRect(sx - bw / 2, sy - bh / 2, bw, bh); }
  }
  ctx.globalAlpha = 1;
  for (const it of [selected?.it, hover].filter(Boolean)) {
    const [sx, sy] = toScreen(it);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.strokeRect(sx - size / 2 - 3, sy - size * 0.33 - 3, size + 6, size * 0.66 + 6);
  }
  drawLegend(mode);
}

function drawLegend(mode) {
  if (mode === 'mu') {
    const has = items.some(i => i.mu !== null);
    ui.legend.innerHTML = has
      ? `好き度 μ（GPPL 事後平均）<div class="bar-grad"></div><div class="row"><span>−${muMax.toFixed(2)} 嫌い</span><span>0</span><span>好き +${muMax.toFixed(2)}</span></div>`
      : 'まだ比較がないため μ はありません（1ラウンド以上回答すると色が付きます）';
  } else if (mode === 'shown') {
    ui.legend.innerHTML = `<span class="sw" style="background:#c4f26a"></span>表示済み ${items.filter(i => i.shown).length} 本　<span class="sw" style="background:#2a2c33"></span>未表示`;
  } else ui.legend.innerHTML = '';
}

// ---------- interaction ----------
function pick(mx, my) {
  const size = Number(ui.size.value);
  let best = null, bestD = Infinity;
  for (const it of items) {
    const [sx, sy] = toScreen(it);
    if (Math.abs(mx - sx) <= size / 2 && Math.abs(my - sy) <= size * 0.33) {
      const d = (mx - sx) ** 2 + (my - sy) ** 2;
      if (d < bestD) { bestD = d; best = it; }
    }
  }
  return best;
}

function cosine(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; } // vectors are L2-normalised

function select(it) {
  const sims = items.filter(o => o !== it).map(o => ({ it: o, sim: cosine(it.vec, o.vec) })).sort((a, b) => b.sim - a.sim);
  selected = { it, nn: sims.slice(0, 12), all: sims.map(s => s.sim) };
  renderPanel();
  draw();
}

function renderPanel() {
  const { it, nn, all } = selected;
  const fmt = (m, s) => m === null ? 'μ —（未学習）' : `μ ${m.toFixed(2)} ± ${s.toFixed(2)}`;
  ui.panel.innerHTML = `
    <div class="sel"><img alt=""><div class="t"></div><div class="s"></div><a target="_blank" rel="noopener">Open ↗</a></div>
    <h3 style="margin-top:14px">512次元空間の近傍 Top 12（コサイン類似度）</h3>
    <div class="nn"></div>
    <div class="hist"><h3>この動画と全候補の類似度分布</h3><canvas width="600" height="140"></canvas><p></p></div>`;
  ui.panel.querySelector('.sel img').src = it.video.thumbnail;
  ui.panel.querySelector('.sel .t').textContent = it.video.title;
  ui.panel.querySelector('.sel .s').textContent = `${fmt(it.mu, it.sigma)} · ${it.shown ? '表示済み' : '未表示'}`;
  ui.panel.querySelector('.sel a').href = it.video.pageUrl;
  ui.panel.querySelector('.nn').replaceChildren(...nn.map(({ it: o, sim }) => {
    const b = document.createElement('button');
    b.innerHTML = '<img alt=""><span><b></b><em></em></span>';
    b.querySelector('img').src = o.video.thumbnail;
    b.querySelector('b').textContent = sim.toFixed(3);
    b.querySelector('em').textContent = o.mu === null ? '' : `μ ${o.mu.toFixed(2)}`;
    b.title = o.video.title;
    b.addEventListener('click', () => select(o));
    return b;
  }));
  // similarity histogram
  const c = ui.panel.querySelector('.hist canvas'), g = c.getContext('2d');
  const bins = 40, lo = Math.min(...all), hi = Math.max(...all);
  const counts = new Array(bins).fill(0);
  all.forEach(s => counts[Math.min(bins - 1, Math.floor((s - lo) / ((hi - lo) || 1) * bins))]++);
  const max = Math.max(...counts);
  g.clearRect(0, 0, c.width, c.height);
  const nnMin = nn[nn.length - 1]?.sim ?? hi;
  counts.forEach((n, i) => {
    const x = i / bins * c.width, bw = c.width / bins - 2, bh = n / max * (c.height - 10);
    g.fillStyle = lo + (i + 0.5) / bins * (hi - lo) >= nnMin ? '#c4f26a' : '#555862';
    g.fillRect(x, c.height - bh, bw, bh);
  });
  const mean = all.reduce((s, v) => s + v, 0) / all.length;
  ui.panel.querySelector('.hist p').textContent = `範囲 ${lo.toFixed(3)} 〜 ${hi.toFixed(3)} · 平均 ${mean.toFixed(3)}（緑 = 上位12の近傍）`;
}

let drag = null;
canvas.addEventListener('pointerdown', e => { drag = { x: e.offsetX, y: e.offsetY, tx: view.tx, ty: view.ty, moved: false }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointermove', e => {
  if (drag) {
    const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; canvas.classList.add('dragging'); }
    if (drag.moved) { view.tx = drag.tx + dx; view.ty = drag.ty + dy; draw(); return; }
  }
  const it = pick(e.offsetX, e.offsetY);
  if (it !== hover) { hover = it; draw(); }
  if (it) {
    ui.tip.hidden = false;
    ui.tip.querySelector('img').src = it.video.thumbnail;
    ui.tip.querySelector('.tip-t').textContent = it.video.title;
    ui.tip.querySelector('.tip-s').textContent = it.mu === null ? (it.shown ? '表示済み' : '未表示') : `μ ${it.mu.toFixed(2)} ± ${it.sigma.toFixed(2)} · ${it.shown ? '表示済み' : '未表示'}`;
    const left = e.offsetX + 240 > canvas.clientWidth ? e.offsetX - 236 : e.offsetX + 16;
    ui.tip.style.left = `${left}px`; ui.tip.style.top = `${Math.min(e.offsetY + 16, canvas.clientHeight - 200)}px`;
  } else ui.tip.hidden = true;
});
canvas.addEventListener('pointerup', e => {
  canvas.classList.remove('dragging');
  if (drag && !drag.moved) { const it = pick(e.offsetX, e.offsetY); if (it) select(it); }
  drag = null;
});
canvas.addEventListener('pointerleave', () => { hover = null; ui.tip.hidden = true; draw(); });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  view.tx = e.offsetX - (e.offsetX - view.tx) * f;
  view.ty = e.offsetY - (e.offsetY - view.ty) * f;
  view.scale *= f;
  draw();
}, { passive: false });

ui.proj.addEventListener('change', layout);
ui.color.addEventListener('change', draw);
ui.size.addEventListener('input', draw);
ui.fit.addEventListener('click', () => { fitView(); draw(); });
addEventListener('resize', draw);

load().catch(err => setStatus(`エラー: ${err.message}`));
