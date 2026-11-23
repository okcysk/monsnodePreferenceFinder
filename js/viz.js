// Embedding visualisation: 2-D map of the pool (t-SNE or PCA) drawn with thumbnails,
// coloured by the GPPL posterior mean, plus 512-D nearest neighbours for any video.
import * as db from './db.js';
import { pcaProject } from './pca.js';
import { TSNE } from './tsne.js';
import { computeRound } from './engine.js';
import { resolveVideoUrl } from './scraper.js';
import { grabFrames } from './frames.js';
import { Embedder } from './embed.js';
import { DEFAULT_TAGS, scoreTags, scoreText } from './tags.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const ui = { proj: $('proj'), color: $('color'), size: $('size'), fit: $('fit'), status: $('status'), meter: $('meter-fill'), legend: $('legend'), tip: $('tip'), panel: $('panel'),
  tagEdit: $('tag-edit'), textQuery: $('text-query'), tagEditor: $('tag-editor'), tagRows: $('tag-rows') };

let items = [];        // [{id, video, vec, pca48, x, y, img, mu, sigma, shown}]
let view = { scale: 1, tx: 0, ty: 0 };
let hover = null, selected = null, tsneRun = 0;
const placeholderHTML = ui.panel.innerHTML;   // restored when the selection is cleared
const frameCache = new Map();                  // id → Promise<Blob[]> (frames are fetched on demand)
let frameURLs = [];
let muMax = 1;
// Zero-shot tags / text search (CLIP text tower, loaded on first use)
let tags = DEFAULT_TAGS.map(t => ({ ...t }));
let tagScores = [];        // per tag: Float32Array aligned with items (probability 0–1)
let search = null;         // { query, score: Float32Array (cosine), pct: Float32Array (rank 0–1) }
let textEmbedder = null;
const getTextEmbedder = () => textEmbedder ??= new Embedder({
  preload: false,
  onModelProgress: p => setStatus(`テキストモデルを読み込み中… ${Math.round(p.progress ?? 0)}%`, (p.progress ?? 0) / 100),
});
const embedTexts = texts => getTextEmbedder().embedTexts(texts);

function setStatus(t, f = null) { ui.status.textContent = t; if (f !== null) ui.meter.style.width = `${Math.round(f * 100)}%`; }

async function load() {
  const videos = new Map((await db.getAll('videos')).map(v => [v.id, v]));
  const session = await db.kvGet('session') || { pairs: [], shown: [], broken: [] };
  const broken = new Set(session.broken || []);
  const settings = await db.kvGet('settings');
  const mode = settings?.embedMode === 'frames' ? 'frames' : 'thumb';
  const gpu = settings?.clipDevice === 'webgpu';
  const store = (mode === 'frames' ? 'frameEmbeddings' : 'embeddings') + (gpu ? 'Gpu' : '');
  const embs = (await db.getAll(store)).filter(e => videos.has(e.id) && !broken.has(e.id));
  document.querySelector('.brand small').textContent = `CLIP 512D (${mode === 'frames' ? 'サムネ＋4フレーム' : 'サムネ'} · ${gpu ? 'WebGPU' : 'WASM'}) → PCA 48D → 2D`;
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
  items.forEach((it, i) => { it.idx = i; });
  muMax = Math.max(1e-6, ...items.map(it => Math.abs(it.mu ?? 0)));
  setStatus(`${items.length} 本 · 512次元 → PCA ${pca48[0].length}次元 · 比較 ${pairs.length} 件`, 1);
  window.__viz = { count: items.length, dim: embs[0].vec.length, pcaDim: pca48[0].length, pairs: pairs.length };
  layout();
  const savedTags = await db.kvGet('tags');
  if (Array.isArray(savedTags) && savedTags.length) tags = savedTags;
  computeTags();
}

// ---------- tags / text search ----------
async function computeTags() {
  refreshColorOptions();
  if (!tags.length) { tagScores = []; draw(); return; }
  try {
    const t0 = performance.now();
    tagScores = await scoreTags(tags, items.map(it => it.vec), embedTexts);
    setStatus(`タグを計算しました（${tags.length} 個 · ${Math.round(performance.now() - t0)} ms）`, 1);
    window.__vizTags = tags.map((t, k) => ({ name: t.name, count: tagScores[k] ? tagScores[k].filter(p => p >= 0.5).length : null }));
  } catch (error) {
    setStatus(`タグを計算できませんでした: ${error.message}`);
  }
  refreshColorOptions();
  if (selected) fillChips(selected.it);
  draw();
}

async function runSearch(query) {
  query = query.trim();
  if (!query) { search = null; refreshColorOptions(); draw(); return; }
  try {
    const score = await scoreText(query, items.map(it => it.vec), embedTexts);
    const order = [...score.keys()].sort((a, b) => score[a] - score[b]);
    const pct = new Float32Array(score.length);
    order.forEach((i, r) => { pct[i] = r / Math.max(1, order.length - 1); });
    search = { query, score, pct };
    refreshColorOptions();
    ui.color.value = 'search';
    setStatus(`「${query}」に近い動画を強調しました（上位10%）`, 1);
  } catch (error) {
    setStatus(`テキスト検索に失敗しました: ${error.message}`);
  }
  draw();
}

function refreshColorOptions() {
  const current = ui.color.value;
  const base = [['mu', '好き度 μ'], ['shown', '表示済み'], ['none', 'なし']];
  const extra = tags.map((t, k) => [`tag:${k}`, `タグ: ${t.name}${tagScores[k] ? '' : '（計算中）'}`]);
  if (search) extra.push(['search', `テキスト: ${search.query}`]);
  ui.color.replaceChildren(...[...base, ...extra].map(([v, label]) => new Option(label, v)));
  ui.color.value = [...base, ...extra].some(([v]) => v === current) ? current : 'mu';
}

/** 0–1 score of an item for the current colour mode, or null when the mode is not score-based. */
function scoreOf(it, mode) {
  if (mode.startsWith('tag:')) return tagScores[Number(mode.slice(4))]?.[it.idx] ?? null;
  if (mode === 'search' && search) return search.pct[it.idx];
  return null;
}
const scoreThreshold = mode => (mode === 'search' ? 0.9 : 0.5);

function scoreColor(p) {
  const lerp = (a, b) => Math.round(a + (b - a) * p);
  return `rgb(${lerp(42, 196)},${lerp(44, 242)},${lerp(51, 106)})`;
}

function fillChips(it) {
  const box = ui.panel.querySelector('.tag-chips');
  if (!box) return;
  box.replaceChildren(...tags.map((t, k) => {
    const p = tagScores[k]?.[it.idx];
    const chip = document.createElement('span');
    chip.className = `chip${p >= 0.5 ? ' on' : ''}`;
    chip.textContent = p === undefined ? `${t.name} …` : `${t.name} ${Math.round(p * 100)}%`;
    chip.title = `${t.pos}\n vs \n${t.neg}`;
    return chip;
  }));
}

// tag editor
function renderTagRows(list) {
  ui.tagRows.replaceChildren(...list.map(t => {
    const row = document.createElement('div');
    row.className = 'tag-row';
    row.innerHTML = `<input class="tn" placeholder="タグ名（例: アニメ）"><input class="tp" placeholder="当てはまる説明（英語、| で複数）"><input class="tg" placeholder="対比する説明（英語、| で複数）"><button type="button" class="ghost del" title="削除">×</button>`;
    row.querySelector('.tn').value = t.name; row.querySelector('.tp').value = t.pos; row.querySelector('.tg').value = t.neg;
    row.querySelector('.del').addEventListener('click', () => row.remove());
    return row;
  }));
}
const readTagRows = () => [...ui.tagRows.querySelectorAll('.tag-row')]
  .map(r => ({ name: r.querySelector('.tn').value.trim(), pos: r.querySelector('.tp').value.trim(), neg: r.querySelector('.tg').value.trim() }))
  .filter(t => t.name && t.pos && t.neg);

ui.tagEdit.addEventListener('click', () => { renderTagRows(tags); ui.tagEditor.hidden = false; });
ui.tagEditor.addEventListener('click', async e => {
  const b = e.target.closest('button');
  if (e.target === ui.tagEditor || b?.classList.contains('close')) ui.tagEditor.hidden = true;
  else if (b?.classList.contains('add')) { renderTagRows([...readTagRows(), { name: '', pos: '', neg: '' }]); ui.tagRows.lastElementChild?.querySelector('.tn').focus(); }
  else if (b?.classList.contains('defaults')) renderTagRows(DEFAULT_TAGS);
  else if (b?.classList.contains('save')) {
    tags = readTagRows();
    await db.kvSet('tags', tags);
    ui.tagEditor.hidden = true;
    tagScores = [];
    computeTags();
  }
});
ui.textQuery.addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(ui.textQuery.value); });
ui.textQuery.addEventListener('search', () => { if (!ui.textQuery.value) runSearch(''); });

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
  // draw low scores first so the interesting ones sit on top
  const scored = scoreOf(items[0] ?? {}, mode) !== null;
  const order = items.slice().sort(scored ? (a, b) => scoreOf(a, mode) - scoreOf(b, mode) : (a, b) => (a.mu ?? -1e9) - (b.mu ?? -1e9));
  for (const it of order) {
    const [sx, sy] = toScreen(it);
    if (sx < -size || sy < -size || sx > w + size || sy > h + size) continue;
    const bw = size, bh = size * 0.66;
    const score = scoreOf(it, mode);
    const dimByScore = score !== null && score < scoreThreshold(mode) && it !== selected?.it;
    ctx.globalAlpha = (neighbours && !neighbours.has(it.id) && it !== selected.it) ? 0.25 : dimByScore ? 0.3 : 1;
    let border = '#2a2c33';
    if (score !== null) border = scoreColor(score);
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
  } else if (mode.startsWith('tag:')) {
    const k = Number(mode.slice(4)), t = tags[k], s = tagScores[k];
    ui.legend.innerHTML = s
      ? `「${escapeHTML(t.name)}」らしさ（CLIP ゼロショット・目安）<div class="bar-grad tag-grad"></div><div class="row"><span>0%</span><span>50%（未満は薄く）</span><span>100%</span></div>該当 ${s.filter(p => p >= 0.5).length} / ${s.length} 本`
      : 'タグを計算中…（初回はテキストモデル約65MBを読み込みます）';
  } else if (mode === 'search' && search) {
    ui.legend.innerHTML = `「${escapeHTML(search.query)}」との類似度<div class="bar-grad tag-grad"></div><div class="row"><span>低</span><span>上位10%を強調</span><span>高</span></div>`;
  } else ui.legend.innerHTML = '';
}

function escapeHTML(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

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

function clearSelection() {
  if (!selected) return;
  selected = null;
  releaseFrames();
  ui.panel.innerHTML = placeholderHTML;
  draw();
}

function releaseFrames() { frameURLs.forEach(u => URL.revokeObjectURL(u)); frameURLs = []; }

const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

/** Show the 4 frames at the same positions the 'サムネ＋4フレーム' embedding uses (fetched on demand, cached). */
async function showFrames(it) {
  const box = ui.panel.querySelector('.frames');
  box.innerHTML = '<p class="frames-note">フレームを読み込み中…</p>';
  if (!frameCache.has(it.id)) {
    const p = resolveVideoUrl(it.video).then(url => grabFrames(url, { count: 4, size: 320 }));
    frameCache.set(it.id, p);
    p.catch(() => frameCache.delete(it.id));
  }
  try {
    const blobs = await frameCache.get(it.id);
    if (selected?.it !== it) return;            // another video was selected meanwhile
    releaseFrames();
    frameURLs = blobs.map(b => URL.createObjectURL(b));
    box.replaceChildren(...blobs.map((b, i) => {
      const fig = document.createElement('figure');
      fig.innerHTML = '<img alt=""><figcaption></figcaption>';
      fig.querySelector('img').src = frameURLs[i];
      fig.querySelector('figcaption').textContent = Number.isFinite(b.time) ? fmtTime(b.time) : `#${i + 1}`;
      return fig;
    }));
  } catch (error) {
    if (selected?.it !== it) return;
    box.innerHTML = `<p class="frames-note">フレームを読み込めませんでした（${error.message}）</p>`;
  }
}

function renderPanel() {
  const { it, nn, all } = selected;
  const fmt = (m, s) => m === null ? 'μ —（未学習）' : `μ ${m.toFixed(2)} ± ${s.toFixed(2)}`;
  ui.panel.innerHTML = `
    <div class="panel-head"><button type="button" class="ghost deselect" title="選択解除 (Esc)">選択解除 ×</button></div>
    <div class="sel"><img alt=""><div class="t"></div><div class="s"></div><div class="tag-chips"></div><a target="_blank" rel="noopener">Open ↗</a></div>
    <h3 style="margin-top:14px">動画フレーム（4区間の中央）</h3>
    <div class="frames"></div>
    <h3 style="margin-top:14px">512次元空間の近傍 Top 12（コサイン類似度）</h3>
    <div class="nn"></div>
    <div class="hist"><h3>この動画と全候補の類似度分布</h3><canvas width="600" height="140"></canvas><p></p></div>`;
  ui.panel.querySelector('.sel img').src = it.video.thumbnail;
  ui.panel.querySelector('.sel .t').textContent = it.video.title;
  ui.panel.querySelector('.sel .s').textContent = `${fmt(it.mu, it.sigma)} · ${it.shown ? '表示済み' : '未表示'}`;
  ui.panel.querySelector('.sel a').href = it.video.pageUrl;
  ui.panel.querySelector('.deselect').addEventListener('click', clearSelection);
  fillChips(it);
  showFrames(it);
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
  if (drag && !drag.moved) { const it = pick(e.offsetX, e.offsetY); if (it) select(it); else clearSelection(); }
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

addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if (!ui.tagEditor.hidden) { ui.tagEditor.hidden = true; return; }
  if (e.target instanceof HTMLInputElement) return;
  clearSelection();
});
ui.proj.addEventListener('change', layout);
ui.color.addEventListener('change', draw);
ui.size.addEventListener('input', draw);
ui.fit.addEventListener('click', () => { fitView(); draw(); });
addEventListener('resize', draw);

load().catch(err => setStatus(`エラー: ${err.message}`));
