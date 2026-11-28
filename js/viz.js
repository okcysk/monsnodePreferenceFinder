// Embedding visualisation: 2-D map of the pool (t-SNE or PCA) drawn with thumbnails,
// coloured by the GPPL posterior mean, plus 512-D nearest neighbours for any video.
import * as db from './db.js';
import { pcaProject } from './pca.js';
import { TSNE } from './tsne.js';
import { computeRound } from './engine.js';
import { popularityScores } from './popularity.js';
import * as scraper from './scraper.js';   // web: has mediaProxyUrl (Worker fallback); extension: direct only
const { resolveVideoUrl } = scraper;
import { grabFrames } from './frames.js';
import { loadPrecomputed, dequantize } from './precomputed.js';
import { Embedder } from './embed.js';
import { DEFAULT_TAGS, DEFAULT_TAGS_VERSION, mergeNewDefaults, scoreTags, scoreText } from './tags.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const ui = { proj: $('proj'), color: $('color'), size: $('size'), fit: $('fit'), status: $('status'), meter: $('meter-fill'), legend: $('legend'), tip: $('tip'), panel: $('panel'),
  tagEdit: $('tag-edit'), textQuery: $('text-query'), tagEditor: $('tag-editor'), tagRows: $('tag-rows'), tagLabels: $('tag-labels'),
  quad: $('quad') };

let items = [];        // [{id, video, vec, pca48, x, y, img, mu, sigma, shown}]
let view = { scale: 1, tx: 0, ty: 0 };
let hover = null, selected = null, tsneRun = 0;
const placeholderHTML = ui.panel.innerHTML;   // restored when the selection is cleared
const frameCache = new Map();                  // id → Promise<Blob[]> (frames are fetched on demand)
let frameURLs = [];
let muMax = 1;
let popInfo = null;     // {auc, shrink, known} from popularity.js, null without ranking data
// Zero-shot tags / text search (CLIP text tower, loaded on first use)
let tags = DEFAULT_TAGS.map(({ name, pos, neg }) => ({ name, pos, neg }));
let tagScores = [];        // per tag: Float32Array aligned with items (probability 0–1)
let search = null;         // { query, score: Float32Array (cosine), pct: Float32Array (rank 0–1) }
let tagAnchors = [];       // per tag: index of its most representative item (or -1)
let searchAnchor = -1;
let simMatrix = null;      // N×N cosine similarities (512-D), built once per load
let labelHits = [];        // clickable label boxes from the last draw: {key, x0, y0, x1, y1}
let tagList = null;        // {key, name, ids: Set, prevColor} while the panel lists a tag's top videos
let textEmbedder = null;
let embedModel = 'clip';   // 'gemma' when the map shows EmbeddingGemma 2 vectors
// Tags and text search always use CLIP (the captions are tuned for it). With EmbeddingGemma 2 on the map they score
// each video's CLIP thumbnail vector, kept alongside: Gemma's own text side agreed only loosely with the tuned captions
// (correlation ≈ 0.46 on 56 videos) and its 50% point drifted per tag, so many tags matched none or nearly all videos.
let clipVecs = null;       // id → CLIP thumbnail vector (gemma mode only)
const getTextEmbedder = () => textEmbedder ??= new Embedder({
  preload: false,
  onModelProgress: p => setStatus(`テキストモデルを読み込み中… ${Math.round(p.progress ?? 0)}%`, (p.progress ?? 0) / 100),
});
/**
 * CLIP text embeddings: the default tags' captions come precomputed (GitHub Actions, same q8 model as the browser's
 * text embedder); only other captions (custom tags, text search) load the text model here.
 */
async function embedTexts(texts) {
  const pre = (await loadPrecomputed())?.clipText || {};
  const missing = [...new Set(texts.filter(t => !pre[t]))];
  const fresh = missing.length ? await getTextEmbedder().embedTexts(missing) : [];
  const got = new Map(missing.map((t, i) => [t, fresh[i]]));
  return texts.map(t => (pre[t] ? dequantize(...pre[t]) : got.get(t)));
}

function setStatus(t, f = null) { ui.status.textContent = t; if (f !== null) ui.meter.style.width = `${Math.round(f * 100)}%`; }

async function load() {
  const videos = new Map((await db.getAll('videos')).map(v => [v.id, v]));
  const session = await db.kvGet('session') || { pairs: [], shown: [], broken: [] };
  const broken = new Set(session.broken || []);
  const settings = await db.kvGet('settings');
  const mode = ['frames', 'gemma'].includes(settings?.embedMode) ? settings.embedMode : 'thumb';
  const gpu = settings?.clipDevice === 'webgpu';
  embedModel = mode === 'gemma' ? 'gemma' : 'clip';
  clipVecs = mode === 'gemma' ? new Map((await db.getAll('embeddings' + (gpu ? 'Gpu' : ''))).map(e => [e.id, e.vec])) : null;
  if (clipVecs) {
    const pre = await loadPrecomputed();
    if (pre?.tsne?.ids) precomputedTsne = new Map(pre.tsne.ids.map((id, i) => [id, [pre.tsne.xy[2 * i], pre.tsne.xy[2 * i + 1]]]));
    // thumbnails the browser did not embed: the batch's CLIP vectors (GitHub Actions, q8)
    const clip = pre?.clip || {};
    for (const [id, [scale, b64]] of Object.entries(clip)) if (!clipVecs.has(id)) clipVecs.set(id, dequantize(scale, b64));
  }
  const store = ({ thumb: 'embeddings', frames: 'frameEmbeddings', gemma: 'gemmaEmbeddings' })[mode] + (gpu ? 'Gpu' : '');
  const embs = (await db.getAll(store)).filter(e => videos.has(e.id) && !broken.has(e.id));
  document.querySelector('.brand small').textContent = mode === 'gemma'
    ? `EmbeddingGemma 2 768D (動画4フレーム＋サムネ · ${gpu ? 'WebGPU' : 'WASM'}) → PCA 48D → 2D`
    : `CLIP 512D (${mode === 'frames' ? 'サムネ＋4フレーム' : 'サムネ'} · ${gpu ? 'WebGPU' : 'WASM'}) → PCA 48D → 2D`;
  if (embs.length < 3) { setStatus('埋め込み済みの動画がまだありません（3本以上必要）'); return; }

  const pool = new Map(embs.map(e => [e.id, e.vec]));
  const shown = new Set(session.shown || []);
  let mu = null, sigma = null;
  const pairs = (session.pairs || []).filter(([w, l]) => pool.has(w) && pool.has(l));
  const popularity = new Map(embs.filter(e => videos.get(e.id).pop).map(e => [e.id, videos.get(e.id).pop.s]));
  const popWeight = { off: 0, low: 0.3, mid: 0.6, high: 1 }[settings?.popLevel ?? 'mid'] ?? 0.6;
  const kernel = ['rbf', 'linear', 'quad'].includes(settings?.kernel) ? settings.kernel : 'rbf';
  if (pairs.length) ({ mu, sigma } = computeRound({ embeddings: pool, pairs, shown, beta: 0, popularity, popWeight, kernel }));

  const pca48 = pcaProject(embs.map(e => e.vec), 48);
  const pop = popularityScores(embs.map(e => e.id), new Map(embs.map((e, i) => [e.id, pca48[i]])), popularity);
  popInfo = pop && { auc: pop.auc, shrink: pop.shrink, known: pop.known };
  items = embs.map((e, i) => {
    const img = new Image();
    img.src = videos.get(e.id).thumbnail || '';
    img.onload = draw;
    return { id: e.id, video: videos.get(e.id), vec: e.vec, pca48: pca48[i], x: 0, y: 0, img, mu: mu?.get(e.id) ?? null, sigma: sigma?.get(e.id) ?? null, shown: shown.has(e.id),
      popScore: pop?.score.get(e.id) ?? null };
  });
  items.forEach((it, i) => { it.idx = i; });
  muMax = Math.max(1e-6, ...items.map(it => Math.abs(it.mu ?? 0)));
  setStatus(`${items.length} 本 · ${embedModel === 'gemma' ? 768 : 512}次元 → PCA ${pca48[0].length}次元 · 比較 ${pairs.length} 件`, 1);
  window.__vizItems = () => items;   // debug hook (positions, scores)
  window.__viz = { count: items.length, dim: embs[0].vec.length, pcaDim: pca48[0].length, pairs: pairs.length, pop: popInfo };
  refreshColorOptions();
  layout();
  const savedTags = await db.kvGet('tags');
  if (Array.isArray(savedTags) && savedTags.length) {
    tags = savedTags;
    const version = (await db.kvGet('tagsVersion')) ?? 1;
    if (version < DEFAULT_TAGS_VERSION) {
      ({ tags } = mergeNewDefaults(savedTags, version));
      await db.kvSet('tags', tags);
    }
  }
  await db.kvSet('tagsVersion', DEFAULT_TAGS_VERSION);
  computeTags();
}

// ---------- tags / text search ----------
/** Vectors to score captions against (CLIP), and which items have none (gemma mode, thumbnail not embedded). */
function tagInputs() {
  const vecs = items.map(it => (clipVecs ? clipVecs.get(it.id) : it.vec) ?? null);
  const dim = vecs.find(Boolean)?.length ?? 512;
  const zero = new Float32Array(dim);
  return { vecs: vecs.map(v => v ?? zero), missing: vecs.map(v => !v) };
}

async function computeTags() {
  refreshColorOptions();
  if (!tags.length) { tagScores = []; draw(); return; }
  try {
    const t0 = performance.now();
    const { vecs, missing } = tagInputs();
    tagScores = (await scoreTags(tags, vecs, embedTexts)).map(p => p && p.map((x, i) => (missing[i] ? 0 : x)));
    tagAnchors = tagScores.map(p => representative(p, 0.5));
    setStatus(`タグを計算しました（${tags.length} 個 · ${Math.round(performance.now() - t0)} ms）`, 1);
    window.__vizTags = tags.map((t, k) => ({ name: t.name, count: tagScores[k] ? tagScores[k].filter(p => p >= 0.5).length : null,
      anchor: tagAnchors[k] >= 0 ? { id: items[tagAnchors[k]].id, p: +tagScores[k][tagAnchors[k]].toFixed(3) } : null }));
  } catch (error) {
    setStatus(`タグを計算できませんでした: ${error.message}`);
  }
  refreshColorOptions();
  if (selected) fillChips(selected.it);
  draw();
}

async function runSearch(query) {
  query = query.trim();
  if (!query) { search = null; searchAnchor = -1; refreshColorOptions(); draw(); return; }
  try {
    const { vecs, missing } = tagInputs();
    const score = (await scoreText(query, vecs, embedTexts)).map((x, i) => (missing[i] ? -1 : x));
    const order = [...score.keys()].sort((a, b) => score[a] - score[b]);
    const pct = new Float32Array(score.length);
    order.forEach((i, r) => { pct[i] = r / Math.max(1, order.length - 1); });
    search = { query, score, pct };
    searchAnchor = representative(pct, 0.9);
    refreshColorOptions();
    ui.color.value = 'search';
    setStatus(`「${query}」に近い動画を強調しました（上位10%）`, 1);
  } catch (error) {
    setStatus(`テキスト検索に失敗しました: ${error.message}`);
  }
  draw();
}

/**
 * Most representative item for a 0–1 score: among items at or above the threshold, the one at the
 * centre of the densest group of high scorers in the 512-D space,
 *   argmax_i  w_i · Σ_j w_j · exp((cos_ij − 1) / τ),   w = max(0, p − threshold) / (1 − threshold)
 * so the label sits on the main cluster rather than on a lone outlier or in empty space.
 */
function representative(p, threshold, tau = 0.05) {
  const n = items.length;
  if (!simMatrix || simMatrix.length !== n * n) {
    simMatrix = new Float32Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      const c = cosine(items[i].vec, items[j].vec); simMatrix[i * n + j] = c; simMatrix[j * n + i] = c;
    }
  }
  const w = Array.from(p, v => Math.max(0, v - threshold) / (1 - threshold));
  let best = -1, bestScore = 0;
  for (let i = 0; i < n; i++) {
    if (!(p[i] >= threshold)) continue;
    let s = 0;
    for (let j = 0; j < n; j++) if (w[j]) s += w[j] * Math.exp((simMatrix[i * n + j] - 1) / tau);
    s *= Math.max(w[i], 1e-3);
    if (s > bestScore) { bestScore = s; best = i; }
  }
  return best;
}

/** One label per tag (and the text search), placed on its representative item. */
function drawTagLabels(mode, size) {
  if (!ui.tagLabels.checked) return;
  const labels = tags.map((t, k) => ({ key: `tag:${k}`, text: t.name, idx: tagAnchors[k] ?? -1, active: mode === `tag:${k}` }));
  if (search) labels.push({ key: 'search', text: `“${search.query}”`, idx: searchAnchor, active: mode === 'search' });
  const placed = [];
  labelHits = [];
  ctx.font = '700 12px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (const l of labels.filter(l => l.idx >= 0).sort((a, b) => a.active - b.active)) {
    const [sx, sy0] = toScreen(items[l.idx]);
    let sy = sy0 - size * 0.33 - 14;
    const tw = ctx.measureText(l.text).width + 14;
    while (placed.some(q => Math.abs(q.x - sx) < (q.w + tw) / 2 && Math.abs(q.y - sy) < 20)) sy -= 22;   // stack instead of overlapping
    placed.push({ x: sx, y: sy, w: tw });
    labelHits.push({ key: l.key, x0: sx - tw / 2, y0: sy - 10, x1: sx + tw / 2, y1: sy + 10 });
    ctx.strokeStyle = l.active ? '#c4f26a' : '#ffffff99'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx, sy + 10); ctx.lineTo(sx, sy0 - size * 0.33 - 2); ctx.stroke();
    ctx.fillStyle = l.active ? '#c4f26a' : '#111319e6';
    ctx.beginPath(); ctx.roundRect(sx - tw / 2, sy - 10, tw, 20, 10); ctx.fill();
    ctx.stroke();
    ctx.fillStyle = l.active ? '#111319' : '#f2f3f5';
    ctx.textAlign = 'center'; ctx.fillText(l.text, sx, sy + 0.5);
  }
  ctx.textAlign = 'start';
}

/** Key of the tag label under the pointer (labels drawn later are on top). */
function hitLabel(mx, my) {
  for (let i = labelHits.length - 1; i >= 0; i--) {
    const h = labelHits[i];
    if (mx >= h.x0 && mx <= h.x1 && my >= h.y0 && my <= h.y1) return h.key;
  }
  return null;
}

/** Panel: the videos that score highest for a tag ('tag:k') or the text search ('search'). */
function showTagList(key) {
  // Clicking the tag that is already selected (its list is open) deselects it.
  if (tagList?.key === key && !selected) { clearSelection(); return; }
  const isSearch = key === 'search';
  const k = isSearch ? -1 : Number(key.slice(4));
  const score = isSearch ? search?.pct : tagScores[k];
  if (!score) { setStatus('タグを計算中です。終わってからもう一度クリックしてください'); return; }
  const name = isSearch ? `“${search.query}”` : tags[k].name;
  const threshold = scoreThreshold(key);
  const ranked = items.slice().sort((a, b) => score[b.idx] - score[a.idx]);
  const hits = ranked.filter(it => score[it.idx] >= threshold).length;
  const top = ranked.slice(0, Math.min(60, Math.max(12, hits)));
  selected = null;
  releaseFrames();
  // remember the colour mode from before any tag was selected, to restore it on deselect
  const prevColor = tagList ? tagList.prevColor : ui.color.value;
  tagList = { key, name, ids: new Set(top.map(it => it.id)), prevColor };
  if ([...ui.color.options].some(o => o.value === key)) ui.color.value = key;
  ui.panel.innerHTML = `
    <div class="panel-head"><button type="button" class="ghost deselect" title="タグの選択を解除 (Esc・もう一度タグをクリック)">選択解除 ×</button></div>
    <h3 class="list-title"></h3>
    <p class="list-note"></p>
    <button type="button" class="play-quad">▶ 上位を4画面で再生</button>
    <div class="nn"></div>`;
  ui.panel.querySelector('.list-title').textContent = `${name} の上位 ${top.length} 本`;
  ui.panel.querySelector('.list-note').textContent = isSearch
    ? `文章との類似度が高い順（上位10% = ${hits} 本）。数字は全候補の中での位置（%）。クリックでその動画の詳細`
    : `「${name}」らしさが高い順（50%以上 = ${hits} / ${items.length} 本）。クリックでその動画の詳細`;
  ui.panel.querySelector('.deselect').addEventListener('click', clearSelection);
  ui.panel.querySelector('.play-quad').addEventListener('click', () => openQuad(top, `${name} の上位`, o => `${Math.round(score[o.idx] * 100)}%`));
  ui.panel.querySelector('.nn').replaceChildren(...top.map(it => {
    const b = document.createElement('button');
    b.innerHTML = '<img alt=""><span><b></b><em></em></span>';
    b.querySelector('img').src = it.video.thumbnail;
    b.querySelector('b').textContent = `${Math.round(score[it.idx] * 100)}%`;
    b.querySelector('em').textContent = it.mu === null ? (it.shown ? '表示済み' : '') : `μ ${it.mu.toFixed(2)}`;
    b.title = it.video.title;
    b.addEventListener('click', () => select(it));
    return b;
  }));
  ui.panel.scrollTop = 0;
  draw();
}

function refreshColorOptions() {
  const current = ui.color.value;
  const base = [['mu', '好き度 μ'], ['shown', '表示済み'], ...(popInfo ? [['pop', '人気（ランキング＋推定）']] : []), ['none', 'なし']];
  const extra = tags.map((t, k) => [`tag:${k}`, `タグ: ${t.name}${tagScores[k] ? '' : '（計算中）'}`]);
  if (search) extra.push(['search', `テキスト: ${search.query}`]);
  ui.color.replaceChildren(...[...base, ...extra].map(([v, label]) => new Option(label, v)));
  ui.color.value = [...base, ...extra].some(([v]) => v === current) ? current : 'mu';
}

/** 0–1 score of an item for the current colour mode, or null when the mode is not score-based. */
function scoreOf(it, mode) {
  if (mode.startsWith('tag:')) return tagScores[Number(mode.slice(4))]?.[it.idx] ?? null;
  if (mode === 'search' && search) return search.pct[it.idx];
  if (mode === 'pop') return it.popScore;
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
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.addEventListener('click', () => showTagList(`tag:${k}`));
    chip.className = `chip${p >= 0.5 ? ' on' : ''}`;
    chip.textContent = p === undefined ? `${t.name} …` : `${t.name} ${Math.round(p * 100)}%`;
    chip.title = `クリックで「${t.name}」の上位一覧\n${t.pos}\n vs \n${t.neg}`;
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
    setStatus(`${items.length} 本 · ${embedModel === 'gemma' ? 768 : 512}次元 → PCA ${items[0].pca48.length}次元 → 第1・第2主成分`, 1);
    return;
  }
  const pre = precomputedStart();
  if (pre && pre.covered >= items.length * 0.9) {
    // nearly all covered: show the batch's layout as it is
    items.forEach((it, i) => { [it.x, it.y] = pre.start[i]; });
    fitView(); draw();
    setStatus(`${items.length} 本 · 768次元 → PCA 48次元 → t-SNE 2次元（GitHub Actions で事前計算した配置${pre.covered < items.length ? `。${items.length - pre.covered} 本は近傍の隣` : ''}）`, 1);
    window.__vizDone = true;
    return;
  }
  const tsne = new TSNE(items.map(it => it.pca48), { perplexity: Math.min(30, Math.max(5, Math.floor(items.length / 4))) });
  let total = 600;
  if (pre) {
    // partly covered: start from the batch's layout and only settle it (no early exaggeration, 150 steps)
    tsne.Y = pre.start.map(([x, y]) => [x, y]);
    tsne.iter = 250;
    total = 400;
  }
  const tick = () => {
    if (run !== tsneRun) return;
    tsne.step(10);
    tsne.Y.forEach((p, i) => { items[i].x = p[0]; items[i].y = p[1]; });
    if (tsne.iter <= 120 || tsne.iter % 50 === 0) fitView();
    draw();
    setStatus(pre ? `t-SNE 調整中（事前計算の配置から） ${tsne.iter - 250}/${total - 250}` : `t-SNE 計算中 ${tsne.iter}/${total}`, pre ? (tsne.iter - 250) / (total - 250) : tsne.iter / total);
    if (tsne.iter < total) requestAnimationFrame(tick);
    else { fitView(); draw(); setStatus(`${items.length} 本 · ${embedModel === 'gemma' ? 768 : 512}次元 → PCA ${items[0].pca48.length}次元 → t-SNE 2次元`, 1); window.__vizDone = true; }
  };
  tick();
}

let precomputedTsne = null;   // id → [x, y] from the batch (gemma mode only)
/**
 * Gemma mode: start from the batch's t-SNE layout. Videos it does not cover start next to their nearest covered
 * neighbour (cosine, 768-D). Returns the starting positions and how many were covered, or null (too few covered).
 */
function precomputedStart() {
  if (!precomputedTsne || !items.length) return null;
  const placed = items.filter(it => precomputedTsne.has(it.id));
  if (placed.length < items.length * 0.3) return null;
  const pos = new Map(placed.map(it => [it, precomputedTsne.get(it.id)]));
  for (const it of items) {
    if (pos.has(it)) continue;
    let best = placed[0], bestSim = -Infinity;
    for (const o of placed) { const c = cosine(it.vec, o.vec); if (c > bestSim) { bestSim = c; best = o; } }
    const [x, y] = pos.get(best);
    pos.set(it, [x + (Math.random() - 0.5) * 2, y + (Math.random() - 0.5) * 2]);
  }
  return { start: items.map(it => pos.get(it)), covered: placed.length };
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
  const neighbours = selected ? new Set(selected.nn.map(n => n.it.id)) : tagList ? tagList.ids : null;
  // draw low scores first so the interesting ones sit on top
  const scored = scoreOf(items[0] ?? {}, mode) !== null;
  const order = items.slice().sort(scored ? (a, b) => scoreOf(a, mode) - scoreOf(b, mode) : (a, b) => (a.mu ?? -1e9) - (b.mu ?? -1e9));
  for (const it of order) {
    const [sx, sy] = toScreen(it);
    if (sx < -size || sy < -size || sx > w + size || sy > h + size) continue;
    const bw = size, bh = size * 0.66;
    const score = scoreOf(it, mode);
    const dimByScore = score !== null && score < scoreThreshold(mode) && it !== selected?.it;
    ctx.globalAlpha = (neighbours && !neighbours.has(it.id) && it !== selected?.it) ? 0.25 : dimByScore ? 0.3 : 1;
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
  drawTagLabels(mode, size);
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
      ? `「${escapeHTML(t.name)}」らしさ（CLIP ゼロショット${embedModel === 'gemma' ? '・サムネで判定' : ''}・目安）<div class="bar-grad tag-grad"></div><div class="row"><span>0%</span><span>50%（未満は薄く）</span><span>100%</span></div>該当 ${s.filter(p => p >= 0.5).length} / ${s.length} 本`
      : 'タグを計算中…（初回はテキストモデル約65MBを読み込みます）';
  } else if (mode === 'pop' && popInfo) {
    ui.legend.innerHTML = `人気（monsnode ランキング＋embedding からの推定）<div class="bar-grad tag-grad"></div><div class="row"><span>低</span><span>50%以上＝ランキング入り</span><span>高</span></div>`
      + `ランキング入り ${popInfo.known} 本・推定の当たり具合 AUC ${popInfo.auc.toFixed(2)}${popInfo.shrink > 0 ? '' : '（偶然と変わらないため推定は使っていません）'}`;
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
  if (!selected && !tagList) return;
  // Deselecting a tag also puts the colour mode back, unless it was changed by hand meanwhile.
  if (tagList && ui.color.value === tagList.key && [...ui.color.options].some(o => o.value === tagList.prevColor)) ui.color.value = tagList.prevColor;
  selected = null;
  tagList = null;
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
    <div class="panel-head">${tagList ? '<button type="button" class="ghost back-list"></button>' : ''}<button type="button" class="ghost deselect" title="選択解除 (Esc)">選択解除 ×</button></div>
    <div class="sel"><img alt=""><div class="t"></div><div class="s"></div><div class="tag-chips"></div><a target="_blank" rel="noopener">Open ↗</a></div>
    <button type="button" class="play-quad">▶ この動画と近傍を4画面で再生</button>
    <h3 style="margin-top:14px">動画フレーム（4区間の中央）</h3>
    <div class="frames"></div>
    <h3 style="margin-top:14px">${embedModel === 'gemma' ? 768 : 512}次元空間の近傍 Top 12（コサイン類似度）</h3>
    <div class="nn"></div>
    <div class="hist"><h3>この動画と全候補の類似度分布</h3><canvas width="600" height="140"></canvas><p></p></div>`;
  ui.panel.querySelector('.sel img').src = it.video.thumbnail;
  ui.panel.querySelector('.sel .t').textContent = it.video.title;
  const popText = it.video.pop ? ` · 人気 ${it.video.pop.period} ${it.video.pop.rank}位` : it.popScore !== null ? ` · 人気推定 ${Math.round(it.popScore * 100)}%` : '';
  ui.panel.querySelector('.sel .s').textContent = `${fmt(it.mu, it.sigma)} · ${it.shown ? '表示済み' : '未表示'}${popText}`;
  ui.panel.querySelector('.sel a').href = it.video.pageUrl;
  ui.panel.querySelector('.deselect').addEventListener('click', clearSelection);
  ui.panel.querySelector('.play-quad').addEventListener('click', () => openQuad([it, ...nn.map(n => n.it)], `${it.video.title.slice(0, 30)} と近傍`, o => o === it ? '選択中' : `類似 ${nn.find(n => n.it === o).sim.toFixed(3)}`));
  const back = ui.panel.querySelector('.back-list');
  if (back) { back.textContent = `← ${tagList.name} の一覧`; back.addEventListener('click', () => showTagList(tagList.key)); }
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
  const onLabel = hitLabel(e.offsetX, e.offsetY);
  canvas.style.cursor = onLabel ? 'pointer' : '';
  const it = onLabel ? null : pick(e.offsetX, e.offsetY);
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
  if (drag && !drag.moved) {
    const key = hitLabel(e.offsetX, e.offsetY);
    const it = key ? null : pick(e.offsetX, e.offsetY);
    if (key) showTagList(key); else if (it) select(it); else clearSelection();
  }
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

// ---------- 4-screen player (selected video + neighbours, or a tag's top videos) ----------
// Plays the list from the top: when a video ends (or fails to load) its cell moves on to the next unplayed one.
// Keys follow the other 4movPlayer players: W/A/S/D・Tab focus, playback keys act on the focused cell, Shift = all four.
const quad = { list: [], next: 0, badge: null, title: '', active: 0, cols: 2 };
// Auto layout: choose the number of columns that shows the most video area for the current videos' shapes
// (portrait videos side by side, landscape ones stacked). Off = fixed 2×2. Remembered per browser.
let quadAuto = true;
try { quadAuto = localStorage.getItem('gppl-quad-auto') !== 'off'; } catch { /* storage unavailable */ }
const fmtClock = t => Number.isFinite(t) ? `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}` : '0:00';
const QUAD_BINDINGS = {
  z: { action: 'rewind', value: 5 }, x: { action: 'advance', value: 5 },
  c: { action: 'rewind', value: 30 }, v: { action: 'advance', value: 30 },
  f: { action: 'reset', value: 1 }, g: { action: 'speedToggle', value: 2.5 }, h: { action: 'speedToggle', value: 10 },
  Space: { action: 'toggle' }, m: { action: 'mute' },
  arrowup: { action: 'volumeUp', value: 0.1 }, arrowdown: { action: 'volumeDown', value: 0.1 },
};
const quadCells = () => [...ui.quad.querySelectorAll('.quad-cell')];
const quadVideos = () => quadCells().map(c => c.querySelector('video'));

function openQuad(list, title, badge) {
  quad.list = list; quad.badge = badge; quad.title = title; quad.active = 0;
  ui.quad.hidden = false;
  ui.quad.querySelector('.quad-grid').replaceChildren(...[0, 1, 2, 3].map(i => quadCell(i)));
  syncAutoButton();
  showFrom(0);
  applyQuadLayout();
}

function closeQuad() {
  for (const v of quadVideos()) stopVideo(v);
  ui.quad.querySelector('.quad-grid').replaceChildren();
  ui.quad.hidden = true;
}

function stopVideo(v) { v.pause(); v.removeAttribute('src'); v.load(); }

/** Fill the four cells with list[start..start+3]; later ones follow as videos end. */
function showFrom(start) {
  start = Math.max(0, Math.min(start, Math.max(0, quad.list.length - 1)));
  quad.next = start;
  for (const cell of quadCells()) loadInto(cell, quad.next < quad.list.length ? quad.next++ : -1);
  updateQuadHead();
}

/** Move one cell on to the next unplayed video (or loop it when the list is used up). */
function advanceCell(cell) {
  if (quad.next < quad.list.length) loadInto(cell, quad.next++);
  else { const v = cell.querySelector('video'); v.currentTime = 0; v.play().catch(() => {}); }
  updateQuadHead();
}

function updateQuadHead() {
  const shown = quadCells().map(c => Number(c.dataset.idx)).filter(i => i >= 0);
  const left = Math.max(0, quad.list.length - quad.next);
  ui.quad.querySelector('.quad-title').textContent = quad.title;
  ui.quad.querySelector('.quad-page').textContent = shown.length
    ? `${Math.min(...shown) + 1}〜${Math.max(...shown) + 1} 本目を再生中 / ${quad.list.length} 本（残り ${left} 本）` : '';
}

function setQuadActive(i) {
  quad.active = i;
  quadCells().forEach((c, k) => c.classList.toggle('active', k === i));
}

function quadCell(slot) {
  const cell = document.createElement('div');
  cell.className = `quad-cell${slot === quad.active ? ' active' : ''}`;
  cell.dataset.idx = '-1';
  cell.innerHTML = `<video muted playsinline preload="auto"></video>
    <div class="quad-label"><span class="key"></span><span class="badge"></span><span class="t"></span><span class="snd"></span><a target="_blank" rel="noopener">Open ↗</a></div>
    <div class="quad-seek"><input type="range" min="0" max="1000" value="0" aria-label="シークバー"><span class="time">0:00 / 0:00</span><span class="speed">1.00x</span></div>`;
  const video = cell.querySelector('video');
  cell.querySelector('a').addEventListener('click', e => e.stopPropagation());
  cell.addEventListener('pointerdown', () => setQuadActive(quadCells().indexOf(cell)));
  // click: hear only this one (click again to mute it)
  cell.addEventListener('click', () => {
    const solo = video.muted;
    for (const v of quadVideos()) v.muted = true;
    video.muted = !solo;
  });
  video.addEventListener('volumechange', () => { for (const c of quadCells()) c.querySelector('.snd').textContent = c.querySelector('video').muted ? '' : '🔊'; });
  video.addEventListener('ratechange', () => { cell.querySelector('.speed').textContent = `${video.playbackRate.toFixed(2)}x`; });
  video.addEventListener('ended', () => advanceCell(cell));
  video.addEventListener('loadedmetadata', applyQuadLayout);
  const seek = cell.querySelector('.quad-seek'), range = seek.querySelector('input'), time = seek.querySelector('.time');
  for (const ev of ['click', 'pointerdown']) seek.addEventListener(ev, e => e.stopPropagation());
  range.addEventListener('input', () => { if (video.duration) video.currentTime = video.duration * range.value / 1000; });
  range.addEventListener('pointerup', () => range.blur());
  video.addEventListener('timeupdate', () => {
    if (video.duration) range.value = String(Math.round(video.currentTime / video.duration * 1000));
    time.textContent = `${fmtClock(video.currentTime)} / ${fmtClock(video.duration)}`;
  });
  return cell;
}

/** Put list[idx] into a cell (idx < 0 = empty). The cell keeps its mute / volume / speed. */
function loadInto(cell, idx) {
  const video = cell.querySelector('video');
  const token = String(Math.random());
  cell.dataset.idx = String(idx); cell.dataset.token = token;
  cell.classList.remove('failed');
  stopVideo(video);
  const it = quad.list[idx];
  cell.classList.toggle('empty', !it);
  cell.querySelector('.key').textContent = it ? idx + 1 : '';
  cell.querySelector('.badge').textContent = it ? quad.badge?.(it) ?? '' : '';
  cell.querySelector('.t').textContent = it ? it.video.title : '（一覧の最後まで再生しました）';
  cell.querySelector('a').href = it ? it.video.pageUrl : '#';
  cell.querySelector('.time').textContent = '0:00 / 0:00';
  if (!it) { video.removeAttribute('poster'); return; }
  if (it.video.thumbnail) video.poster = it.video.thumbnail;
  const fail = () => {
    if (cell.dataset.token !== token) return;
    cell.classList.add('failed');
    setTimeout(() => { if (cell.dataset.token === token && cell.isConnected) advanceCell(cell); }, 1500);   // skip unplayable ones
  };
  (async () => {
    try {
      const direct = await resolveVideoUrl(it.video);
      if (cell.dataset.token !== token) return;
      // web: direct twimg first, then the Worker proxy; extension: direct only
      const sources = scraper.mediaProxyUrl ? [direct, scraper.mediaProxyUrl(direct)] : [direct];
      let attempt = 0;
      video.onerror = () => {
        if (cell.dataset.token !== token) return;
        if (++attempt < sources.length) { video.src = sources[attempt]; video.play().catch(() => {}); } else fail();
      };
      video.src = sources[0];
      video.play().catch(() => {});
    } catch { fail(); }
  })();
}

function quadOperate(video, { action, value = 0 }) {
  if (action === 'rewind' || action === 'advance') {
    const d = Number.isFinite(video.duration) ? video.duration : Infinity;
    video.currentTime = Math.max(0, Math.min(d, video.currentTime + (action === 'rewind' ? -value : value)));
  } else if (action === 'reset') video.playbackRate = value;
  else if (action === 'speedToggle') video.playbackRate = Math.abs(video.playbackRate - value) < 0.01 ? 1 : value;
  else if (action === 'toggle') { if (video.paused || video.ended) video.play().catch(() => {}); else video.pause(); }
  else if (action === 'mute') video.muted = !video.muted;
  else if (action === 'volumeUp') { video.volume = Math.min(1, video.volume + value); video.muted = false; }
  else if (action === 'volumeDown') { video.volume = Math.max(0, video.volume - value); if (!video.volume) video.muted = true; }
}

/** Shift + a playback key: all four at once (toggles act as one group so the cells stay in sync). */
function quadOperateAll({ action, value = 0 }) {
  const vs = quadVideos().filter(v => v.getAttribute('src'));
  if (action === 'toggle') { const pause = vs.some(v => !v.paused && !v.ended); vs.forEach(v => (pause ? v.pause() : v.play().catch(() => {}))); }
  else if (action === 'mute') { const mute = vs.some(v => !v.muted); vs.forEach(v => { v.muted = mute; }); }
  else if (action === 'speedToggle') { const at = vs.every(v => Math.abs(v.playbackRate - value) < 0.01); vs.forEach(v => { v.playbackRate = at ? 1 : value; }); }
  else vs.forEach(v => quadOperate(v, { action, value }));
}

function quadMoveFocus(key, reverse = false) {
  const count = quadCells().length, cols = quad.cols, i = quad.active, col = i % cols;
  let n = i;
  if (key === 'tab') n = (i + (reverse ? count - 1 : 1)) % count;
  else if (key === 'w' && i - cols >= 0) n -= cols;
  else if (key === 's' && i + cols < count) n += cols;
  else if (key === 'a' && col > 0) n -= 1;
  else if (key === 'd' && col < cols - 1 && i + 1 < count) n += 1;
  setQuadActive(n);
}

/** Aspect ratio (w/h) of each cell's video; 16:9 until its size is known. */
const quadAspects = () => quadVideos().map(v => (v.videoWidth && v.videoHeight ? v.videoWidth / v.videoHeight : 16 / 9));

function applyQuadLayout() {
  if (ui.quad.hidden) return;
  const grid = ui.quad.querySelector('.quad-grid');
  const n = quadCells().length || 4;
  let cols = 2;
  if (quadAuto) {
    const { width, height } = grid.getBoundingClientRect();
    const aspects = quadAspects();
    let best = -1;
    for (let c = 1; c <= n; c++) {
      const rows = Math.ceil(n / c), cw = width / c, ch = height / rows;
      // area each video gets when fitted (object-fit: contain) into its cell
      const area = aspects.reduce((sum, a) => { const w = Math.min(cw, ch * a); return sum + w * (w / a); }, 0);
      if (area > best * 1.02) { best = area; cols = c; }   // prefer the earlier (2-column-ish) choice on near ties
    }
  }
  quad.cols = cols;
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  grid.style.gridTemplateRows = `repeat(${Math.ceil(n / cols)}, minmax(0, 1fr))`;
  const b = ui.quad.querySelector('.auto-layout');
  b.title = `自動レイアウト: 動画の縦横比に合わせて並べ方を選ぶ（いまは ${cols}列 × ${Math.ceil(n / cols)}行）`;
}

function syncAutoButton() {
  const b = ui.quad.querySelector('.auto-layout');
  b.textContent = `自動レイアウト ${quadAuto ? 'ON' : 'OFF'}`;
  b.setAttribute('aria-pressed', String(quadAuto));
}
addEventListener('resize', applyQuadLayout);

function quadKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (document.activeElement instanceof HTMLButtonElement || document.activeElement instanceof HTMLInputElement) document.activeElement.blur();
  const key = e.key.toLowerCase();
  const binding = e.code === 'Space' ? QUAD_BINDINGS.Space : QUAD_BINDINGS[key];
  if (e.key === 'Escape') closeQuad();
  else if (e.key === 'Tab') quadMoveFocus('tab', e.shiftKey);
  else if (e.shiftKey && binding) quadOperateAll(binding);
  else if (['w', 'a', 's', 'd'].includes(key)) quadMoveFocus(key);
  else if (key === 'n' || key === 'enter') advanceCell(quadCells()[quad.active]);        // skip the focused one
  else if (e.key === 'ArrowRight') { if (quad.next < quad.list.length) showFrom(quad.next); }
  else if (e.key === 'ArrowLeft') { const first = Math.min(...quadCells().map(c => Number(c.dataset.idx)).filter(i => i >= 0)); showFrom(first - 4); }
  else if (binding) { const v = quadVideos()[quad.active]; if (v) quadOperate(v, binding); }
  else return;
  e.preventDefault();
}

ui.quad.addEventListener('click', e => {
  const b = e.target.closest('button');
  if (b?.classList.contains('close')) closeQuad();
  else if (b?.classList.contains('auto-layout')) {
    quadAuto = !quadAuto;
    try { localStorage.setItem('gppl-quad-auto', quadAuto ? 'on' : 'off'); } catch { /* storage unavailable */ }
    syncAutoButton();
    applyQuadLayout();
  }
});

addEventListener('keydown', e => {
  if (!ui.quad.hidden) { quadKey(e); return; }
  if (e.key !== 'Escape') return;
  if (!ui.tagEditor.hidden) { ui.tagEditor.hidden = true; return; }
  if (e.target instanceof HTMLInputElement) return;
  clearSelection();
});
ui.proj.addEventListener('change', layout);
ui.color.addEventListener('change', draw);
ui.size.addEventListener('input', draw);
ui.tagLabels.addEventListener('change', draw);
ui.fit.addEventListener('click', () => { fitView(); draw(); });
addEventListener('resize', draw);

load().catch(err => setStatus(`エラー: ${err.message}`));
