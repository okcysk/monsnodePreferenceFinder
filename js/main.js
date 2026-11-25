import * as db from './db.js';
import { collectPool, collectRanking, resolveVideoUrl, mediaProxyUrl } from './scraper.js';
import { mergePop } from './popularity.js';
import { Embedder } from './embed.js';
import { computeRound, topSeen } from './engine.js';
import { rankingToPairs, KERNELS } from './gppl.js';
import { nextBeta } from './acquisition.js';
import { grabFrames, meanEmbedding } from './frames.js';

const MIN_POOL = 12;          // embedded videos needed before the first round
const STREAK_TO_FINISH = 3;   // best ranked #1 this many rounds in a row → show results
const FRAME_COUNT = 4;        // frames per video in 'frames' mode (averaged with the thumbnail)
const FRAME_CONCURRENCY = 4;  // videos whose frames are fetched at the same time (network-bound)
const GEMMA_FRAMES = 4;       // frames per video in 'gemma' mode (same positions as 'frames'; one clip embedding)
const GEMMA_CONCURRENCY = 2;  // the model is the bottleneck there: fetch the next video's frames while one runs
const MODE_LABEL = { thumb: 'サムネ', frames: 'サムネ＋フレーム', gemma: '動画（EmbeddingGemma 2）' };

const $ = id => document.getElementById(id);
const ui = {
  query: $('query'), target: $('target'), start: $('start'), status: $('status'), meter: $('meter-fill'),
  round: $('round'), bestThumb: $('best-thumb'), showResults: $('show-results'), reset: $('reset'),
  grid: $('grid'), actions: $('actions'), undo: $('undo'), skip: $('skip'), submit: $('submit'),
  empty: $('empty'), results: $('results'), resultsTitle: $('results-title'),
  resultsList: $('results-list'), closeResults: $('close-results'),
  help: $('help'), helpButton: $('help-button'), embedMode: $('embed-mode'), clipDevice: $('clip-device'), popWeight: $('pop-weight'), kernel: $('kernel'),
  notice: $('notice'),
};

/** id → {id, title, pageUrl, redirectUrl, thumbnail} */
const videos = new Map();
/**
 * Embedding stores per mode: 'thumb' = CLIP of the thumbnail, 'frames' = mean of thumbnail + FRAME_COUNT video frames,
 * 'gemma' = EmbeddingGemma 2 of the clip (GEMMA_FRAMES frames as one video, plus the thumbnail) — another model, another space.
 */
const thumbEmb = new Map();
const frameEmb = new Map();
const gemmaEmb = new Map();
const embOf = mode => (mode === 'gemma' ? gemmaEmb : mode === 'frames' ? frameEmb : thumbEmb);
const failedIn = { frames: new Set(), gemma: new Set() };   // per mode: failed this session (not retried until reload)
let embedMode = 'thumb';
let clipDevice = 'wasm';         // 'wasm' (q8, ~90MB) or 'webgpu' (fp32, ~350MB, faster)
const POP_WEIGHTS = { off: 0, low: 0.3, mid: 0.6, high: 1 };   // popularity prior strength (GP prior mean scale)
let popLevel = 'mid';
let kernel = 'rbf';              // GP kernel: 'rbf' (several peaks) / 'linear' (one direction) / 'quad' (one elliptical peak)
const saveSettings = () => db.kvSet('settings', { embedMode, clipDevice, popLevel, kernel });
/** Embeddings are stored per CLIP backend (q8 WASM vs fp32 WebGPU give different vectors). */
const storeOf = kind => (clipDevice === 'webgpu' ? `${kind}Gpu` : kind);
/** id → Float32Array for the active mode. Only embedded, playable videos: this IS the candidate pool. */
let pool = new Map();
const embedders = {};   // 'clip' / 'gemma' → Embedder (one worker each, created when first needed)
let embedding = false;
let lastRound = null;   // result of computeRound for the current screen

const newSession = () => ({ pairs: [], shown: [], broken: [], excluded: [], current: [], round: 0, beta: 2, streak: 0, query: '' });
let session = newSession();
let order = [];         // ids in the order the user clicked (best first)
let skipped = new Set(); // videos on screen the user chose not to rate this round (Q / 評価しない)
let activeIndex = 0;    // focused cell: playback shortcuts act on it (W/A/S/D, Tab, or click)

/** A notice that stays until closed (the status line is overwritten by progress updates). */
function showNotice(text) {
  ui.notice.querySelector('span').textContent = text;
  ui.notice.hidden = false;
}
ui.notice.querySelector('button').addEventListener('click', () => { ui.notice.hidden = true; });
const NOTICE_KEY = 'gppl-notice';   // carries a notice across the reload that a backend switch needs

/** Disable the WebGPU option up front when this browser cannot provide a GPU adapter. */
async function checkWebGPUOption() {
  const option = ui.clipDevice.querySelector('option[value="webgpu"]');
  let problem = null;
  if (!navigator.gpu) problem = 'このブラウザは WebGPU に対応していません';
  else if (!(await navigator.gpu.requestAdapter().catch(() => null))) problem = 'GPU アダプタを取得できません';
  if (problem) {
    option.disabled = true;
    option.textContent = '計算: WebGPU（この環境では使えません）';
    ui.clipDevice.title = `WebGPU: ${problem}`;
  }
}

function setStatus(text, fraction = null) {
  ui.status.textContent = text;
  if (fraction !== null) ui.meter.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
}
const save = () => db.kvSet('session', session);
/** Common computeRound inputs; `shown` can be widened (e.g. to avoid what is on screen). */
const roundArgs = (shown = new Set(session.shown)) => ({
  embeddings: pool, pairs: session.pairs.filter(([w, l]) => pool.has(w) && pool.has(l)),
  shown, beta: session.beta, excluded: new Set(session.excluded),
  popularity: popularityMap(), popWeight: POP_WEIGHTS[popLevel] ?? 0, kernel,
});
/** id → ranking score for videos that appeared in a monsnode ranking (popularity.js). */
function popularityMap() {
  const m = new Map();
  for (const v of videos.values()) if (v.pop && pool.has(v.id)) m.set(v.id, v.pop.s);
  return m;
}
const popLabel = pop => (pop ? `人気 ${pop.period} ${pop.rank}位` : '');

// ---------- boot ----------
async function boot() {
  for (const v of await db.getAll('videos')) videos.set(v.id, v);
  const saved = await db.kvGet('session');
  if (saved) session = { ...newSession(), ...saved };
  const settings = await db.kvGet('settings');
  embedMode = settings?.embedMode in MODE_LABEL ? settings.embedMode : 'thumb';
  clipDevice = settings?.clipDevice === 'webgpu' ? 'webgpu' : 'wasm';
  popLevel = settings?.popLevel in POP_WEIGHTS ? settings.popLevel : 'mid';
  ui.popWeight.value = popLevel;
  kernel = KERNELS.includes(settings?.kernel) ? settings.kernel : 'rbf';
  ui.kernel.value = kernel;
  ui.embedMode.value = embedMode;
  ui.clipDevice.value = clipDevice;
  try {
    const notice = sessionStorage.getItem(NOTICE_KEY);
    if (notice) { sessionStorage.removeItem(NOTICE_KEY); showNotice(notice); }
  } catch { /* ignore */ }
  checkWebGPUOption();
  for (const e of await db.getAll(storeOf('embeddings'))) thumbEmb.set(e.id, e.vec);
  for (const e of await db.getAll(storeOf('frameEmbeddings'))) frameEmb.set(e.id, e.vec);
  for (const e of await db.getAll(storeOf('gemmaEmbeddings'))) gemmaEmb.set(e.id, e.vec);
  buildPool();
  ui.query.value = session.query || '';
  setStatus(videos.size ? `候補 ${videos.size} 本 / 埋め込み済み ${pool.size} 本` : '「プール取得」で候補を集めてください', videos.size ? pool.size / videos.size : 0);
  if (pool.size < videos.size) startEmbedding();
  if (session.current.length === 4 && session.current.every(id => pool.has(id))) {
    // Resume: rebuild the posterior (for the best thumbnail / results) but keep the same four on screen.
    if (session.pairs.length) lastRound = computeRound(roundArgs());
    showRound(session.current);
  } else if (pool.size >= MIN_POOL) nextRound();
  else if (videos.size) waitForPool();
  updateHeader();
}

/** Candidate pool for the active embedding mode. A new Map each time so the PCA cache is rebuilt. */
function buildPool() {
  const broken = new Set(session.broken);
  const src = embOf(embedMode);
  pool = new Map([...src].filter(([id]) => videos.has(id) && !broken.has(id)));
}

function waitForPool() {
  ui.grid.replaceChildren();
  ui.grid.hidden = true;
  ui.actions.hidden = true;
  ui.empty.hidden = false;
  lastRound = null;
  setStatus(`${MODE_LABEL[embedMode]}の埋め込みが ${MIN_POOL} 本たまるとラウンドを始めます（現在 ${pool.size} 本）`);
}

/** Switch the embedding used by GPPL. Comparisons are kept: they are about videos, not features. */
function refreshForMode() {
  buildPool();
  if (pool.size < MIN_POOL) { waitForPool(); return; }
  if (!ui.grid.hidden && session.current.length === 4 && session.current.every(id => pool.has(id))) {
    lastRound = session.pairs.length ? computeRound(roundArgs()) : null;   // same four stay on screen
  } else nextRound();
}

// The CLIP backend is fixed when the worker loads the model, so a change takes effect after a reload.
ui.clipDevice.addEventListener('change', async () => {
  clipDevice = ui.clipDevice.value;
  await saveSettings();
  location.reload();
});

ui.popWeight.addEventListener('change', async () => {
  popLevel = ui.popWeight.value;
  await saveSettings();
  if (pool.size >= MIN_POOL && !ui.grid.hidden) {
    lastRound = session.pairs.length ? computeRound(roundArgs()) : lastRound;   // same four stay on screen
    updateHeader();
  }
  setStatus(`人気の事前情報: ${ui.popWeight.selectedOptions[0].textContent}（次のラウンドから反映）`);
});

ui.kernel.addEventListener('change', async () => {
  kernel = ui.kernel.value;
  await saveSettings();
  if (pool.size >= MIN_POOL && !ui.grid.hidden && session.pairs.length) {
    lastRound = computeRound(roundArgs());   // same four stay on screen; the best / results use the new kernel
    updateHeader();
  }
  const lm = lastRound?.model?.logMarginal;
  setStatus(`${ui.kernel.selectedOptions[0].textContent}（次のラウンドから反映）${Number.isFinite(lm) ? `・周辺尤度 ${lm.toFixed(2)}（大きいほど今までの回答をよく説明）` : ''}`);
});

ui.embedMode.addEventListener('change', async () => {
  embedMode = ui.embedMode.value;
  await saveSettings();
  refreshForMode();
  if (pool.size >= MIN_POOL) setStatus(`${MODE_LABEL[embedMode]}に切り替えました（プール ${pool.size} 本）`);
  if (embedMode === 'gemma') showNotice(`動画（EmbeddingGemma 2）: 初回はモデル（WASM 約280MB／WebGPU 約880MB）を取得します。1本あたり WebGPU で約10秒、WASM ではさらに長くかかるので、埋め込みは裏で少しずつ進みます（${MIN_POOL} 本たまるとラウンド開始）。`);
  startEmbedding();
  updateHeader();
});

// ---------- pool collection ----------
ui.start.addEventListener('click', async () => {
  ui.start.disabled = true;
  try {
    const query = ui.query.value.trim();
    const target = Number(ui.target.value);
    setStatus('Monsnode から一覧を取得中…', 0);
    const listed = await collectPool({ target, query, onProgress: (n, page) => setStatus(`一覧取得中… ${n}/${target} 本（${page} ページ）`, n / target) });
    // Popularity rankings: added to the pool when browsing the new list; with a search query only
    // the videos already collected get their ranking (the rankings are not about the query).
    setStatus('人気ランキングを取得中…');
    const ranked = await collectRanking({ onProgress: (n, period) => setStatus(`人気ランキングを取得中… ${period} まで ${n} 本`) }).catch(() => []);
    const byId = new Map(listed.map(v => [v.id, v]));
    for (const r of ranked) {
      if (byId.has(r.id)) byId.get(r.id).pop = r.pop;
      else if (!query) byId.set(r.id, r);
      else if (videos.has(r.id)) byId.set(r.id, { ...videos.get(r.id), pop: r.pop });
    }
    const items = [...byId.values()].map(v => ({ ...videos.get(v.id), ...v, pop: mergePop(videos.get(v.id)?.pop, v.pop) }));
    const fresh = items.filter(v => !videos.has(v.id));
    for (const v of items) videos.set(v.id, v);
    await db.putMany('videos', items);
    if (query !== session.query) { session.query = query; await save(); }
    const rankedIn = items.filter(v => v.pop).length;
    setStatus(`候補 ${videos.size} 本（新規 ${fresh.length}・人気ランキング ${rankedIn} 本）。埋め込みを開始します…`, pool.size / videos.size);
    startEmbedding();
  } catch (error) {
    setStatus(`一覧取得に失敗: ${error.message}`);
  } finally {
    ui.start.disabled = false;
  }
});

// ---------- embedding ----------
/** One worker per model, created when first needed. */
function getEmbedder(model) {
  const modelName = model === 'gemma' ? 'EmbeddingGemma 2' : 'CLIP';
  return embedders[model] ??= new Embedder({
    model,
    device: clipDevice,
    onModelProgress: p => setStatus(`${modelName} モデルを読み込み中… ${p.file ?? ''} ${Math.round(p.progress ?? 0)}%`, (p.progress ?? 0) / 100),
    onReady: async (device, fallbackReason) => {
      if (device === clipDevice) { setStatus(`${modelName} 準備完了（${device}）`); return; }
      // WebGPU was requested but is unavailable: vectors would differ, so switch the setting (and stores) to WASM,
      // and keep the reason visible after the reload.
      try { sessionStorage.setItem(NOTICE_KEY, `WebGPU を使えなかったため WASM に戻しました。理由: ${fallbackReason || '不明'}`); } catch { /* ignore */ }
      clipDevice = 'wasm';
      await saveSettings();
      location.reload();
    },
  });
}

function startEmbedding() {
  if (embedding) return;
  embedding = true;
  const mode = embedMode;
  const embedder = getEmbedder(mode === 'gemma' ? 'gemma' : 'clip');
  const clip = getEmbedder('clip');
  (async () => {
    const broken = new Set(session.broken);
    const src = embOf(mode);
    // gemma mode also fills in missing CLIP thumbnail vectors (the map's tags use them)
    const needed = v => !src.has(v.id) || (mode === 'gemma' && !thumbEmb.has(v.id));
    const todo = [...videos.values()].filter(v => needed(v) && !broken.has(v.id) && v.thumbnail && !failedIn[mode]?.has(v.id));
    let done = 0, failed = 0, thumbOnly = 0;
    const label = mode === 'thumb' ? '埋め込み中' : `${MODE_LABEL[mode]}の埋め込み中`;
    const embedOne = async v => {
      try {
        if (mode === 'gemma') {
          if (!gemmaEmb.has(v.id)) {
            // The frames (same positions as 'frames') go in as one video, the thumbnail as a poster image in the same
            // sample. When the frames cannot be grabbed (URL not resolvable, very long video…), the thumbnail alone is
            // embedded by the same model, so the video stays in the pool instead of being dropped.
            const [frames, poster] = await Promise.all([
              resolveVideoUrl(v).then(url => grabFrames(url, { count: GEMMA_FRAMES, size: 320 })).catch(error => {
                console.debug('[gppl] gemma: frames unavailable, thumbnail only', v.id, error?.message);
                return null;
              }),
              fetch(v.thumbnail, { credentials: 'omit' }).then(r => (r.ok ? r.blob() : null)).catch(() => null),
            ]);
            if (!frames && !poster) throw new Error('no frames and no thumbnail');
            const vec = frames ? await embedder.embedVideo(frames, frames[0]?.duration, poster) : await embedder.embedBlob(poster);
            if (!frames) thumbOnly++;
            gemmaEmb.set(v.id, vec);
            await db.put(storeOf('gemmaEmbeddings'), { id: v.id, vec });
            if (embedMode === 'gemma') pool.set(v.id, vec);
          }
          // The map's tags / text search use CLIP on the thumbnail (the captions are tuned for CLIP), so keep that too.
          if (!thumbEmb.has(v.id)) {
            try {
              const t = await clip.embedThumbnail(v.id, v.thumbnail);
              thumbEmb.set(v.id, t);
              await db.put(storeOf('embeddings'), { id: v.id, vec: t });
            } catch { /* tags only: not needed for the pool */ }
          }
          return;
        }
        let thumb = thumbEmb.get(v.id);
        if (!thumb) {
          thumb = await clip.embedThumbnail(v.id, v.thumbnail);
          thumbEmb.set(v.id, thumb);
          await db.put(storeOf('embeddings'), { id: v.id, vec: thumb });
          if (embedMode === 'thumb') pool.set(v.id, thumb);
        }
        if (mode === 'frames') {
          // Seek-and-grab a few frames (only the needed byte ranges are downloaded), then average with the thumbnail.
          const frames = await grabFrames(await resolveVideoUrl(v), { count: FRAME_COUNT });
          const vecs = await clip.embedBlobs(frames);   // all frames in one batched model call
          const vec = meanEmbedding([thumb, ...vecs]);
          frameEmb.set(v.id, vec);
          await db.put(storeOf('frameEmbeddings'), { id: v.id, vec });
          if (embedMode === 'frames') pool.set(v.id, vec);
        }
      } catch (error) {
        failed++;
        console.debug('[gppl] embedding failed', mode, v.id, error?.message);
        failedIn[mode]?.add(v.id);
      } finally {
        done++;
        progress();
      }
    };
    const progress = () => {
      if (embedMode !== mode) return;   // switched away: the new mode owns the status line
      setStatus(`${label} ${done}/${todo.length}（失敗 ${failed}${thumbOnly ? `・サムネのみ ${thumbOnly}` : ''}）・プール ${pool.size} 本${embedder.device ? ` · ${embedder.device}` : ''}`, done / todo.length);
      if (ui.grid.hidden && pool.size >= MIN_POOL) nextRound();
    };
    // Frame grabbing is network-bound: several videos at a time; CLIP batches still run one by one in the worker.
    let next = 0;
    await Promise.all(Array.from({ length: mode === 'frames' ? FRAME_CONCURRENCY : mode === 'gemma' ? GEMMA_CONCURRENCY : 1 }, async () => {
      while (next < todo.length && embedMode === mode) await embedOne(todo[next++]);
    }));
    embedding = false;
    if (embedMode !== mode) { startEmbedding(); return; }   // mode switched mid-way: continue for the new mode
    if (failed && pool.size < MIN_POOL) setStatus(`埋め込みに失敗しました（失敗 ${failed}）。時間をおいて再度「プール取得」を試してください。`);
    else setStatus(`候補 ${videos.size} 本 / 埋め込み済み ${pool.size} 本（${MODE_LABEL[mode]}）${failed ? `・失敗 ${failed}` : ''}`, 1);
  })();
}


// ---------- rounds ----------
function nextRound() {
  if (pool.size < 4) return;
  const t0 = performance.now();
  lastRound = computeRound(roundArgs());
  console.debug(`[gppl] round computed in ${(performance.now() - t0).toFixed(0)} ms, kernel=${kernel}, pairs=${session.pairs.length}, ell=${lastRound.model?.ell?.toFixed(3)}, logMarginal=${lastRound.model?.logMarginal?.toFixed(2)}`);
  session.current = lastRound.selection;
  save();
  showRound(session.current);
  updateHeader();
}

function showRound(ids) {
  order = [];
  skipped = new Set();
  ui.empty.hidden = true;
  ui.grid.hidden = false;
  ui.actions.hidden = false;
  ui.grid.replaceChildren(...ids.map((id, i) => makeCell(id, i)));
  syncRanks();
}

function makeCell(id, index) {
  const v = videos.get(id);
  const cell = document.createElement('div');
  cell.className = 'cell';
  cell.dataset.id = id;
  const keepBest = index === 0 && session.round > 0 && session.pairs.length > 0;
  cell.innerHTML = `
    ${keepBest ? '<span class="tag">暫定1位</span>' : ''}
    <video muted loop playsinline preload="auto"></video>
    <div class="rank" hidden></div>
    <div class="label"><span class="key">${index + 1}</span><span class="title"></span>${v?.pop ? '<span class="pop"></span>' : ''}
      <button class="exclude" type="button" title="この動画は評価しない (Q)">評価しない</button>
      <button class="mute" type="button" title="ミュート切替 (M)">🔇</button><a target="_blank" rel="noopener">Open ↗</a></div>
    <div class="excluded-mark">評価しない<small>クリック / Q で戻す</small></div>
    <div class="seek"><input type="range" min="0" max="1000" value="0" aria-label="シークバー"><span class="time">0:00 / 0:00</span><span class="speed">1.00x</span></div>`;
  cell.querySelector('.title').textContent = v?.title ?? id;
  if (v?.pop) { const b = cell.querySelector('.pop'); b.textContent = `🔥${v.pop.period} ${v.pop.rank}位`; b.title = `monsnode の人気ランキング（${v.pop.period}）で ${v.pop.rank} 位`; }
  const link = cell.querySelector('a');
  link.href = v?.pageUrl ?? '#';
  link.addEventListener('click', e => e.stopPropagation());
  const video = cell.querySelector('video');
  if (v?.thumbnail) video.poster = v.thumbnail;
  const mute = cell.querySelector('.mute');
  mute.addEventListener('click', e => { e.stopPropagation(); video.muted = !video.muted; });
  video.addEventListener('volumechange', () => { mute.textContent = video.muted ? '🔇' : '🔊'; });
  attachSeekBar(cell.querySelector('.seek'), video);
  cell.querySelector('.exclude').addEventListener('click', e => { e.stopPropagation(); toggleExclude(id); });
  cell.addEventListener('click', () => (skipped.has(id) ? toggleExclude(id) : toggleRank(id)));
  cell.addEventListener('pointerdown', () => setActive(index));
  cell.classList.toggle('active', index === activeIndex);
  loadVideo(cell, video, v, index);
  return cell;
}

// ---------- seek bar & playback shortcuts (same bindings as the other multi players) ----------
const fmtTime = s => Number.isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : '0:00';

function attachSeekBar(bar, video) {
  const range = bar.querySelector('input'), time = bar.querySelector('.time'), speed = bar.querySelector('.speed');
  let dragging = false;
  const update = () => {
    const d = video.duration, p = Number.isFinite(d) && d > 0 ? video.currentTime / d : 0;
    if (!dragging) range.value = String(Math.round(p * 1000));
    range.style.setProperty('--played', `${(dragging ? Number(range.value) / 10 : p * 100)}%`);
    time.textContent = `${fmtTime(video.currentTime)} / ${fmtTime(d)}`;
  };
  // The bar must not count as a click on the cell (which would rank it).
  for (const ev of ['click', 'pointerdown', 'dblclick']) bar.addEventListener(ev, e => e.stopPropagation());
  range.addEventListener('pointerdown', () => { dragging = true; });
  range.addEventListener('pointerup', () => { dragging = false; range.blur(); });
  range.addEventListener('input', () => {
    const d = video.duration;
    if (Number.isFinite(d) && d > 0) video.currentTime = d * Number(range.value) / 1000;
    update();
  });
  for (const ev of ['timeupdate', 'durationchange', 'loadedmetadata', 'seeked']) video.addEventListener(ev, update);
  video.addEventListener('ratechange', () => { speed.textContent = `${video.playbackRate.toFixed(2)}x`; });
}

const BINDINGS = {
  z: { action: 'rewind', value: 5 }, x: { action: 'advance', value: 5 },
  c: { action: 'rewind', value: 30 }, v: { action: 'advance', value: 30 },
  f: { action: 'reset', value: 1 }, g: { action: 'speedToggle', value: 2.5 }, h: { action: 'speedToggle', value: 10 },
  Space: { action: 'toggle' }, m: { action: 'mute' },
  arrowup: { action: 'volumeUp', value: 0.1 }, arrowdown: { action: 'volumeDown', value: 0.1 },
};

function setActive(index) {
  activeIndex = index;
  [...ui.grid.children].forEach((c, i) => c.classList.toggle('active', i === index));
}

/** Move focus in the 2×2 grid: W/A/S/D by direction, Tab / Shift+Tab cyclically. */
function moveFocus(key, reverse = false) {
  const n = ui.grid.children.length, cols = 2;
  const row = Math.floor(activeIndex / cols), col = activeIndex % cols;
  let next = activeIndex;
  if (key === 'tab') next = (activeIndex + (reverse ? -1 : 1) + n) % n;
  else if (key === 'w' && row > 0) next -= cols;
  else if (key === 's' && activeIndex + cols < n) next += cols;
  else if (key === 'a' && col > 0) next -= 1;
  else if (key === 'd' && col < cols - 1 && activeIndex + 1 < n) next += 1;
  setActive(next);
}
const cellVideos = () => [...ui.grid.querySelectorAll('.cell video')];

function operate(video, { action, value = 0 }) {
  if (action === 'rewind' || action === 'advance') {
    const d = Number.isFinite(video.duration) ? video.duration : Infinity;
    video.currentTime = Math.max(0, Math.min(d, video.currentTime + (action === 'rewind' ? -value : value)));
  } else if (action === 'decrease') video.playbackRate = Math.max(0.1, video.playbackRate - value);
  else if (action === 'increase') video.playbackRate = Math.min(16, video.playbackRate + value);
  else if (action === 'reset') video.playbackRate = value;
  else if (action === 'speedToggle') video.playbackRate = Math.abs(video.playbackRate - value) < 0.01 ? 1 : value;
  else if (action === 'toggle') { if (video.paused || video.ended) video.play().catch(() => {}); else video.pause(); }
  else if (action === 'mute') video.muted = !video.muted;
  else if (action === 'volumeUp') { video.volume = Math.min(1, video.volume + value); video.muted = false; }
  else if (action === 'volumeDown') { video.volume = Math.max(0, video.volume - value); if (!video.volume) video.muted = true; }
}

/** Shift + any playback key: apply it to all four videos. Toggles act as one group so the cells stay in sync. */
function operateAll({ action, value = 0 }) {
  const vs = cellVideos();
  if (action === 'toggle') {
    const pause = vs.some(v => !v.paused && !v.ended);
    vs.forEach(v => pause ? v.pause() : v.play().catch(() => {}));
  } else if (action === 'mute') {
    const mute = vs.some(v => !v.muted);
    vs.forEach(v => { v.muted = mute; });
  } else if (action === 'speedToggle') {
    const atTarget = vs.every(v => Math.abs(v.playbackRate - value) < 0.01);
    vs.forEach(v => { v.playbackRate = atTarget ? 1 : value; });
  } else vs.forEach(v => operate(v, { action, value }));
}

async function loadVideo(cell, video, v, index) {
  const showError = () => {
    if (cell.querySelector('.err')) return;
    const err = document.createElement('div');
    err.className = 'err';
    err.innerHTML = '<span>動画を読み込めませんでした</span><span class="err-actions"><button type="button" class="replace">差し替え</button><button type="button" class="skip">評価しない</button></span>';
    err.querySelector('.replace').addEventListener('click', e => { e.stopPropagation(); replaceSlot(index); });
    err.querySelector('.skip').addEventListener('click', e => { e.stopPropagation(); toggleExclude(v.id); });
    cell.append(err);
  };
  try {
    const direct = await resolveVideoUrl(v);
    // 1) direct twimg (no Referer, like monsnodeWebPlayer)  2) Worker /media  3) Worker /media once more  → error
    const sources = [direct, mediaProxyUrl(direct), `${mediaProxyUrl(direct)}&retry=1`];
    let attempt = 0;
    video.addEventListener('error', () => {
      if (++attempt >= sources.length) { showError(); return; }
      setTimeout(() => { video.src = sources[attempt]; video.play().catch(() => {}); }, attempt === 1 ? 0 : 800);
    });
    video.src = sources[0];
    video.play().catch(() => {});
  } catch (error) {
    showError();
  }
}

/** Drop a broken video from the pool and fill its slot with the next-best candidate. */
async function replaceSlot(index) {
  const id = session.current[index];
  session.broken.push(id);
  pool.delete(id);
  const res = computeRound(roundArgs(new Set([...session.shown, ...session.current])));
  const replacement = res.selection.find(x => !session.current.includes(x));
  if (!replacement) return;
  session.current[index] = replacement;
  order = order.filter(x => x !== id);
  skipped.delete(id);
  await save();
  ui.grid.children[index].replaceWith(makeCell(replacement, index));
  syncRanks();
}

/** Videos on screen that still need a rank. */
const rateable = () => session.current.filter(id => !skipped.has(id));

/** Once all but one are ranked, the last one is implied. */
function fillImplied() {
  const ids = rateable();
  if (ids.length >= 2 && order.length === ids.length - 1) order.push(ids.find(x => !order.includes(x)));
}

function toggleRank(id) {
  if (skipped.has(id)) return;
  const at = order.indexOf(id);
  if (at >= 0) order = order.slice(0, at);   // un-ranking also clears the ranks after it
  else if (order.length < rateable().length) order.push(id);
  fillImplied();
  syncRanks();
}

/** Leave one video out of this round's ranking (e.g. it does not play). It will not be shown again. */
function toggleExclude(id) {
  if (!session.current.includes(id)) return;
  // If the ranking was complete, its last entry was implied, not clicked: drop it before re-evaluating.
  const prevNeed = rateable().length;
  if (prevNeed >= 2 && order.length === prevNeed) order = order.slice(0, -1);
  if (skipped.has(id)) skipped.delete(id); else skipped.add(id);
  order = order.filter(x => !skipped.has(x));
  fillImplied();
  syncRanks();
}

function syncRanks() {
  for (const cell of ui.grid.children) {
    const r = order.indexOf(cell.dataset.id);
    const badge = cell.querySelector('.rank');
    badge.hidden = r < 0;
    badge.textContent = r >= 0 ? String(r + 1) : '';
    cell.classList.toggle('ranked', r >= 0);
    cell.classList.toggle('excluded', skipped.has(cell.dataset.id));
  }
  const need = rateable().length;
  // With 0–1 videos left to rate there is nothing to compare; submitting just moves on.
  ui.submit.disabled = need >= 2 && order.length !== need;
}

/**
 * While the next round is being computed: show it at once, and ignore further clicks / Enter
 * (a second submit would record the same ranking twice).
 */
let busy = false;
const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));
async function whileBusy(label, work) {
  if (busy) return;
  busy = true;
  const text = ui.submit.textContent;
  ui.submit.disabled = true; ui.skip.disabled = true; ui.undo.disabled = true;
  ui.submit.textContent = label;
  ui.grid.classList.add('busy');
  setStatus(`${label}（次の4本を選んでいます）`);
  try {
    await nextFrame();                 // let the button and the dimmed grid paint before the heavy work
    await work();
  } finally {
    ui.submit.textContent = text;
    ui.skip.disabled = false; ui.undo.disabled = false;
    ui.grid.classList.remove('busy');
    busy = false;
    syncRanks();
  }
}

function submit() {
  if (ui.submit.disabled || busy) return;
  return whileBusy('計算中…', submitNow);
}
async function submitNow() {
  const wasBest = lastRound?.best ?? (session.pairs.length ? session.current[0] : null);
  if (order.length >= 2) session.pairs.push(...rankingToPairs(order));
  session.shown = [...new Set([...session.shown, ...session.current])];
  session.excluded = [...new Set([...session.excluded, ...skipped])];
  session.streak = wasBest && order[0] === wasBest ? session.streak + 1 : 0;
  session.round++;
  session.beta = nextBeta(session.beta);
  await save();
  nextRound();
  setStatus(`ラウンド ${session.round} · 比較 ${session.pairs.length} 件`);
  if (session.streak >= STREAK_TO_FINISH) showResults(true);
}
function skip() {
  if (busy) return;
  return whileBusy('入れ替え中…', skipNow);
}
async function skipNow() {
  session.shown = [...new Set([...session.shown, ...session.current])];
  session.excluded = [...new Set([...session.excluded, ...skipped])];
  session.round++;
  await save();
  nextRound();
}

function updateHeader() {
  ui.round.textContent = `ラウンド ${session.round}${session.streak ? ` · 連続1位 ${session.streak}` : ''}`;
  const best = lastRound?.best ? videos.get(lastRound.best) : null;
  ui.bestThumb.hidden = !best?.thumbnail;
  if (best?.thumbnail) { ui.bestThumb.src = best.thumbnail; ui.bestThumb.title = `暫定1位: ${best.title}`; }
}

function showResults(converged = false) {
  if (!lastRound?.mu) { setStatus('まだ結果がありません。1ラウンド以上回答してください。'); return; }
  ui.resultsTitle.textContent = converged ? `収束しました — あなたのお気に入り Top 10` : 'あなたのお気に入り Top 10（暫定）';
  const excluded = new Set(session.excluded);
  const ids = topSeen(lastRound.mu, new Set(session.shown.filter(id => !excluded.has(id))), 10);
  ui.resultsList.replaceChildren(...ids.map(id => {
    const v = videos.get(id);
    const li = document.createElement('li');
    li.innerHTML = '<img alt=""><span class="t"></span><span class="s"></span><a target="_blank" rel="noopener">Open ↗</a>';
    li.querySelector('img').src = v?.thumbnail ?? '';
    li.querySelector('.t').textContent = v?.title ?? id;
    li.querySelector('.s').textContent = `μ ${lastRound.mu.get(id).toFixed(2)} ± ${lastRound.sigma.get(id).toFixed(2)}${v?.pop ? ` · ${popLabel(v.pop)}` : ''}`;
    li.querySelector('a').href = v?.pageUrl ?? '#';
    return li;
  }));
  ui.results.hidden = false;
}

// ---------- controls ----------
ui.submit.addEventListener('click', submit);
ui.skip.addEventListener('click', skip);
ui.undo.addEventListener('click', () => {
  const need = rateable().length;
  order = need >= 2 && order.length === need ? order.slice(0, need - 2) : order.slice(0, -1);
  syncRanks();
});
ui.showResults.addEventListener('click', () => showResults(false));
ui.closeResults.addEventListener('click', () => { ui.results.hidden = true; });
ui.reset.addEventListener('click', async () => {
  if (!confirm('学習した好み（比較履歴）をリセットしますか？\n候補プールと埋め込みは残ります。')) return;
  session = { ...newSession(), query: session.query };
  lastRound = null;
  await save();
  nextRound();
  updateHeader();
});

document.addEventListener('keydown', e => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const t = e.target;
  if ((t instanceof HTMLInputElement && t.type !== 'range') || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
  if (!ui.help.hidden) { if (e.key === 'Escape' || e.key === '?') ui.help.hidden = true; e.preventDefault(); return; }
  if (!ui.results.hidden) { if (e.key === 'Escape') ui.results.hidden = true; return; }
  if (e.key === '?') { ui.help.hidden = false; e.preventDefault(); return; }
  if (ui.grid.hidden) return;
  // Keep Space/Enter from also "clicking" whichever button has focus.
  if (document.activeElement instanceof HTMLButtonElement || t instanceof HTMLInputElement) document.activeElement.blur();
  const key = e.key.toLowerCase();
  if (e.key === 'Tab') { moveFocus('tab', e.shiftKey); e.preventDefault(); return; }
  if (e.shiftKey) {
    const binding = e.code === 'Space' ? BINDINGS.Space : BINDINGS[key];
    if (binding) { operateAll(binding); e.preventDefault(); }
    return;
  }
  const digit = /^Digit([1-4])$/.exec(e.code)?.[1];
  if (digit && session.current[digit - 1]) { toggleRank(session.current[digit - 1]); e.preventDefault(); return; }
  if (['w', 'a', 's', 'd'].includes(key)) { moveFocus(key); e.preventDefault(); return; }
  if (key === 'q') { const id = session.current[activeIndex]; if (id) toggleExclude(id); e.preventDefault(); return; }
  if (e.key === 'Enter') { submit(); e.preventDefault(); return; }
  if (e.key === 'Backspace') { ui.undo.click(); e.preventDefault(); return; }
  const binding = e.code === 'Space' ? BINDINGS.Space : BINDINGS[key];
  const video = cellVideos()[activeIndex];
  if (binding && video) { operate(video, binding); e.preventDefault(); }
});

ui.helpButton.addEventListener('click', () => { ui.help.hidden = false; });
ui.help.addEventListener('click', e => { if (e.target === ui.help || e.target.closest('.close')) ui.help.hidden = true; });

boot().catch(error => setStatus(`起動エラー: ${error.message}`));
