// Web version: Monsnode is read through the Cloudflare Worker shared with monsnodeWebPlayer
// (monsnodeWebPlayer/worker/src/index.js), because a web page cannot read monsnode.com directly (CORS).
//   GET /search?q=&page=   → { videos: [{id, redirectId, title, pageUrl, redirectUrl, thumbnail}] }
//   GET /resolve?id=       → { videoUrl }   (video.twimg.com MP4)
//   GET /media?src=        → MP4 stream proxy (fallback when direct playback fails)
//   GET /search?ranking=1&period=24h|3d|7d&page=  → ranking pages (page 0 = first page)
// Thumbnails (pbs.twimg.com) send CORS headers, so they are fetched directly for CLIP.
import { POP_PERIODS, rankScore } from './popularity.js';

const API = (window.GPPL_API_BASE || '').replace(/\/+$/, '');
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function api(path) {
  if (!API) throw new Error('config.js の GPPL_API_BASE が未設定です');
  const res = await fetch(`${API}${path}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || `Worker HTTP ${res.status}`);
  return data;
}

/**
 * Collect up to `target` videos from list pages, with a pause between requests.
 * @param {{target?: number, query?: string, startPage?: number, delayMs?: number, onProgress?: (n:number, page:number)=>void}} opts
 */
export async function collectPool({ target = 300, query = '', startPage = 1, delayMs = 500, onProgress } = {}) {
  const out = []; const seen = new Set();
  for (let page = startPage; out.length < target && page <= 100; page++) {
    let data;
    try {
      data = await api(`/search?q=${encodeURIComponent(query)}&page=${page}`);
    } catch (error) {
      if (out.length) break;           // past the last page, or a transient failure after some results
      throw error;
    }
    const items = (data.videos || []).filter(v => v.thumbnail && v.redirectId && !seen.has(v.id) && seen.add(v.id));
    if (!items.length) break;
    out.push(...items);
    onProgress?.(out.length, page);
    await sleep(delayMs);
  }
  return out.slice(0, target);
}

/**
 * Videos in monsnode's popularity rankings, each with `pop = {s, period, rank, seenAt}`.
 * Pages that fail or are empty are skipped (a period may simply have no second page).
 */
export async function collectRanking({ delayMs = 400, onProgress } = {}) {
  const byId = new Map();
  for (const { period, pages } of POP_PERIODS) {
    const list = [];
    for (const page of pages) {
      try {
        const data = await api(`/search?ranking=1&period=${period}&page=${page}`);
        for (const v of data.videos || []) if (v.thumbnail && v.redirectId && !list.some(x => x.id === v.id)) list.push(v);
      } catch { /* no such page */ }
      await sleep(delayMs);
    }
    const seenAt = Date.now();
    list.forEach((v, rank) => {
      const pop = { s: rankScore(period, rank, list.length), period, rank: rank + 1, seenAt };
      if (!byId.has(v.id) || byId.get(v.id).pop.s < pop.s) byId.set(v.id, { ...v, pop });
    });
    onProgress?.(byId.size, period);
  }
  return [...byId.values()];
}

const mediaCache = new Map();
/** Resolve the twimg MP4 for a video (cached, lazy: only for the 4 on screen). */
export function resolveVideoUrl(item) {
  if (mediaCache.has(item.id)) return mediaCache.get(item.id);
  const redirectId = item.redirectId || new URL(item.redirectUrl).searchParams.get('v');
  const pending = api(`/resolve?id=${encodeURIComponent(redirectId)}`).then(d => d.videoUrl);
  mediaCache.set(item.id, pending);
  pending.catch(() => mediaCache.delete(item.id));
  return pending;
}

/** Worker-proxied stream URL, used when the direct twimg URL will not play. */
export function mediaProxyUrl(videoUrl) {
  return `${API}/media?src=${encodeURIComponent(videoUrl)}`;
}
