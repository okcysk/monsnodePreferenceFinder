// Web version: Monsnode is read through the Cloudflare Worker shared with monsnodeWebPlayer
// (monsnodeWebPlayer/worker/src/index.js), because a web page cannot read monsnode.com directly (CORS).
//   GET /search?q=&page=   → { videos: [{id, redirectId, title, pageUrl, redirectUrl, thumbnail}] }
//   GET /resolve?id=       → { videoUrl }   (video.twimg.com MP4)
//   GET /media?src=        → MP4 stream proxy (fallback when direct playback fails)
// Thumbnails (pbs.twimg.com) send CORS headers, so they are fetched directly for CLIP.

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
