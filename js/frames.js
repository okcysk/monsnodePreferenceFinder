// Grab a few frames from a remote MP4 without downloading the whole file.
// A hidden <video> with crossOrigin="anonymous" seeks to evenly spaced points; the browser fetches only
// the moov box (twimg MP4s are fast-start) plus the ranges around each seek point via HTTP Range.
// video.twimg.com answers with Access-Control-Allow-Origin, so the canvas is not tainted and can be read.
// The page must not send a foreign Referer (see <meta name="referrer" content="no-referrer">) or twimg returns 403.

/**
 * @param {string} url  video.twimg.com MP4
 * @param {{count?: number, metaTimeoutMs?: number, timeoutMs?: number, size?: number}} [opts]
 *   metaTimeoutMs: give up early when the moov box is slow (very long videos have multi-MB indexes)
 * @returns {Promise<Blob[]>}  JPEG frames, short side = `size` px (CLIP's processor centre-crops to 224)
 */
export function grabFrames(url, { count = 4, metaTimeoutMs = 8000, timeoutMs = 20000, size = 256 } = {}) {
  const video = document.createElement('video');
  video.crossOrigin = 'anonymous';
  video.muted = true;
  video.playsInline = true;
  video.preload = 'metadata';

  const cleanup = () => { video.removeAttribute('src'); video.load(); };
  const once = event => new Promise((resolve, reject) => {
    const ok = () => { off(); resolve(); };
    const bad = () => { off(); reject(new Error(`video ${event} failed (${video.error?.code ?? '?'})`)); };
    const off = () => { video.removeEventListener(event, ok); video.removeEventListener('error', bad); };
    video.addEventListener(event, ok);
    video.addEventListener('error', bad);
  });

  const work = (async () => {
    video.src = url;
    let metaTimer;
    await Promise.race([
      once('loadedmetadata'),
      new Promise((_, reject) => { metaTimer = setTimeout(() => reject(new Error(`metadata timed out (${metaTimeoutMs / 1000}s)`)), metaTimeoutMs); }),
    ]).finally(() => clearTimeout(metaTimer));
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('unknown duration');
    const scale = size / Math.min(video.videoWidth || size, video.videoHeight || size);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round((video.videoWidth || size) * scale);
    canvas.height = Math.round((video.videoHeight || size) * scale);
    const ctx = canvas.getContext('2d');
    const blobs = [];
    for (let i = 0; i < count; i++) {
      // centres of `count` equal segments: avoids black intro/outro frames at 0% / 100%
      const time = duration * (i + 0.5) / count;
      video.currentTime = time;
      await once('seeked');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('toBlob failed')), 'image/jpeg', 0.9));
      blob.time = time;   // seconds into the video (shown by the map)
      blob.duration = duration;
      blobs.push(blob);
    }
    return blobs;
  })();

  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`frame grab timed out (${timeoutMs / 1000}s)`)), timeoutMs); });
  return Promise.race([work, timeout]).finally(() => { clearTimeout(timer); cleanup(); });
}

/** Average L2-normalised vectors and re-normalise. */
export function meanEmbedding(vectors) {
  const out = new Float32Array(vectors[0].length);
  for (const v of vectors) for (let i = 0; i < out.length; i++) out[i] += v[i];
  let n = 0; for (const x of out) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}
