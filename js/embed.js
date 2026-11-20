// Page-side wrapper around embed-worker.js: fetches thumbnails and queues them one by one.
export class Embedder {
  constructor({ onModelProgress, onReady, device = 'wasm' } = {}) {
    this.worker = new Worker(new URL('./embed-worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.seq = 0;
    this.device = null;
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'model-progress') onModelProgress?.(data);
      else if (data.type === 'ready') { this.device = data.device; onReady?.(data.device); }
      else if (data.type === 'error') this.initError = new Error(data.message);
      else if (data.type === 'embedded' || data.type === 'embed-error') {
        const p = this.pending.get(data.id);
        if (!p) return;
        this.pending.delete(data.id);
        if (data.type === 'embedded') p.resolve(data.vec); else p.reject(new Error(data.message));
      }
    };
    this.worker.postMessage({ type: 'init', device });
  }

  async embedThumbnail(id, url) {
    const res = await fetch(url, { credentials: 'omit' });
    if (!res.ok) throw new Error(`Thumbnail ${res.status}`);
    return this.embedBlob(await res.blob());
  }

  /** Embed any image blob (thumbnail or a video frame). Requests are queued in the worker in order. */
  embedBlob(blob) {
    const key = `r${++this.seq}`;
    return new Promise((resolve, reject) => {
      this.pending.set(key, { resolve, reject });
      this.worker.postMessage({ type: 'embed', id: key, blob });
    });
  }
}
