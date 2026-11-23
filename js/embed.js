// Page-side wrapper around embed-worker.js: fetches thumbnails and queues them one by one.
export class Embedder {
  /** @param {{onModelProgress?, onReady?, device?: 'wasm'|'webgpu', preload?: boolean}} opts  preload=false: load the image model only when first used (e.g. the map, which only needs text) */
  constructor({ onModelProgress, onReady, device = 'wasm', preload = true } = {}) {
    this.worker = new Worker(new URL('./embed-worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.seq = 0;
    this.device = null;
    this.worker.onmessage = ({ data }) => {
      if (data.type === 'model-progress') onModelProgress?.(data);
      else if (data.type === 'ready') { this.device = data.device; onReady?.(data.device, data.fallbackReason); }
      else if (data.type === 'error') this.initError = new Error(data.message);
      else if (data.type === 'embedded' || data.type === 'embed-error') {
        const p = this.pending.get(data.id);
        if (!p) return;
        this.pending.delete(data.id);
        if (data.type === 'embedded') p.resolve(data.vecs); else p.reject(new Error(data.message));
      }
    };
    if (preload) this.worker.postMessage({ type: 'init', device });
  }

  async embedThumbnail(id, url) {
    const res = await fetch(url, { credentials: 'omit' });
    if (!res.ok) throw new Error(`Thumbnail ${res.status}`);
    return this.embedBlob(await res.blob());
  }

  /** Embed one image blob (thumbnail or a video frame). */
  async embedBlob(blob) { return (await this.request([blob]))[0]; }

  /**
   * Embed several images. On WebGPU (fp32) they go as one batch: same vectors as one by one, ~15% faster.
   * On WASM (q8) a batch changes the vectors (activation quantization depends on the batch) and is barely
   * faster, so images are sent one by one there. Requests are queued in the worker in order.
   */
  async embedBlobs(blobs) {
    if (this.device !== 'webgpu') {
      const out = [];
      for (const b of blobs) out.push(...await this.request([b]));
      return out;
    }
    return this.request(blobs);
  }

  request(blobs) { return this.send({ type: 'embed', blobs }); }

  /** CLIP text embeddings (L2-normalised), in the same space as the image embeddings. */
  embedTexts(texts) { return this.send({ type: 'embed-text', texts }); }

  send(message) {
    const key = `r${++this.seq}`;
    return new Promise((resolve, reject) => {
      this.pending.set(key, { resolve, reject });
      this.worker.postMessage({ ...message, id: key });
    });
  }
}
