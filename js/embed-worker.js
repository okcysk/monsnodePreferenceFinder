// Module worker: CLIP ViT-B/32 image embeddings with transformers.js (vendored, no remote code).
import { env, AutoProcessor, CLIPVisionModelWithProjection, RawImage } from '../vendor/transformers.min.js';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
env.allowLocalModels = false;
env.useBrowserCache = true;                       // weights cached in Cache Storage after first download
env.backends.onnx.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
env.backends.onnx.wasm.numThreads = 1;            // extension pages are not cross-origin isolated
env.backends.onnx.wasm.proxy = false;

let ready = null;
let device = 'wasm';

// Default: q8 on WASM (~90MB download, ~0.25s/image). 'webgpu' uses fp32 (~350MB download, faster; falls back to WASM).
let preferredDevice = 'wasm';

async function load() {
  const hasWebGPU = !!(self.navigator && navigator.gpu && await navigator.gpu.requestAdapter().catch(() => null));
  device = preferredDevice === 'webgpu' && hasWebGPU ? 'webgpu' : 'wasm';
  const progress_callback = p => { if (p.status === 'progress') postMessage({ type: 'model-progress', file: p.file, progress: p.progress }); };
  const processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback });
  let model;
  try {
    model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { device, dtype: device === 'webgpu' ? 'fp32' : 'q8', progress_callback });
  } catch (error) {
    if (device !== 'webgpu') throw error;
    device = 'wasm';
    model = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { device, dtype: 'q8', progress_callback });
  }
  postMessage({ type: 'ready', device });
  return { processor, model };
}

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    if (data.device) preferredDevice = data.device;
    ready ??= load();
    try { await ready; } catch (error) { postMessage({ type: 'error', message: String(error?.message || error) }); }
    return;
  }
  if (data.type === 'embed') {
    // `blobs`: one or more images run as a single batch (one processor + one model call).
    try {
      ready ??= load();
      const { processor, model } = await ready;
      const images = await Promise.all(data.blobs.map(b => RawImage.fromBlob(b)));
      const inputs = await processor(images);
      const { image_embeds } = await model(inputs);
      const dim = image_embeds.dims.at(-1);
      const vecs = images.map((_, k) => {
        const vec = Float32Array.from(image_embeds.data.subarray(k * dim, (k + 1) * dim));
        let norm = 0; for (const v of vec) norm += v * v;
        norm = Math.sqrt(norm) || 1;
        for (let i = 0; i < vec.length; i++) vec[i] /= norm;
        return vec;
      });
      postMessage({ type: 'embedded', id: data.id, vecs }, vecs.map(v => v.buffer));
    } catch (error) {
      postMessage({ type: 'embed-error', id: data.id, message: String(error?.message || error) });
    }
  }
};
