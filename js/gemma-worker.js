// Module worker: EmbeddingGemma 2 (text / image / video in one 768-D space) with transformers.js 4.3.1.
// Same message protocol as embed-worker.js (init / embed / embed-video / embed-text), so embed.js can drive either.
// Only the text model and the vision encoder are loaded (q4 ≈ 175 + 109 MB); the audio encoder is skipped.
import { env, AutoConfig, AutoProcessor, AutoModel, RawImage, RawVideo } from '../vendor/tjs4/transformers.min.js';

const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
// WASM: q4 (≈ 175 + 109 MB) on the plain CPU build. WebGPU: fp16 (≈ 540 + 340 MB): the WebGPU-capable builds
// have no kernel for GatherBlockQuantized, which every quantized variant uses (onnxruntime issue #33961).
const DTYPE = { wasm: 'q4', webgpu: 'fp16' };
// Vision tokens per image / video frame (allowed: 70, 140, 280, 560, 1120; model defaults 280 / 140).
// Compute grows with the patch count (tokens × 9), so the smallest budget keeps one video at a few seconds
// instead of a minute; frames are grabbed small anyway (short side 256–384 px).
const VISION_TOKENS = Number(new URLSearchParams(self.location.search).get('tokens')) || 70;
const TEXT_PREFIX = 'task: search result | query: ';   // captions are "queries", videos/images the "documents"
const vendor = new URL('../vendor/tjs4/', import.meta.url).href;
env.allowLocalModels = false;
env.useBrowserCache = true;
// Which onnxruntime build is loaded is decided once, before the first session: plain (CPU only) or asyncify (WebGPU).
const useBuild = name => { env.backends.onnx.wasm.wasmPaths = { mjs: `${vendor}ort-wasm-simd-threaded${name}.mjs`, wasm: `${vendor}ort-wasm-simd-threaded${name}.wasm` }; };
env.backends.onnx.wasm.numThreads = 1;   // pages are not cross-origin isolated
env.backends.onnx.wasm.proxy = false;

let ready = null;
let preferredDevice = 'wasm';
let device = 'wasm';
const progress_callback = p => { if (p.status === 'progress') postMessage({ type: 'model-progress', file: p.file, progress: p.progress }); };

async function webgpuProblem() {
  if (!self.navigator?.gpu) return 'このブラウザでは Worker から WebGPU を使えません（navigator.gpu がありません）';
  try { return (await navigator.gpu.requestAdapter()) ? null : 'GPU アダプタを取得できませんでした'; }
  catch (error) { return `GPU アダプタの取得でエラー: ${error?.message || error}`; }
}

async function load() {
  let fallbackReason = preferredDevice === 'webgpu' ? await webgpuProblem() : null;
  device = preferredDevice === 'webgpu' && !fallbackReason ? 'webgpu' : 'wasm';
  useBuild(device === 'webgpu' ? '.asyncify' : '');
  const processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback });
  processor.image_processor.max_soft_tokens = VISION_TOKENS;
  if (processor.video_processor) processor.video_processor.frame_processor.max_soft_tokens = VISION_TOKENS;
  const config = await AutoConfig.from_pretrained(MODEL_ID, { progress_callback });
  delete config.audio_config;   // the audio encoder session is created only when the config has audio_config
  let model;
  model = await AutoModel.from_pretrained(MODEL_ID, { config, device, dtype: DTYPE[device], progress_callback });
  postMessage({ type: 'ready', device, fallbackReason });
  return { processor, model };
}

function normalized(tensor) {
  const vec = Float32Array.from(tensor.data.subarray(0, tensor.dims.at(-1)));
  let n = 0; for (const v of vec) n += v * v;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= n;
  return vec;
}

/** One sample → one embedding. Inputs go through the processor as (text, images, audio, videos). */
async function embedOne(text, images, videos) {
  ready ??= load();
  const { processor, model } = await ready;
  const inputs = await processor(text, images, null, videos);
  const { sentence_embedding } = await model(inputs);
  return normalized(sentence_embedding);
}

self.onmessage = async ({ data }) => {
  const reply = vecs => postMessage({ type: 'embedded', id: data.id, vecs }, vecs.map(v => v.buffer));
  const fail = error => postMessage({ type: 'embed-error', id: data.id, message: String(error?.message || error) });
  if (data.type === 'init') {
    if (data.device) preferredDevice = data.device;
    ready ??= load();
    try { await ready; } catch (error) { ready = null; postMessage({ type: 'error', message: String(error?.message || error) }); }
    return;
  }
  try {
    if (data.type === 'embed-text') {
      const vecs = [];
      for (const text of data.texts) vecs.push(await embedOne([TEXT_PREFIX + text], null, null));
      reply(vecs);
    } else if (data.type === 'embed') {
      // each blob is its own image sample
      const vecs = [];
      for (const b of data.blobs) vecs.push(await embedOne(null, [await RawImage.fromBlob(b)], null));
      reply(vecs);
    } else if (data.type === 'embed-video') {
      // frames (+ optional poster image) → a single embedding of the whole clip
      const frames = await Promise.all(data.frames.map(b => RawImage.fromBlob(b)));
      const video = new RawVideo(frames, data.duration || frames.length);
      const poster = data.poster ? [await RawImage.fromBlob(data.poster)] : null;
      reply([await embedOne(null, poster, [video])]);
    }
  } catch (error) {
    fail(error);
  }
};
