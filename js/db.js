// Minimal IndexedDB wrapper. Stores: videos (id), kv (session state, settings) and embeddings:
//   embeddings / frameEmbeddings          CLIP q8 on WASM   (thumbnail / thumbnail + video frames)
//   embeddingsGpu / frameEmbeddingsGpu    CLIP fp32 on WebGPU
//   gemmaEmbeddings / gemmaEmbeddingsGpu  EmbeddingGemma 2 (q4 on WASM / fp16 on WebGPU) of 4 frames + thumbnail
// q8 and fp32 vectors of the same image differ (cos ≈ 0.9), so they are never mixed in one pool.
const DB_NAME = 'monsnode-gppl';
const VERSION = 4;
let dbPromise;

function open() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('videos')) db.createObjectStore('videos', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('embeddings')) db.createObjectStore('embeddings', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('kv')) db.createObjectStore('kv');
      for (const name of ['frameEmbeddings', 'embeddingsGpu', 'frameEmbeddingsGpu', 'gemmaEmbeddings', 'gemmaEmbeddingsGpu'])
        if (!db.objectStoreNames.contains(name)) db.createObjectStore(name, { keyPath: 'id' });
    };
    req.onsuccess = () => {
      // Another tab (or a newer version of this page) wants to upgrade the schema: let it, instead of blocking it.
      req.result.onversionchange = () => { req.result.close(); dbPromise = null; };
      resolve(req.result);
    };
    req.onblocked = () => console.warn('[db] upgrade waiting for another tab to close the old version');
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function wrap(req) { return new Promise((resolve, reject) => { req.onsuccess = () => resolve(req.result); req.onerror = () => reject(req.error); }); }

async function store(name, mode = 'readonly') { return (await open()).transaction(name, mode).objectStore(name); }

export async function getAll(name) { return wrap((await store(name)).getAll()); }
export async function put(name, value) { return wrap((await store(name, 'readwrite')).put(value)); }
export async function putMany(name, values) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(name, 'readwrite');
    const s = tx.objectStore(name);
    for (const v of values) s.put(v);
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error);
  });
}
export async function clear(name) { return wrap((await store(name, 'readwrite')).clear()); }
export async function kvGet(key) { return wrap((await store('kv')).get(key)); }
export async function kvSet(key, value) { return wrap((await store('kv', 'readwrite')).put(value, key)); }
