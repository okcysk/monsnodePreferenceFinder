// Minimal IndexedDB wrapper. Stores: videos (id), embeddings (thumbnail CLIP, id → Float32Array),
// frameEmbeddings (thumbnail + video frames, averaged), kv (session state, settings).
const DB_NAME = 'monsnode-gppl';
const VERSION = 2;
let dbPromise;

function open() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('videos')) db.createObjectStore('videos', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('embeddings')) db.createObjectStore('embeddings', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('kv')) db.createObjectStore('kv');
      if (!db.objectStoreNames.contains('frameEmbeddings')) db.createObjectStore('frameEmbeddings', { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
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
