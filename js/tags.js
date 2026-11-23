// Zero-shot tags with CLIP: each tag is a pair of English captions (positive vs. contrast).
// For an image embedding x and caption embeddings p (positive) and n (contrast), all L2-normalised,
//   P(tag) = softmax(100·[x·p, x·n])[0] = sigmoid(100·(x·p − x·n))
// 100 is CLIP's logit scale. Several captions separated by "|" are averaged (prompt ensembling).

export const DEFAULT_TAGS = [
  {
    name: 'アニメ',
    pos: 'an anime illustration | a cartoon drawing | a 3D CG animation render',
    neg: 'a real photograph of a person | a photo of real people',
  },
  {
    name: '露出',
    pos: 'a photo of a person outdoors in a public place | a person on the street or in nature',
    neg: 'a photo of a person indoors in a room | a person in a bedroom',
  },
  {
    name: '海外',
    pos: 'a photo of a caucasian person | a photo of a western person',
    neg: 'a photo of an east asian person | a photo of a japanese person',
  },
];

export const splitPrompts = s => String(s || '').split('|').map(t => t.trim()).filter(Boolean);

function meanNormalized(vectors) {
  const out = new Float32Array(vectors[0].length);
  for (const v of vectors) for (let i = 0; i < out.length; i++) out[i] += v[i];
  let n = 0; for (const x of out) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/**
 * @param {{name: string, pos: string, neg: string}[]} tags
 * @param {ArrayLike<number>[]} imageVecs  L2-normalised CLIP image embeddings
 * @param {(texts: string[]) => Promise<Float32Array[]>} embedTexts
 * @returns {Promise<Float32Array[]>}  per tag, a probability in [0, 1] for each image
 */
export async function scoreTags(tags, imageVecs, embedTexts) {
  const valid = tags.filter(t => splitPrompts(t.pos).length && splitPrompts(t.neg).length);
  const texts = valid.flatMap(t => [...splitPrompts(t.pos), ...splitPrompts(t.neg)]);
  const vecs = texts.length ? await embedTexts(texts) : [];
  let k = 0;
  const byTag = new Map();
  for (const t of valid) {
    const np = splitPrompts(t.pos).length, nn = splitPrompts(t.neg).length;
    const p = meanNormalized(vecs.slice(k, k + np)); k += np;
    const n = meanNormalized(vecs.slice(k, k + nn)); k += nn;
    byTag.set(t, Float32Array.from(imageVecs, x => 1 / (1 + Math.exp(-100 * (dot(x, p) - dot(x, n))))));
  }
  return tags.map(t => byTag.get(t) ?? null);
}

/** Cosine similarity of each image to a free-text query (prompt-ensembled with "|"). */
export async function scoreText(query, imageVecs, embedTexts) {
  const prompts = splitPrompts(query);
  if (!prompts.length) return null;
  const q = meanNormalized(await embedTexts(prompts));
  return Float32Array.from(imageVecs, x => dot(x, q));
}
