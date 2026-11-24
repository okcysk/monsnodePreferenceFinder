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
  // added in tag set version 2. Contrast captions name the nearby alternatives (nude, underwear, swimsuit),
  // otherwise e.g. "swimsuit vs ordinary clothes" claims most lightly dressed thumbnails.
  { since: 2, name: '浴室', pos: 'a photo taken in a bathroom | a shower room', neg: 'a photo taken in another room' },
  { since: 2, name: 'ベッド', pos: 'a person on a bed | a bedroom', neg: 'a photo without a bed' },
  { since: 2, name: '車内', pos: 'a photo taken inside a car', neg: 'a photo taken indoors in a room' },
  { since: 2, name: '水着', pos: 'a person in a swimsuit at a beach or pool | a person wearing a bikini', neg: 'a person in clothes indoors | a naked person | a person wearing lingerie' },
  { since: 2, name: 'コスプレ', pos: 'a person in a cosplay costume | a person dressed as an anime character', neg: 'a person in everyday clothes | a person in underwear | a naked person' },
  { since: 2, name: 'ランジェリー', pos: 'a woman wearing lingerie', neg: 'a nude woman | a woman wearing a t-shirt | a woman in a bikini' },
  { since: 2, name: '金髪', pos: 'a person with blonde hair', neg: 'a person with dark hair' },
  { since: 2, name: '自撮り', pos: 'a selfie video taken with a smartphone | a mirror selfie', neg: 'a professionally filmed video | a video filmed by another person' },
  // added in tag set version 3
  { since: 3, name: 'オーラル', pos: 'a woman giving oral sex to a man', neg: 'a man and a woman having intercourse | a nude woman posing alone | a woman taking a selfie | a woman in lingerie' },
  { since: 3, name: 'グラマー', pos: 'a woman with very large breasts and wide hips', neg: 'a woman with small breasts and a slim body | a woman with medium-sized breasts | a woman with an average figure' },
  { since: 3, name: 'スレンダー', pos: 'a slim, slender woman | a thin woman with a slender body', neg: 'a curvy woman with large breasts | a woman with an average body' },
  // added in tag set version 4
  { since: 4, name: 'おもちゃ', pos: 'a woman using a vibrator', neg: 'a woman masturbating with her fingers | a nude woman posing | a man and a woman having sex | a woman in lingerie' },
  { since: 4, name: '拘束', pos: "ropes tied around a woman's body", neg: 'a nude woman without ropes | a woman wearing lingerie | a woman wearing clothes' },
  { since: 4, name: 'アップ', pos: 'an extreme close-up of skin and genitals', neg: 'a full-body shot of a person | a medium shot showing a face and upper body | a wide shot of a room' },
  // added in tag set version 5
  { since: 5, name: '芸術的', pos: 'an artistic fine-art nude photograph with dramatic lighting', neg: 'an amateur smartphone video | a professional porn video | a casual selfie' },
  // added in tag set version 6
  { since: 6, name: '楽しそう', pos: 'a woman smiling and laughing happily', neg: 'a woman with a neutral, expressionless face | a woman with her eyes closed | a woman whose face is not visible' },
  // added in tag set version 9 (two or more people), changed in version 10 to three or more
  { since: 10, name: '複数人', pos: 'a group of several people together | three or more people', neg: 'a single person alone | a couple of two people' },
  // added in tag set version 11
  { since: 11, name: 'ロマンチック', pos: 'a romantic and intimate moment between a couple | a couple embracing tenderly', neg: 'a woman alone | a rough, explicit sex scene | a group of people' },
  // added in tag set version 14
  { since: 14, name: '店舗', pos: 'a photo taken inside a shop or store | a store interior with shelves', neg: 'a photo taken in a bedroom | a photo taken in a bathroom | a photo taken outdoors' },
  // added in tag set version 17
  { since: 17, name: '美しい', pos: 'a stunningly beautiful woman, like a model', neg: 'an average-looking woman | a woman whose face is not visible | a blurry low-quality image' },
  // added in tag set version 18 (captions say "adult woman" on purpose)
  { since: 18, name: 'かわいい', pos: 'an adorable, cute-looking adult woman with a charming face', neg: 'a mature, elegant adult woman | a woman whose face is not visible | a blurry low-quality image' },
];

/** Bumped when default tags change; saved tag lists are updated once (see mergeNewDefaults). */
export const DEFAULT_TAGS_VERSION = 18;

/** Default tags that were dropped (or replaced by a newer version): removed from saved lists only if the user left them exactly as shipped. */
export const REMOVED_DEFAULTS = [
  { removedIn: 6, name: 'おしっこ', pos: 'a woman urinating | a person peeing', neg: 'a woman taking a shower | a nude woman posing | a wet woman in a bath' },
  { removedIn: 8, name: 'おしゃれ', pos: 'a fashionable, stylish woman in trendy clothes', neg: 'a woman in plain everyday clothes | a nude woman | a woman in lingerie' },
  { removedIn: 9, name: 'メガネ', pos: 'a woman wearing eyeglasses | a person with glasses on her face', neg: 'a woman with no glasses | a woman whose face is not visible' },
  { removedIn: 10, name: '複数人', pos: 'two or more people together in one scene', neg: 'a single woman alone | a close-up of one person' },   // replaced by the 3+ version
  { removedIn: 13, name: 'オナニー', pos: 'a woman masturbating alone | a woman touching herself', neg: 'a man and a woman having sex | a woman posing without touching herself | a nude woman lying on a bed | a woman taking a selfie' },
  { removedIn: 16, name: 'AI生成', pos: 'a digital artwork made by AI, CGI render', neg: 'a real photograph | a real video frame | a real photo with a beauty filter | a real professional photo' },
];

/**
 * Bring a saved tag list up to date: append default tags newer than `version` that it does not have
 * (matched by name) and drop removed defaults the user never edited.
 */
export function mergeNewDefaults(saved, version) {
  const dropped = REMOVED_DEFAULTS.filter(r => r.removedIn > version);
  const kept = saved.filter(t => !dropped.some(r => r.name === t.name && r.pos === t.pos && r.neg === t.neg));
  const names = new Set(kept.map(t => t.name));
  const added = DEFAULT_TAGS.filter(t => (t.since ?? 1) > version && !names.has(t.name)).map(({ name, pos, neg }) => ({ name, pos, neg }));
  return { tags: [...kept, ...added], added: added.length, removed: saved.length - kept.length };
}

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
