# Monsnode Preference Finder（ウェブ版）

4本の動画を同時に再生し、好きな順に 1〜4位 を付けるだけで、Pairwise GP 選好学習（GPPL）が一番好きな動画を探すウェブサイトです。インストールは不要で、学習と CLIP 埋め込みはすべてブラウザ内で行います。

**公開URL: https://okcysk.github.io/monsnodePreferenceFinder/**

Chrome 拡張版（`../monsnodeGPPL/`）と中身は同じです。違いは monsnode の読み方だけです。

| 処理 | ウェブ版での方法 |
|---|---|
| 一覧取得 | Cloudflare Worker の `/search` |
| 動画 URL の解決 | Worker の `/resolve` |
| 再生 | twimg を直接再生し、失敗したら Worker の `/media` 経由で再生 |
| サムネ取得（CLIP 用） | `pbs.twimg.com` から直接取得（CORS 対応） |

Worker は Monsnode Multi Player と共用です（`../monsnodeWebPlayer/worker/`）。接続先は `config.js` の `GPPL_API_BASE` で指定します。

## 使い方
- 「プール取得」で候補を集める。初回は CLIP モデル（約90MB）をダウンロードし、以後はブラウザのキャッシュを使う
- 好きな順にクリック、または数字キー 1〜4 で順位を付ける。3つ選ぶと4位は自動で決まり、Enter で送信する
- 再生されない動画は「評価しない」ボタンか Q で外す
- W/A/S/D・Tab でフォーカスを移動する。Space・Z/X・C/V・F/G/H・M・↑↓ はフォーカス中のセルに効き、Shift を付けると4本すべてに効く。`?` で一覧を表示する
- 「マップ」（`viz.html`）で embedding の2次元マップと近傍を確認できる

学習データ（候補・埋め込み・比較履歴）は、このサイトの IndexedDB に保存されます。ブラウザごと・端末ごとに別々で、拡張版とも共有されません。

## 埋め込み方式（ヘッダのプルダウンで切り替え）
- **サムネのみ**（既定）：サムネ1枚を CLIP で埋め込む。速くて通信量も少ない
- **サムネ＋動画4フレーム**：動画を4区間に分け、各区間の中央のフレームを取り出して CLIP で埋め込み、サムネと合わせた5枚を平均する。中身をより反映するが、1本あたり数秒かかる（実測 2〜5秒、300本で十数分）
  - 非表示の `<video crossOrigin="anonymous">` でシークしてフレームを取り出す。twimg の MP4 は目次（moov）が先頭にあり、Range リクエストにも対応しているので、動画全体はダウンロードしない。それでも1本あたり数MBの通信が発生する
  - `video.twimg.com` は CORS を許可しているので、canvas から画素を読める。ただし Referer が付くと 403 になるため、ページで `no-referrer` を指定している
  - 25秒以内にフレームを取れない動画（とても長い動画など）は、この方式の候補から外す
- 埋め込みは方式ごとに別々に保存する。比較履歴は共通なので、方式を切り替えても学習結果を引き継いだまま、新しい特徴量で学習し直す
- マップ（`viz.html`）は、選んでいる方式の埋め込みを表示する

## ローカルで動かす
ES モジュールと Web Worker を使うので、`file://` では動きません。HTTP で配信してください。

```
python -m http.server 8771 --directory monsnodeGPPLWeb
```

`.claude/launch.json` の `monsnode-gppl-web` でも起動できます。

## 公開
静的ファイルだけなので、GitHub Pages などにそのまま置けます。`vendor/ort-wasm-simd-threaded.jsep.wasm`（21MB）は GitHub の1ファイル100MB制限に収まります。
