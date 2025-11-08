 # soccer_support_AI

**放送映像からサッカー選手の動きを座標化し，オンボール／オフボール双方の貢献度を数値化する研究用リポジトリ**

> 目的：単眼（放送）映像だけから，任意選手のトラッキング→ピッチ座標へのマッピング→xT/VAEP/Pitch Control 等の指標算出までを，個人でも再現可能な最小パイプラインとして公開・検証する。

---

## 🔍 背景

野球ではWARなどの総合指標が一般化している一方，サッカーでは**オフボール貢献の定量化**が難題です。本プロジェクトは，

* **単眼映像**から選手位置を復元（検出・追跡・再識別・ホモグラフィ）
* **オンボール価値**（例：xT，VAEP）と**オフボール価値**（例：Pitch Control）を統合
* 選手ごとの**総合貢献スコア**を算出

する実験基盤（research code）を整備します。

---

## ✨ できること（MVP範囲）

* 放送映像（mp4）からの**人物検出**（YOLO系）
* **多人数追跡**（ByteTrack/StrongSORT想定）＋**再識別（Re-ID）**でIDの継続
* **手動または自動ホモグラフィ**でピクセル→ピッチ座標変換
* **任意選手の半自動追跡**（クリックで指定→Re-IDで追随）
* **簡易xT算出**（ゾーン価値Tを学習/既知テーブルから参照し，ボール位置の遷移価値を加算）
* （拡張予定）**Pitch Control** によるオフボール価値推定，**VAEP** 実装，**ボール追跡**

---

## 🧱 アーキテクチャ／パイプライン

```
video(mp4)
  └─ detection(YOLO) ─> tracking(ByteTrack/StrongSORT)
                         └─ re-id(optional)
  └─ field calibration(homography) ─> image xy → pitch XY
  └─ (ball detection) → possession inference
  └─ metrics: xT / VAEP / Pitch Control
  └─ aggregation: per-action / per-possession / per-player / per-game
```

---

## 📦 ディレクトリ構成（案）

```
.
├─ data/
│  ├─ raw/               # 入力映像，アノテーション
│  ├─ interim/           # 中間成果物（BB, tracks, homography）
│  └─ processed/         # ピッチ座標CSV，イベントCSV
├─ models/               # Re-ID/検出器のweights
├─ soccer/
│  ├─ scripts/           # CLIスクリプト群（run_detect.py など）
│  ├─ core/              # 追跡・座標変換・指標計算の実装
│  └─ notebooks/         # 実験ノート
├─ configs/              # モデル・閾値・xTテーブル等
├─ tests/                # 単体テスト
└─ README.md
```

---

## ⚙️ セットアップ

* Python: 3.10+
* 推奨：`uv` or `conda` の仮想環境（GPUがあればCUDA対応PyTorch）

```bash
# 例: uv (高速パッケージ管理)
uv venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate
uv pip install -r requirements.txt

# 例: conda
conda create -n soccer python=3.10 -y
conda activate soccer
pip install -r requirements.txt
```

> **注意**：YOLO/ByteTrack/StrongSORTやMMTracking等，外部ライブラリのバージョン整合が必要です。`requirements.txt` と `configs/` に固定します。

---

## 🚀 クイックスタート

**1) 検出＋追跡**

```bash
python soccer/scripts/run_detect_track.py \
  --input data/raw/sample_match.mp4 \
  --detector yolov8n \
  --tracker bytetrack \
  --out data/interim/sample_tracks.json
```

**2) ホモグラフィ（手動）**

```bash
python soccer/scripts/run_homography.py \
  --input data/raw/sample_frame.png \
  --point-csv data/interim/corner_points.csv \
  --out configs/homography_sample.yaml
```

**3) ピッチ座標に射影**

```bash
python soccer/scripts/warp_to_pitch.py \
  --tracks data/interim/sample_tracks.json \
  --H configs/homography_sample.yaml \
  --out data/processed/sample_xy.csv
```

**4) 簡易xTの計算**

```bash
python soccer/scripts/compute_xt.py \
  --xy data/processed/sample_xy.csv \
  --xt-table configs/xt_table.csv \
  --out data/processed/sample_xt_player.csv
```

（任意）**Pitch Control** や **VAEP** を使う場合は，`soccer/core/metrics/` の設定を参照してください。

---

## 📊 指標の考え方（簡潔版）

* **xT**：ピッチを格子化し，その地点から得点につながる脅威の期待値を学習・付与。ボールをより脅威の高いゾーンへ運ぶほど加点。
* **VAEP**：各アクションがチームの得点/失点確率をどれだけ変化させたかを推定し，価値として合算。
* **Pitch Control**：選手位置・速度から「いまボールが落ちたらどのチームが先に支配するか」の確率場を計算し，スペース創出/遮断を可視化。

> 本リポジトリではまず **xT（MVP）** を再現し，順次 **Pitch Control** → **VAEP** の順に拡張予定です。

---

## 🔬 研究上の留意点

* 放送映像は**カメラ切替・被遮蔽**が多く，IDロストが避けられません → Re-IDや背番号OCRで補完。
* **ホモグラフィ誤差**は最終スコアに伝搬します → 白線点の精度・本数・分布を確保。
* データ・モデルの**再現性**を担保するため，**乱数種**と**バージョン**を固定します。

---

## 📈 ロードマップ

* [ ] MVP：検出・追跡・手動ホモグラフィ・簡易xT
* [ ] ボール検出・ポゼッション推定
* [ ] Pitch Control 実装（全員座標の推定安定化）
* [ ] VAEP 近似（オンボールイベント抽出）
* [ ] 評価：公開データでの定量検証（精度/速度/再現性）

---

## 📚 参考・関連（例示）

* Expected Threat (xT)
* Valuing Actions by Estimating Probabilities (VAEP)
* Pitch Control / Friends of Tracking 解説
* SoccerNet（Tracking / Re-ID / Calibration）

※ 論文・実装リンクは `docs/references.md` に整理予定。

---

## 💾 データと権利

* 放送映像の著作権・肖像権に留意してください。研究目的の私的利用を前提とし，**成果物の公開時は権利者ガイドラインに従う**こと。
* 公開データセット（例：SoccerNet）の利用規約を必ず確認してください。

---

## 🧪 テスト

```bash
pytest -q
```

---

## 🤝 貢献

Issue/PR歓迎です。コーディング規約は `CONTRIBUTING.md` を参照してください（準備中）。

---

## 🔐 ライセンス

MIT（予定）。詳細は `LICENSE` を参照。

---

## 🙌 謝辞

本プロジェクトは，公開研究・コミュニティの知見（検出・追跡・再識別・座標化・戦術指標）に学び，個人が学習・検証できる**最小構成**を目指しています。