# Data Science Template

Kaggle等のデータサイエンスコンペティション用の分析環境テンプレートです。
パッケージマネージャーとして `uv` を使用し、インタラクティブなノートブックとして `marimo` を採用しています。データ処理からモデル構築までの各プロセスを `.py` ファイルに分割することで、保守性の高いコードベースを構築できるようにしています。

## 必須要件

- [uv](https://github.com/astral-sh/uv) (依存関係とPythonバージョンの管理)

## 環境構築と起動

すでに `uv init` によって初期化と主要なライブラリ群のインストールは完了しています。以降は `uv run` を用いて、このプロジェクトの仮想環境内でコマンドを実行できます。

### 新しいライブラリの追加

```bash
uv add <package_name>
```

### ノートブック (marimo) の起動

EDA（探索的データ分析）を行ったりコードを可視化したりする場合は、以下のコマンドで `marimo` を起動します。

```bash
uv run marimo edit notebooks/01_eda.py
```

### スクリプトの実行例

各処理のテンプレートファイルは以下のように順番に実行・開発を進めることができます。
モジュール名前空間（`src.`）を正しく解決するため、`-m` フラグを利用して実行してください。

```bash
# 1. 生データを読み込んで前処理を行う
uv run python -m src.data.preprocess

# 2. 前処理済みデータから特徴量を生成する
uv run python -m src.features.build_features

# 3. K-Fold等をもちいてモデルの学習とローカル評価を行う
uv run python -m src.models.train

# 4. テストデータに対して推論を行い、提出用ファイルを作成する
uv run python -m src.models.predict
```

## ディレクトリ構成

```text
├── data/                  # データ置き場（Git管理外に設定済み）
│   ├── raw/               # コンペティションの元データ（Read Onlyで扱う）
│   ├── processed/         # 前処理済み、または中間データ（.pklなど）
│   └── output/            # 推論結果（submission.csv等）や学習済みモデル
├── notebooks/             # marimoを用いたノートブック環境
│   └── 01_eda.py          # EDA用の雛形スクリプト
├── src/                   # ソースコード
│   ├── config.py          # ディレクトリパス等の定数管理
│   ├── data/              
│   │   └── preprocess.py  # データ読み込み・基礎集計・前処理
│   ├── features/          
│   │   └── build_features.py # 特徴量エンジニアリング
│   ├── models/            
│   │   ├── train.py       # モデルの学習 (デフォルトはLightGBM CV)
│   │   └── predict.py     # テストデータ推論・提出ファイル作成
│   └── utils/             
│       └── logger.py      # ロガー作成などの共通ユーティリティ
├── pyproject.toml         # uv によるプロジェクト情報および依存パッケージ管理
└── README.md              # このファイル
```
