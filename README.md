# Data Science Template

Kaggle等のデータサイエンスコンペティション用の分析環境テンプレートです。
パッケージマネージャーとして `uv` を使用し、インタラクティブなノートブックとして `marimo` を採用しています。データ処理からモデル構築までの各プロセスを `.py` ファイルに分割することで、保守性の高いコードベースを構築できるようにしています。

## 必須要件

- [uv](https://github.com/astral-sh/uv) (依存関係とPythonバージョンの管理)

## 環境構築と起動

このリポジトリを `git clone` してローカル環境に配置した後、以下のコマンドを実行することで、お使いの OS（Windows, Mac, Linux 等）に合わせて仮想環境（`.venv`）が自動的に構築され、必要なパッケージ群が自動インストールされます。

```bash
uv sync
```

以降は `uv run` を用いて、このプロジェクトの仮想環境内でコマンドを実行できます。

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
ノートブックファイルは `marimo edit` で開いて対話的に編集・実行するか、通常のPythonスクリプトとして `python` コマンドで実行可能です。

```bash
# EDA（探索的データ分析）を行う
uv run marimo edit notebooks/01_eda.py

# 前処理ロジックを開発・実行する
uv run marimo edit notebooks/02_preprocess.py

# 特徴量を生成する
uv run marimo edit notebooks/03_features.py

# モデルの学習を行う
uv run marimo edit notebooks/04_train.py

# テストデータ推論と提出ファイル作成を行う
uv run marimo edit notebooks/05_predict.py
```

### パイプラインの一括自動実行

ノートブックでの試行錯誤が終わったあと、手作業で複数のスクリプトを実行するのが手間な場合は、ルートディレクトリの `run_pipeline.py` を用いることで**全自動でのバッチ実行**が可能です。

```bash
# 前処理(02)から推論・提出(05)までを全て一括実行する
uv run python run_pipeline.py

# 特徴量生成(03)から実行し、学習と推論まで完遂する（02の再実行はスキップ）
uv run python run_pipeline.py --start-from 3
```

## ディレクトリ構成

```text
├── data/                  # データ置き場（Git管理外に設定済み）
│   ├── raw/               # コンペティションの元データ（Read Onlyで扱う）
│   ├── processed/         # 前処理済み、または中間データ（.pklなど）
│   └── output/            # 推論結果（submission.csv等）や学習済みモデル
├── notebooks/             # marimoを用いたノートブック環境（メイン開発ディレクトリ）
│   ├── 01_eda.py          # EDA用スクリプト
│   ├── 02_preprocess.py   # データ読み込み・基礎集計・前処理
│   ├── 03_features.py     # 特徴量エンジニアリング
│   ├── 04_train.py        # モデルの学習 (デフォルトはLightGBM CV)
│   └── 05_predict.py      # テストデータ推論・提出ファイル作成
├── src/                   # 共通ユーティリティ（オプショナル）
│   ├── config.py          # ディレクトリパス等の定数管理
│   └── utils/             
│       └── logger.py      # ロガー作成などの共通ユーティリティ
├── pyproject.toml         # uv によるプロジェクト情報および依存パッケージ管理
└── README.md              # このファイル
```
