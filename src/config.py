import os
from pathlib import Path

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(__file__).resolve().parent.parent

# 各種ディレクトリ
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# 乱数シードなどのグローバル定数
RANDOM_STATE = 42

# 必要なディレクトリが存在しない場合は作成
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
