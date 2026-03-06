import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_data(file_name: str) -> pd.DataFrame:
    """生データを読み込む"""
    file_path = RAW_DATA_DIR / file_name
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な前処理（欠損値処理、型変換など）を行う"""
    logger.info("Starting preprocessing")
    df = df.copy()
    
    # TODO: ここにデータ固有の前処理を実装する
    
    logger.info("Finished preprocessing")
    return df

if __name__ == "__main__":
    # 実行例
    # train_df = load_data("train.csv")
    # train_processed = preprocess_data(train_df)
    # train_processed.to_pickle(PROCESSED_DATA_DIR / "train_processed.pkl")
    pass
