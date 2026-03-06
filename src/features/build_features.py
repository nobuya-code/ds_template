import pandas as pd
from src.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量エンジニアリングを行う"""
    logger.info("Starting feature engineering")
    df = df.copy()
    
    # TODO: ここに特徴量の追加処理を実装する
    
    logger.info(f"Finished feature engineering. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # 実行例
    # train_processed = pd.read_pickle(PROCESSED_DATA_DIR / "train_processed.pkl")
    # train_features = create_features(train_processed)
    # train_features.to_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    pass
