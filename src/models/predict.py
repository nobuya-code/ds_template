import lightgbm as lgb
import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_DIR, OUTPUT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict_test(models: list[lgb.Booster], df_test: pd.DataFrame) -> np.ndarray:
    """学習済みモデル群を用いて推論を行う"""
    logger.info("Starting prediction on test data")
    
    preds = np.zeros(len(df_test))
    
    for i, model in enumerate(models):
        logger.info(f"Predicting with model {i + 1}/{len(models)}")
        preds += model.predict(df_test, num_iteration=model.best_iteration) / len(models)
        
    logger.info("Finished prediction")
    return preds

def create_submission(preds: np.ndarray, test_ids: pd.Series, file_name: str = "submission.csv") -> None:
    """提出用ファイルを作成する"""
    sub = pd.DataFrame({
        "id": test_ids,
        "target": preds
    })
    out_path = OUTPUT_DIR / file_name
    sub.to_csv(out_path, index=False)
    logger.info(f"Saved submission to {out_path}")

if __name__ == "__main__":
    # 実行例
    # test_features = pd.read_pickle(PROCESSED_DATA_DIR / "test_features.pkl")
    # test_ids = test_features["id"]
    # X_test = test_features.drop(columns=["id"])
    
    # preds = predict_test(models, X_test)
    # create_submission(preds, test_ids)
    pass
