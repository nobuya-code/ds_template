import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from src.config import PROCESSED_DATA_DIR, OUTPUT_DIR, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_cv(df: pd.DataFrame, target_col: str, n_splits: int = 5) -> list[lgb.Booster]:
    """K-Foldを用いた交差検証によるモデル学習"""
    logger.info(f"Starting Cross Validation training (n_splits={n_splits})")
    
    features = [c for c in df.columns if c != target_col]
    X = df[features]
    y = df[target_col]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    models = []
    oof_preds = np.zeros(len(df))
    
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        logger.info(f"Training Fold {fold + 1}/{n_splits}")
        
        model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds
        
        fold_rmse = root_mean_squared_error(y_val, val_preds)
        logger.info(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")
        
        models.append(model)
        
    oof_rmse = root_mean_squared_error(y, oof_preds)
    logger.info(f"OOF RMSE: {oof_rmse:.4f}")
    
    # 必要ならモデルを OUTPUT_DIR に保存する
    
    return models

if __name__ == "__main__":
    # 実行例
    # train_features = pd.read_pickle(PROCESSED_DATA_DIR / "train_features.pkl")
    # target_col = "target"
    # models = train_cv(train_features, target_col)
    pass
