import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
    from sklearn.metrics import root_mean_squared_error, roc_auc_score, log_loss
    from pathlib import Path
    import pickle
    
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    output_dir = root_dir / "data" / "output"
    
    mo.md(
        r"""
        # 04_train
        
        作成された特徴量データを用いて、機械学習モデルの学習およびクロスバリデーション（交差検証）評価を行います。
        様々なタスク（回帰・分類）に応じたCV戦略や評価指標を設定可能です。
        """
    )
    return KFold, Path, StratifiedKFold, TimeSeriesSplit, lgb, log_loss, mo, np, output_dir, pd, pickle, processed_dir, roc_auc_score, root_dir, root_mean_squared_error


@app.cell
def _(mo):
    # 【全体設定】コンペティションのタスクに合わせた設定項目
    CONFIG = {
        "target_col": "price",               # 目的変数のカラム名
        "problem_type": "regression",       # "regression" or "binary" or "multiclass"
        "cv_method": "KFold",               # "KFold", "StratifiedKFold", "TimeSeriesSplit"
        "eval_metric": "rmse",              # "rmse", "auc", "logloss" 等
        "n_splits": 5,                      # 分割数
        "random_state": 42,
    }
    
    mo.md(f"**現在の学習設定:**\n- タスク: `{CONFIG['problem_type']}`\n- CV: `{CONFIG['cv_method']}` ({CONFIG['n_splits']} Folds)\n- 評価指標: `{CONFIG['eval_metric']}`")
    return CONFIG,


@app.cell
def _(mo, pd, processed_dir):
    train_features = pd.read_pickle(processed_dir / "train_features.pkl")
    
    mo.md(f"特徴量データの読み込み完了\n- train_features: {train_features.shape}")
    return train_features,


@app.cell
def _(CONFIG, KFold, StratifiedKFold, TimeSeriesSplit, log_loss, roc_auc_score, root_mean_squared_error):
    # CVストラテジーと評価指標の共通化関数
    def get_cv_strategy(X, y):
        if CONFIG["cv_method"] == "KFold":
            return KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["random_state"])
        elif CONFIG["cv_method"] == "StratifiedKFold":
            return StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["random_state"])
        elif CONFIG["cv_method"] == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=CONFIG["n_splits"])
        else:
            raise ValueError(f"Unknown CV method: {CONFIG['cv_method']}")
            
    def calculate_score(y_true, y_pred):
        if CONFIG["eval_metric"] == "rmse":
            return root_mean_squared_error(y_true, y_pred)
        elif CONFIG["eval_metric"] == "auc":
            return roc_auc_score(y_true, y_pred)
        elif CONFIG["eval_metric"] == "logloss":
            return log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown evaluation metric: {CONFIG['eval_metric']}")
            
    return calculate_score, get_cv_strategy


@app.cell
def _(CONFIG, calculate_score, get_cv_strategy, lgb, np, train_features):
    features = [c for c in train_features.columns if c != CONFIG["target_col"]]
    X = train_features[features]
    y = train_features[CONFIG["target_col"]]
    
    kf = get_cv_strategy(X, y)
    models = []
    
    train_scores = []
    val_scores = []
    oof_preds = np.zeros(len(train_features))
    
    # LightGBMの基本パラメータ
    lgb_params = {
        "learning_rate": 0.05,
        "seed": CONFIG["random_state"],
        "verbose": -1,
    }
    
    if CONFIG["problem_type"] == "regression":
        lgb_params["objective"] = "regression"
        lgb_params["metric"] = "rmse"
    elif CONFIG["problem_type"] == "binary":
        lgb_params["objective"] = "binary"
        lgb_params["metric"] = "binary_logloss"
    elif CONFIG["problem_type"] == "multiclass":
        lgb_params["objective"] = "multiclass"
        lgb_params["num_class"] = train_features[CONFIG["target_col"]].nunique()
    
    print("-" * 50)
    print(f"🚀 学習開始: {CONFIG['problem_type']} | {CONFIG['cv_method']}")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=10000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0) # 学習中の詳細ログは非表示にし、結果出力のみ行う
            ]
        )
        
        # TrainとVal双方の推論とスコア計算
        trn_preds = model.predict(X_train, num_iteration=model.best_iteration)
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        
        trn_score = calculate_score(y_train, trn_preds)
        val_score = calculate_score(y_val, val_preds)
        
        train_scores.append(trn_score)
        val_scores.append(val_score)
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        print(f"Fold {fold} - Train Score: {trn_score:.4f} | Val Score: {val_score:.4f}")
        
    print("-" * 50)
    print(f"✅ Overall Train Score : {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
    print(f"✅ Overall Val Score   : {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    
    oof_score = calculate_score(y, oof_preds)
    print(f"🏆 OOF Score (Total)   : {oof_score:.4f}")
    return X, X_train, X_val, features, fold, kf, lgb_params, model, models, oof_preds, oof_score, train_data, train_idx, train_scores, trn_preds, trn_score, val_data, val_idx, val_preds, val_score, val_scores, y, y_train, y_val


@app.cell
def _(mo, models, oof_score, output_dir, pickle):
    for i, _model in enumerate(models):
        _model_path = output_dir / f"lgb_model_fold{i}.pkl"
        with open(_model_path, "wb") as f:
            pickle.dump(_model, f)
            
    mo.md(f"**モデル保存完了!**\n- OOF スコア: `{oof_score:.4f}`\n- `{len(models)}` Fold分の学習済みモデルを `{output_dir}` に保存しました。")
    return f, i


if __name__ == "__main__":
    app.run()
