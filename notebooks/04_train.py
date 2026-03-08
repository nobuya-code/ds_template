import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    # import lightgbm as lgb
    # from sklearn.model_selection import KFold
    import pickle
    
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    output_dir = root_dir / "data" / "output"
    
    mo.md(
        r"""
        # 04_train
        
        作成された特徴量データを用いて、機械学習モデルの学習およびクロスバリデーション（交差検証）評価を行います。
        """
    )
    return Path, mo, np, output_dir, pd, pickle, processed_dir, root_dir

@app.cell
def _(mo, pd, processed_dir):
    # train_features = pd.read_pickle(processed_dir / "train_features.pkl")
    mo.md("読み込みセル")
    return

@app.cell
def _(mo):
    # TODO: モデル学習ロジック（LightGBM CV等）の実装
    mo.md("モデル学習セル")
    return

@app.cell
def _(mo, output_dir, pickle):
    # TODO: モデルの保存処理
    mo.md("保存セル")
    return

if __name__ == "__main__":
    app.run()
