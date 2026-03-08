import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import pickle
    from pathlib import Path
    
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    output_dir = root_dir / "data" / "output"
    
    mo.md(
        r"""
        # 05_predict
        
        学習済みモデルとテストデータの特徴量を読み込み、最終的な推論（予測）を行って提出物（`submission.csv`）を作成します。
        """
    )
    return Path, mo, np, output_dir, pd, pickle, processed_dir, raw_dir

@app.cell
def _(mo, pd, processed_dir, raw_dir):
    # test_features = pd.read_pickle(processed_dir / "test_features.pkl")
    # test_ids = pd.read_csv(raw_dir / "test.csv")["id"]
    mo.md("読み込みセル")
    return

@app.cell
def _(mo, np, output_dir, pickle):
    # TODO: モデル群の読み込みとアンサンブル推論処理
    mo.md("推論セル")
    return

@app.cell
def _(mo, output_dir, pd):
    # sub.to_csv(output_dir / "submission.csv", index=False)
    mo.md("提出ファイル保存セル")
    return

if __name__ == "__main__":
    app.run()
