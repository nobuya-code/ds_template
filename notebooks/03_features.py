import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    
    root_dir = Path(__file__).resolve().parent.parent
    processed_dir = root_dir / "data" / "processed"
    
    mo.md(
        r"""
        # 03_features
        
        前処理済みのデータを読み込み、新規変数の作成やカテゴリカル変数のエンコーディングを行います。
        """
    )
    return Path, mo, pd, processed_dir, root_dir

@app.cell
def _(mo, pd, processed_dir):
    # train_processed = pd.read_pickle(processed_dir / "train_processed.pkl")
    # test_processed = pd.read_pickle(processed_dir / "test_processed.pkl")
    mo.md("読み込みセル")
    return

@app.cell
def _(mo):
    # TODO: 特徴量エンジニアリングを実装
    mo.md("特徴量作成セル")
    return

@app.cell
def _(mo, processed_dir):
    # train_features.to_pickle(processed_dir / "train_features.pkl")
    # test_features.to_pickle(processed_dir / "test_features.pkl")
    mo.md("保存セル")
    return

if __name__ == "__main__":
    app.run()
