import marimo

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    
    root_dir = Path(__file__).resolve().parent.parent
    raw_dir = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    
    mo.md(
        r"""
        # 02_preprocess
        
        生のデータを読み込み、欠損値の補完や不要な列の削除といった初期の前処理を実施します。
        実行後は結果を `processed/` フォルダへ書き出します。
        """
    )
    return Path, mo, pd, processed_dir, raw_dir, root_dir

@app.cell
def _(mo, pd, raw_dir):
    # train_df = pd.read_csv(raw_dir / "train.csv")
    # test_df = pd.read_csv(raw_dir / "test.csv")
    mo.md("生データの読み込みセル (コメントアウトを外して使用)")
    return

@app.cell
def _(mo):
    def preprocess_data(df):
        df = df.copy()
        # TODO: IDの削除や欠損値補完などを実装
        return df
        
    mo.md("前処理の適用セル")
    return preprocess_data, 

@app.cell
def _(mo, processed_dir):
    # train_processed.to_pickle(processed_dir / "train_processed.pkl")
    # test_processed.to_pickle(processed_dir / "test_processed.pkl")
    mo.md(f"保存セル（コメントアウトを外して使用）")
    return

if __name__ == "__main__":
    app.run()
