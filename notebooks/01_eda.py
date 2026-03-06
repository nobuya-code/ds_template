import marimo

__generated_with = "0.9.14"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt
    
    # プロジェクトルートにパスを通す
    import sys
    from pathlib import Path
    
    # src以下のモジュールを読み込めるようにパスを追加
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
        
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from src.utils.logger import get_logger
    
    # marimoでmatplotlibの表示設定
    try:
        sns.set_theme(style="whitegrid")
    except:
        pass

    return Path, PROCESSED_DATA_DIR, RAW_DATA_DIR, alt, get_logger, mo, np, pd, plt, root_dir, sns, sys


@app.cell
def __(mo):
    mo.md(
        r"""
        # 探索的データ分析 (EDA)
        
        このノートブックでは、コンペティションの元データの基本的な傾向や欠損値を調査します。
        """
    )
    return


@app.cell
def __(RAW_DATA_DIR, mo, pd):
    # 生データの読み込み例（ファイル名を適宜変更してください）
    # train_df = pd.read_csv(RAW_DATA_DIR / "train.csv")
    
    mo.md(
        r"""
        ## データの読み込み
        
        まずは `raw` フォルダからデータを読み込みます。
        パスは `src.config.RAW_DATA_DIR` から取得しています。
        """
    )
    return


@app.cell
def __():
    # グラフの描画例
    # sns.histplot(train_df["target"], kde=True)
    # plt.show()
    pass
    return


if __name__ == "__main__":
    app.run()
