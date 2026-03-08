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
    test_features = pd.read_pickle(processed_dir / "test_features.pkl")
    
    # IDの取得のために元のtestデータを読み込む
    test_raw = pd.read_csv(raw_dir / "test.csv")
    test_ids = test_raw["id"]
    
    mo.md(f"テストデータの読み込み完了\n- test_features: {test_features.shape}")
    return test_features, test_ids, test_raw


@app.cell
def _(mo, np, output_dir, pickle, test_features):
    models = []
    
    # K-Fold学習で保存した5つのモデルを読み込み
    for i in range(5):
        with open(output_dir / f"lgb_model_fold{i}.pkl", "rb") as f:
            models.append(pickle.load(f))
            
    preds = np.zeros(len(test_features))
    
    # 全モデルの予測値の平均（アンサンブル）
    for _model in models:
        preds += _model.predict(test_features, num_iteration=_model.best_iteration) / len(models)
        
    mo.md(f"全Fold（{len(models)}モデル）によるアンサンブル推論完了")
    return f, i, models, preds


@app.cell
def _(mo, output_dir, pd, preds, test_ids):
    import subprocess
    import glob
    import os
    
    # --- 1. Gitのコミット情報の自動取得 ---
    try:
        # 最新の短いハッシュを取得
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        
        # 未コミットの変更があるかチェック
        git_status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        is_dirty = len(git_status) > 0
        if is_dirty:
            git_hash = f"{git_hash}_dirty"
            
        # 最新のコミットメッセージを取得し、ファイル名に使える安全な文字列（英数字とアンダースコアのみ等）にする
        git_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%s"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        safe_msg = "".join([c if c.isalnum() else "_" for c in git_msg])[:20]  # 長すぎないように20文字でカット
    except Exception:
        # Git管理されていない場合やエラー時はデフォルト値
        git_hash = "nohash"
        safe_msg = "local"
        
    # --- 2. 出力ディレクトリ内の既存 submission ファイルから最大連番を取得 ---
    existing_subs = glob.glob(str(output_dir / "submission_*.csv"))
    max_num = 0
    for sub_file in existing_subs:
        filename = os.path.basename(sub_file)
        # submission_001_xxxx.csv の 001 部分を取り出す
        parts = filename.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            num = int(parts[1])
            max_num = max(max_num, num)
            
    # 新しいバージョン番号
    next_num = max_num + 1
    
    # 最終的なファイル名の構築 (例: submission_002_a1b2c3d_add_milage.csv)
    sub_filename = f"submission_{next_num:03d}_{git_hash}_{safe_msg}.csv"
    out_path = output_dir / sub_filename
    
    sub = pd.DataFrame({
        "id": test_ids,
        "price": preds
    })
    
    sub.to_csv(out_path, index=False)
    
    mo.md(f"提出ファイルを自動ナンバリングで保存しました: `{out_path}`")
    return existing_subs, filename, git_hash, git_msg, max_num, next_num, os, out_path, parts, safe_msg, sub, sub_filename, subprocess


if __name__ == "__main__":
    app.run()
