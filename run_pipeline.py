import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="各プロセスのNotebookを一括で順次実行するバッチスクリプト")
    parser.add_argument(
        "--start-from",
        type=int,
        choices=[2, 3, 4, 5],
        default=2,
        help="実行を開始するステップ番号 (2: preprocess, 3: features, 4: train, 5: predict)",
    )
    args = parser.parse_args()

    steps = [
        (2, "notebooks/02_preprocess.py"),
        (3, "notebooks/03_features.py"),
        (4, "notebooks/04_train.py"),
        (5, "notebooks/05_predict.py"),
    ]

    print(f"🚀 パイプラインの実行を ステップ {args.start_from} から開始します...")
    print("-" * 50)

    for step_num, script_path in steps:
        if step_num >= args.start_from:
            print(f"\n▶ [{step_num}/5] 実行中: {script_path} ...")
            
            # subprocess.runでuv経由でPythonスクリプトとして実行
            result = subprocess.run(["uv", "run", "python", script_path])
            
            # エラーが起きた場合は即座にパイプラインを停止
            if result.returncode != 0:
                print(f"\n❌ エラー: {script_path} の実行中にエラーが発生しました。パイプラインを停止します。(Exit Code: {result.returncode})")
                sys.exit(result.returncode)
                
            print(f"✅ 完了: {script_path}")

    print("-" * 50)
    print("🎉 全てのパイプラインステップが正常に完了しました！")

if __name__ == "__main__":
    main()
