import concurrent.futures
import subprocess
import queue
import time

def run_command(cmd, task_id):
    """指定されたコマンドを実行する"""
    print(f"タスク {task_id} 開始...")
    subprocess.run(cmd, shell=True)
    print(f"タスク {task_id} 完了")

# 実行するコマンド
cmd = "python3 main.py random_velocity_spread_no_ability rel_ability2"

# 同時に実行する最大タスク数
max_workers = 5

# 合計で実行するタスク数
total_runs = 30

# スレッドプールエグゼキュータを使用してタスクを管理
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    task_id = 1

    # 初期のタスクを追加
    for _ in range(max_workers):
        future = executor.submit(run_command, cmd, task_id)
        futures[future] = task_id
        task_id += 1

    completed_runs = 0

    while completed_runs < total_runs:
        # 完了したタスクを確認
        done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

        for future in done:
            completed_runs += 1
            completed_task_id = futures[future]

            if completed_runs < total_runs:
                # 新しいタスクを追加
                new_future = executor.submit(run_command, cmd, task_id)
                futures[new_future] = task_id
                task_id += 1

            del futures[future]
            print(f"完了タスク数: {completed_runs}/{total_runs}")

# 全てのタスクが完了
print("全てのタスクが完了しました。")
