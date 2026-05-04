

import os
import json
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 檢查條件
        if "playerLHandPosLogs" not in data or data["summary"].get("score") == 0 or "punchPower" not in data or not data["punchPower"] or data["summary"].get("maxPunchPower")==0 or "puncherIdx" not in data or not data["puncherIdx"] or "punchTimeCode" not in data or not["punchTimeCode"] or "punchPower" not in data or not ["punchPower"] or "punchSpeed" not in data or not["punchSpeed"] or "reactionTime" not in data or not["reactionTime"] or "playerLHandPosLogs" not in data or not ["playerLHandPosLogs"] or "playerRHandPosLogs" not in data or not["playerRHandPosLogs"] or "playerPosLogs" not in data or not ["playerPosLogs"]:
            os.remove(file_path)
            print(f"文件已删除: {os.path.basename(file_path)}")
        else:
            print(f"文件保留: {os.path.basename(file_path)}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"處理文件時出錯: {os.path.basename(file_path)}, 錯誤: {e}")

def check_and_delete_files(folder_path, max_workers=10):
    # 獲取所有 JSON 檔案的路徑
    file_paths = [
        os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".json")
    ]

    # 使用 ThreadPoolExecutor 加速處理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, file_paths)

# 使用範例
os.chdir(os.path.dirname(os.path.abspath(__file__)))
folder_path = "./processed_users1"
check_and_delete_files(folder_path)
