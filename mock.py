import os
import math
import finalstylefeedback as app  # 引入你的主程式

# --- 1. 偽造 (Mock) 外部依賴 ---

# 偽造 extract_features_for_rf，讓它直接回傳我們指定的數據
def mock_extract_features(file_path):
    # 這裡我們約定 file_path 傳進來就是一個字典，方便測試
    return file_path 

# 偽造播放函式，只印出播放清單，不真的播聲音
def mock_play_sequence(playlist):
    print("\n[測試結果]  預計播放清單：")
    for p in playlist:
        print(f"  -> {p}")
    print("---")

# 替換掉原本的函式
app.extract_features_for_rf = mock_extract_features
app.play_voice_sequence = mock_play_sequence
# 為了不讓執行緒干擾測試，我們也可以把 threading.Thread 換成直接執行
import threading
# --- 修正後的 Mock Thread ---
import threading

# 定義一個假的 Thread 類別
class MockThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        # 直接執行目標函式，不開啟新執行緒
        self.target(*self.args)

# 將系統的 Thread 替換成我們的假 Thread
threading.Thread = MockThread

# --- 2. 設定測試環境 (Setup) ---

def setup_environment():
    # 設定風格與目標
    app.CURRENT_USER_STYLE = "maxPunchPower"
    
    # 偽造上一筆數據 (基準)
    app.PREVIOUS_DATA = {
        "maxPunchPower": 100,
        "totalPunchNum": 50, # 上一次每5秒打50拳
        "minReactionTime": 0.5,
        "score": 100
    }
    
    # 設定目標總拳數 (假設目標是 100 拳)
    app.previousboxstyle_totalpunchnum = 200 
    
    # 重置累積變數
    app.ROUND_PUNCH_ACCUMULATOR = 0
    
    # 偽造權重地圖：讓 totalPunchNum 成為最重要的特徵，且一定會被選中
    # 這樣只要 totalPunchNum 退步，就一定會觸發語音
    app.STYLE_WEIGHT_MAP = {
        "maxPunchPower": {
            "training": [("totalPunchNum", 0.9), ("maxPunchPower", 0.1)], # 權重很高
            "scoring":  [("totalPunchNum", 0.9), ("maxPunchPower", 0.1)]
        }
    }
    
    # 確保資料夾檢查會通過 (騙過 os.path.exists)
    # 這裡我們用一個簡單的 lambda 讓所有路徑檢查都回傳 True
    app.os.path.exists = lambda path: True 

# --- 3. 撰寫測試案例 (Test Cases) ---

def run_tests():
    print("=== 開始測試 totalPunchNum 邏輯 ===\n")
    setup_environment()

    # --- 案例 A: 還差很多 (> 70) ---
    print("案例 A: 剛開始打，累積很少 (還差 > 70)")
    # 當前數據：只打了 10 拳 (比上一筆 50 少，所以是退步，會觸發 training)
    # 累積：0 + 10 = 10。 剩餘：200 - 10 = 190 (> 70)
    # 預期：播放 totalPunchNum.wav + 70over_.wav
    test_data_A = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
    app.process_feedback_with_style_logic(test_data_A)


    # --- 案例 B: 快到了 (剩餘 45) ---
    print("\n案例 B: 打了一陣子了 (剩餘 45)")
    # 我們手動把累積加到 145
    setup_environment()
    app.ROUND_PUNCH_ACCUMULATOR = 145 
    # 當前數據：打了 10 拳 (依然退步)。
    # 累積：145 + 10 = 155。 剩餘：200 - 155 = 45。
    # 邏輯：ceil(45/10)*10 = 50
    # 預期：播放 totalPunchNum.wav + 50.wav
    test_data_B = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
    app.process_feedback_with_style_logic(test_data_B)


    # --- 案例 C: 已經達標 (< 0) ---
    print("\n案例 C: 已經達標 (剩餘 < 0)")
    # 手動把累積加到 200
    setup_environment()
    app.ROUND_PUNCH_ACCUMULATOR = 200
    # 當前數據：打了 10 拳 (退步)。
    # 累積：200 + 10 = 210。 剩餘：200 - 210 = -10。
    # 預期：播放 already_exceed.wav (playlist 會被清空只剩這個)
    test_data_C = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
    app.process_feedback_with_style_logic(test_data_C)

if __name__ == "__main__":
    run_tests()