# import os
# import math
# import finalstylefeedback as app  # 引入你的主程式

# # --- 1. 偽造 (Mock) 外部依賴 ---

# # 偽造 extract_features_for_rf，讓它直接回傳我們指定的數據
# def mock_extract_features(file_path):
#     # 這裡我們約定 file_path 傳進來就是一個字典，方便測試
#     return file_path 

# # 偽造播放函式，只印出播放清單，不真的播聲音
# def mock_play_sequence(playlist):
#     print("\n[測試結果]  預計播放清單：")
#     for p in playlist:
#         print(f"  -> {p}")
#     print("---")

# # 替換掉原本的函式
# app.extract_features_for_rf = mock_extract_features
# app.play_voice_sequence = mock_play_sequence
# # 為了不讓執行緒干擾測試，我們也可以把 threading.Thread 換成直接執行
# import threading
# # --- 修正後的 Mock Thread ---
# import threading

# # 定義一個假的 Thread 類別
# class MockThread:
#     def __init__(self, target, args=(), daemon=False):
#         self.target = target
#         self.args = args
#         self.daemon = daemon

#     def start(self):
#         # 直接執行目標函式，不開啟新執行緒
#         self.target(*self.args)

# # 將系統的 Thread 替換成我們的假 Thread
# threading.Thread = MockThread

# # --- 2. 設定測試環境 (Setup) ---

# def setup_environment():
#     # 設定風格與目標
#     app.CURRENT_USER_STYLE = "maxPunchPower"
    
#     # 偽造上一筆數據 (基準)
#     app.PREVIOUS_DATA = {
#         "maxPunchPower": 100,
#         "totalPunchNum": 50, # 上一次每5秒打50拳
#         "minReactionTime": 0.5,
#         "score": 100
#     }
    
#     # 設定目標總拳數 (假設目標是 100 拳)
#     app.previousboxstyle_totalpunchnum = 200 
    
#     # 重置累積變數
#     app.ROUND_PUNCH_ACCUMULATOR = 0
    
#     # 偽造權重地圖：讓 totalPunchNum 成為最重要的特徵，且一定會被選中
#     # 這樣只要 totalPunchNum 退步，就一定會觸發語音
#     app.STYLE_WEIGHT_MAP = {
#         "maxPunchPower": {
#             "training": [("totalPunchNum", 0.9), ("maxPunchPower", 0.1)], # 權重很高
#             "scoring":  [("totalPunchNum", 0.9), ("maxPunchPower", 0.1)]
#         }
#     }
    
#     # 確保資料夾檢查會通過 (騙過 os.path.exists)
#     # 這裡我們用一個簡單的 lambda 讓所有路徑檢查都回傳 True
#     app.os.path.exists = lambda path: True 

# # --- 3. 撰寫測試案例 (Test Cases) ---

# def run_tests():
#     print("=== 開始測試 totalPunchNum 邏輯 ===\n")
#     setup_environment()

#     # --- 案例 A: 還差很多 (> 70) ---
#     print("案例 A: 剛開始打，累積很少 (還差 > 70)")
#     # 當前數據：只打了 10 拳 (比上一筆 50 少，所以是退步，會觸發 training)
#     # 累積：0 + 10 = 10。 剩餘：200 - 10 = 190 (> 70)
#     # 預期：播放 totalPunchNum.wav + 70over_.wav
#     test_data_A = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
#     app.process_feedback_with_style_logic(test_data_A)


#     # --- 案例 B: 快到了 (剩餘 45) ---
#     print("\n案例 B: 打了一陣子了 (剩餘 45)")
#     # 我們手動把累積加到 145
#     setup_environment()
#     app.ROUND_PUNCH_ACCUMULATOR = 145 
#     # 當前數據：打了 10 拳 (依然退步)。
#     # 累積：145 + 10 = 155。 剩餘：200 - 155 = 45。
#     # 邏輯：ceil(45/10)*10 = 50
#     # 預期：播放 totalPunchNum.wav + 50.wav
#     test_data_B = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
#     app.process_feedback_with_style_logic(test_data_B)


#     # --- 案例 C: 已經達標 (< 0) ---
#     print("\n案例 C: 已經達標 (剩餘 < 0)")
#     # 手動把累積加到 200
#     setup_environment()
#     app.ROUND_PUNCH_ACCUMULATOR = 200
#     # 當前數據：打了 10 拳 (退步)。
#     # 累積：200 + 10 = 210。 剩餘：200 - 210 = -10。
#     # 預期：播放 already_exceed.wav (playlist 會被清空只剩這個)
#     test_data_C = {"totalPunchNum": 10, "maxPunchPower": 100, "minReactionTime": 0.5}
#     app.process_feedback_with_style_logic(test_data_C)

# if __name__ == "__main__":
#     run_tests()
import os
import sys
import importlib

# 嘗試匯入主程式
try:
    import finalstylefeedback as app
except ImportError:
    print(" 找不到 finalstylefeedback.py")
    sys.exit(1)

# 1. 偽造 (Mock) 外部依賴
# 偽造 Pygame
class MockMixer:
    def init(self): pass
    def get_init(self): return True
    def stop(self): pass
    def load(self, path): pass
    def play(self): pass
    def get_busy(self): return False 

app.pygame.mixer = MockMixer()
app.pygame.mixer.music = MockMixer()

# 偽造 Thread (同步執行)
class MockThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args
    def start(self):
        self.target(*self.args)
app.threading.Thread = MockThread

# 偽造 extract_features_for_rf (直接回傳測試數據)
app.extract_features_for_rf = lambda x: x

# 偽造 os.path.exists (永遠回傳 True，但我們會印出路徑來檢查)
app.os.path.exists = lambda path: True

# 偽造播放器 (核心驗證邏輯)
def mock_play_sequence(playlist):
    print(f"       [播放序列]: {playlist}")
    # 這裡可以加入斷言 (Assert) 來自動檢查路徑格式
app.play_voice_sequence = mock_play_sequence

def mock_play_quick(path):
    print(f"       [單檔播放]: {path}")
app.play_quick_voice = mock_play_quick

# ==========================================
# 2. 測試輔助函式
# ==========================================

def reset_state():
    """重置所有全域變數"""
    app.ROUND_PUNCH_ACCUMULATOR = 0
    app.PREVIOUS_DATA = None
    # 預設基準數據 (表現中等)
    app.PREVIOUS_DATA = {
        k: 50.0 for k in app.FEATURE_AUDIO_MAP.keys()
    }
    app.PREVIOUS_DATA['score'] = 100
    app.PREVIOUS_DATA['totalPunchNum'] = 50 # 基準拳數
    app.previousboxstyle_totalpunchnum = 200 # 目標拳數

def force_feature_regression(target_feature):
    """
    強制讓某個特徵「退步」，並將其權重設為最高，
    確保程式一定會選中這個特徵來播放。
    """
    # 1. 製造退步數據
    current_data = app.PREVIOUS_DATA.copy()
    
    if target_feature in app.LOWER_IS_BETTER_FEATURES:
        # 越低越好 -> 變大就是退步
        current_data[target_feature] = current_data[target_feature] + 20 
    else:
        # 越高越好 -> 變小就是退步
        current_data[target_feature] = current_data[target_feature] - 20
        
    # 2. 操控權重地圖 (Hacking the Matrix)
    # 我們把所有風格的 Training 和 Scoring 權重都改成只包含這個 target_feature
    # 這樣程式就別無選擇，只能抓這個缺點
    fake_weights = {
        "training": [(target_feature, 0.99)],
        "scoring":  [(target_feature, 0.99)]
    }
    
    app.STYLE_WEIGHT_MAP = {
        "maxPunchPower": fake_weights,
        "maxPunchSpeed": fake_weights,
        "minReactionTime": fake_weights
    }
    
    return current_data

# ==========================================
# 3. 執行全面覆蓋測試
# ==========================================

def run_full_coverage_test():

    print(" 開始執行 White-Box 全覆蓋測試")


    # 定義要測試的維度
    languages = [
        ('C', app.QUICK_VOICE_FOLDER),
        ('E', app.ENG_QUICK_VOICE_FOLDER),
        ('J', app.JAP_QUICK_VOICE_FOLDER)
    ]
    
    styles = ["maxPunchPower", "maxPunchSpeed", "minReactionTime"]
    
    # 取得所有特徵列表 (包含你新加的 avg 系列)
    all_features = list(app.FEATURE_AUDIO_MAP.keys())

    # --- 測試迴圈 ---
    
    for lang_code, lang_path in languages:
        print(f"\n [測試語言]: {lang_code} (路徑: {lang_path})")
        app.CURRENT_VOICE_FOLDER = lang_path # 切換語言
        
        for style in styles:
            print(f"   [測試風格]: {style}")
            app.CURRENT_USER_STYLE = style
            
            # ------------------------------------------------
            # 測試場景 1: 針對每一個特徵，測試「Style Mode (練功)」的語音
            # ------------------------------------------------
            print(f"    --- 測試所有特徵 (Style Mode - 退步) ---")
            for feature in all_features:
                if feature == "totalPunchNum": continue # 這個後面單獨測
                
                reset_state()
                # 強制主風格沒進步 (進入 Style Mode)
                current_data = force_feature_regression(feature)
                # 確保主風格變差
                if style not in app.LOWER_IS_BETTER_FEATURES:
                    current_data[style] = app.PREVIOUS_DATA[style] - 10 
                
                print(f"      測項: {feature} 下降", end="")
                app.process_feedback_with_style_logic(current_data)

            # ------------------------------------------------
            # 測試場景 2: 針對每一個特徵，測試「Score Mode (得分)」的語音
            # ------------------------------------------------
            print(f"    --- 測試所有特徵 (Score Mode - 進步但有缺點) ---")
            for feature in all_features:
                if feature == "totalPunchNum": continue
                
                reset_state()
                current_data = force_feature_regression(feature)
                # 強制主風格進步 (進入 Score Mode)
                if style not in app.LOWER_IS_BETTER_FEATURES:
                    current_data[style] = app.PREVIOUS_DATA[style] + 10
                else:
                    current_data[style] = app.PREVIOUS_DATA[style] - 10
                
                print(f"      測項: {feature} 下降", end="")
                app.process_feedback_with_style_logic(current_data)

            # ------------------------------------------------
            # 測試場景 3: 完美表現 (Best.wav)
            # ------------------------------------------------
            print(f"    --- 測試完美表現 (Best Case) ---")
            reset_state()
            current_data = app.PREVIOUS_DATA.copy()
            # 讓所有數據都變好
            if style in app.LOWER_IS_BETTER_FEATURES:
                current_data[style] = current_data[style] - 10 # 數值變小才是進步 (反應時間)
            else:
                current_data[style] = current_data[style] + 10 # 數值變大才是進步 (力量/速度)
            # 讓權重最高的特徵也進步 (避免被抓缺點)
            # 這裡我們隨便設一個權重地圖，只要不觸發 is_worse 即可
            app.STYLE_WEIGHT_MAP = {style: {"scoring": [("hitRate", 0.5)]}} 
            current_data["hitRate"] = 100 # 完美
            
            print(f"      測項: 完美表現", end="")
            app.process_feedback_with_style_logic(current_data)

    # ------------------------------------------------
    # 測試場景 4: TotalPunchNum 的特殊數字邏輯 (跨語言)
    # ------------------------------------------------
    print("\n [測試 TotalPunchNum 特殊數字邏輯]")
    
    # 定義數字測試案例
    punch_cases = [
        (10,  "10.wav",       "剩餘 190 (進位200?) 邏輯需確認"), # 依你的邏輯 >70 播 70over
        (130, "70over_.wav",  "剩餘 70 (200-130)"),
        (155, "50.wav",       "剩餘 45 -> 50"),
        (210, "already_exceed.wav", "剩餘 -10")
    ]
    
    # 修正你的邏輯測試：
    # 你的邏輯是 remaining > 70 播 70over
    # 0 < remaining <= 70 播數字
    
    punch_scenarios = [
        (10,  "70over_.wav"),        # 累積10, 剩190 (>70)
        (130, "70.wav"),             # 累積130, 剩70 (<=70) -> 70.wav
        (155, "50.wav"),             # 累積155, 剩45 -> 50.wav
        (210, "already_exceed.wav")  # 累積210, 剩-10
    ]

    for lang_code, lang_path in languages:
        print(f"\n  語言: {lang_code}")
        app.CURRENT_VOICE_FOLDER = lang_path
        app.CURRENT_USER_STYLE = "maxPunchPower" # 假設風格
        
        for accum, expected_wav in punch_scenarios:
            reset_state()
            # 強制設定 totalPunchNum 為最高權重
            app.STYLE_WEIGHT_MAP = {"maxPunchPower": {"training": [("totalPunchNum", 1.0)]}}
            
            # 設定累積與目標
            app.ROUND_PUNCH_ACCUMULATOR = accum
            app.previousboxstyle_totalpunchnum = 200
            
            # 設定當次出拳數很少 (造成退步, 觸發語音)
            current_data = app.PREVIOUS_DATA.copy()
            current_data["totalPunchNum"] = 0 # 這次沒打，肯定退步
            
            # 為了不讓 ROUND_PUNCH_ACCUMULATOR 在函式內又被加一次 current_data，我們先扣掉
            # 因為函式內有 ROUND_PUNCH_ACCUMULATOR += current_punches
            app.ROUND_PUNCH_ACCUMULATOR -= current_data["totalPunchNum"] 
            
            print(f"    累積 {accum}, 預期: {expected_wav}", end="")
            app.process_feedback_with_style_logic(current_data)

if __name__ == "__main__":
    run_full_coverage_test()