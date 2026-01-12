import os
import json
import numpy as np
import pandas as pd
import time
import math
import pygame
import threading
import traceback
import openai
import google.generativeai as genai
from google import genai as google_genai
from google.genai import types
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from scipy import stats

#設定 
FOLDER_PATH = "./processed_users1/"  # 歷史資料資料夾 (用於訓練 RF)
MONITOR_FOLDER = "./10sec/"          # 監控資料夾 (即時數據)
QUICK_VOICE_FOLDER = "./quick_voice/" # 語音檔案資料夾
ENG_QUICK_VOICE_FOLDER = './english_quick_voice/'
JAP_QUICK_VOICE_FOLDER = './japanese_quick_voice/'
CURRENT_VOICE_FOLDER = QUICK_VOICE_FOLDER
load_dotenv("./.env",override=True)
# openai.api_key= os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     print("未輸入openAPIkey")
# else:
#     print("已取得apikey")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
boxing_df = pd.DataFrame()
STYLE_WEIGHT_MAP = {} 
PREVIOUS_DATA = None
CURRENT_USER_STYLE = None # 預設風格
FILE_PROCESS_COUNT = 0  # 計算處理過的檔案數量
MAX_FILES_BEFORE_RESET = 6  #看你幾秒一組就 30/x= MAX_FILES_BEFORE_RESET
stop_monitoring = False
ROUND_PUNCH_ACCUMULATOR = 0

# 特徵對應語音檔名的表,總共九大類
FEATURE_AUDIO_MAP = {
    "totalPunchNum": "totalPunchNum.wav",           
    "total_user_body_movement": "total_user_body_movement.wav", 
    "minReactionTime": "minReactionTime.wav",     
    "total_hand_move_per_punch": "total_hand_move_per_punch.wav",
    "range_z": "range_z.wav",              
    "range_x": "range_x.wav", 
    "range_y": "range_y.wav",                 
    "hitRate": "hitRate.wav",                 
    "maxPunchSpeed": "maxPunchSpeed.wav",         
    "maxPunchPower": "maxPunchPower.wav"          
}

#計算移動參數
def calculate_path_length(logs):
    if not logs or len(logs) < 2:
        return 0.0
    coords = np.array([[p['x'], p['y'], p['z']] for p in logs])
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

def get_ranges(logs):
    if not logs: return {'x_range': 0, 'y_range': 0, 'z_range': 0}
    xs = [p['x'] for p in logs]
    ys = [p['y'] for p in logs]
    zs = [p['z'] for p in logs]
    return {
        'x_range': max(xs) - min(xs),
        'y_range': max(ys) - min(ys),
        'z_range': max(zs) - min(zs)
    }
#計算即時回饋的參數
# def extract_features_for_rf(file_path): #舊版
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
#         print(f"讀取檔案錯誤: {file_path}, 原因: {e}")
#         return None

#     summary = data.get('summary', {})
    
#     min_rt = summary.get('minReactionTime', 0)
#     avg_rt = summary.get('avgReactionTime', 0)
#     if min_rt == 3.5835 or avg_rt == 3.5835:
#         print(f"過濾異常檔案 (ReactionTime 異常): {os.path.basename(file_path)}")
#         return None

#     # 取得其他數據
#     punch_power = data.get("punchPower", [])
#     max_power = max(punch_power) if punch_power else 0
#     max_speed = summary.get('maxPunchSpeed', 0)
#     punch_num = summary.get('totalPunchNum', 1)
    
#     #這邊可以過濾一些數值異常的檔案
#     # if not (punch_num >= 0 and 0 < max_speed <= 10 and max_power <= 1000):
#     #     print(f"過濾異常數據檔案 : {os.path.basename(file_path)}")
#     #     if not punch_num >= 0: print(" totalPunchNum < 0")
#     #     if not 0 < max_speed <= 10: print(f"maxPunchSpeed 異常: {max_speed}")
#     #     if not max_power <= 900: print(f"maxPunchPower 異常: {max_power}")
#     #     return None
    
#     player_logs = data.get('playerPosLogs', [])
#     r_hand_logs = data.get('playerRHandPosLogs', [])
#     l_hand_logs = data.get('playerLHandPosLogs', [])
    
#     if punch_num == 0: punch_num = 1

#     body_dist = calculate_path_length(player_logs)
#     r_dist = calculate_path_length(r_hand_logs)
#     l_dist = calculate_path_length(l_hand_logs)
#     total_move_per = (r_dist + l_dist) / punch_num
    
#     rng = get_ranges(player_logs)

#     features = {
#         'score': summary.get('score', 0),
#         'maxPunchPower': max_power,
#         'maxPunchSpeed': max_speed,
#         'minReactionTime': min_rt,
#         'hitRate': summary.get('hitRate', 0),
#         'totalPunchNum': punch_num,
#         'total_user_body_movement': body_dist,
#         'total_hand_move_per_punch': total_move_per,
#         'range_x': rng['x_range'],
#         'range_y': rng['y_range'],
#         'range_z': rng['z_range']
#     }
#     return features

def extract_features_for_rf(file_path): #處理boxing_df需要的資料row,cloumn
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"讀取檔案錯誤: {file_path}, 原因: {e}")
        return None

    summary = data.get('summary', {})
    
    # 保留原本的過濾，防止異常數據影響權重
    min_rt = summary.get('minReactionTime', 0)
    avg_rt = summary.get('avgReactionTime', 0)
    if min_rt == 3.5835 or avg_rt == 3.5835:
        # print(f"過濾異常檔案 (ReactionTime 異常): {os.path.basename(file_path)}")
        return None
    punch_power = data.get("punchPower", [])
    if punch_power:
        avgPunchPower = np.mean(punch_power)
    else:
        avgPunchPower = 0
    max_power = max(punch_power) if punch_power else 0

    score = summary.get('score', 0)
    punch_num = summary.get('totalPunchNum', 1)
    if punch_num == 0: punch_num = 1
    
    # punch_power = data.get("punchPower", [])  #檢查異常
    # max_power = max(punch_power) if punch_power else 0
    # max_speed = summary.get('maxPunchSpeed', 0)
    # punch_num = summary.get('totalPunchNum', 1)
    # if not (punch_num >= 0 and 0 < max_speed <= 10 and max_power <= 1000):
    #     print(f"過濾異常數據檔案 : {os.path.basename(file_path)}")
    #     if not punch_num >= 0: print(" totalPunchNum < 0")
    #     if not 0 < max_speed <= 10: print(f"maxPunchSpeed 異常: {max_speed}")
    #     if not max_power <= 900: print(f"maxPunchPower 異常: {max_power}")
    #     return None
    
    player_logs = data.get('playerPosLogs', [])
    r_hand_logs = data.get('playerRHandPosLogs', [])
    l_hand_logs = data.get('playerLHandPosLogs', [])
    
    body_dist = calculate_path_length(player_logs)
    r_dist = calculate_path_length(r_hand_logs)
    l_dist = calculate_path_length(l_hand_logs)
    total_move_per = (r_dist + l_dist) / punch_num
    
    # 計算範圍 
    rng = get_ranges(player_logs)
    
    features = {
        'score': score,
        # 直接使用 Summary 的數值，而非自己用 list 算 max，確保與 test3 一致
        'maxPunchPower': max_power,
        'maxPunchSpeed': summary.get('maxPunchSpeed', 0),
        'minReactionTime': summary.get('minReactionTime', 0),
        'hitRate': summary.get('hitRate', 0),
        'avgPunchSpeed' : summary.get('avgPunchSpeed',0),
        'avgReactionTime':summary.get('avgReactionTime',0),
        'avgPunchPower': avgPunchPower,
        # 新特徵 (Formative)
        'total_user_body_movement': body_dist,
        'total_hand_move_per_punch': total_move_per, 
        'totalPunchNum': punch_num,
        'range_x': rng['x_range'],
        'range_y': rng['y_range'],
        'range_z': rng['z_range']
    }
    
    return features

def load_all_json_files():# 將一開始的資料進行權重分析, 並建立boxing_df
    global boxing_df
    data_list = []
    files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
    print(f"正在載入 {len(files)} 筆歷史資料...")
    
    for filename in files:
        file_path = os.path.join(FOLDER_PATH, filename)
        if os.path.getsize(file_path) > 0:
            try:
                feats = extract_features_for_rf(file_path)
                if feats is not None:
                    data_list.append(feats)
            except Exception as e:
                print(f"處理檔案發生未預期錯誤 {filename}: {e}")
                continue
                
    boxing_df = pd.DataFrame(data_list)
    print(f"boxinfdf={boxing_df}")
    print("歷史資料載入完成")


# def init_style_weights(all_data): #訓練RF 模型，並將每個風格特質拳種 score拳種紀錄起來, 舊版
#     global STYLE_WEIGHT_MAP
#     if not all_data: return

#     df = pd.DataFrame(all_data)
#     df = df.dropna()

#     formative_cols = [
#         'total_user_body_movement', 'total_hand_move_per_punch', 
#         'range_x', 'range_y', 'range_z'
#     ]
#     base_summary_cols = ['maxPunchPower', 'maxPunchSpeed', 'minReactionTime', 'hitRate']
#     targets = ["maxPunchPower", "maxPunchSpeed", "minReactionTime"]

#     print("正在初始化風格權重系統...")

#     for target in targets:
#         STYLE_WEIGHT_MAP[target] = {"training": [], "scoring": []}

#         # style權重
#         other_summary_cols = [col for col in base_summary_cols if col != target]
#         feature_cols_A = formative_cols + other_summary_cols
#         rf_train = RandomForestRegressor(n_estimators=100, random_state=42)
#         rf_train.fit(df[feature_cols_A], df[target])
#         imp_train = pd.Series(rf_train.feature_importances_, index=feature_cols_A).sort_values(ascending=False)
#         STYLE_WEIGHT_MAP[target]["training"] = list(imp_train.items())

#         # score權重
#         features_for_score = formative_cols + [target, 'hitRate', 'totalPunchNum']
#         rf_score = RandomForestRegressor(n_estimators=100, random_state=42)
#         rf_score.fit(df[features_for_score], df['score'])
#         imp_score = pd.Series(rf_score.feature_importances_, index=features_for_score).sort_values(ascending=False)
#         STYLE_WEIGHT_MAP[target]["scoring"] = list(imp_score.items())
        
#         print(f"{target}style處理完成{STYLE_WEIGHT_MAP[target]["training"]}")
#         print(f"{target}style的score權重處理完成{STYLE_WEIGHT_MAP[target]["scoring"]}")
def init_style_weights(all_data): #權重處裡
    global STYLE_WEIGHT_MAP
    if not all_data: return

    df = pd.DataFrame(all_data)
    df = df.dropna()

    formative_cols = [
        'total_user_body_movement', 'total_hand_move_per_punch', 
        'range_x', 'range_y', 'range_z'
    ]
    base_summary_cols = ['maxPunchPower', 'maxPunchSpeed', 'minReactionTime', 'hitRate']
    targets = ["maxPunchPower", "maxPunchSpeed", "minReactionTime"]

    print("正在初始化風格權重系統...")

    for target in targets:
        STYLE_WEIGHT_MAP[target] = {"training": [], "scoring": []}

        # style
        other_summary_cols = [col for col in base_summary_cols if col != target]
        if target == "maxPunchSpeed" and "maxPunchPower" in other_summary_cols:
            other_summary_cols.remove("maxPunchPower")

        feature_cols_A = formative_cols + other_summary_cols
        rf_train = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_train.fit(df[feature_cols_A], df[target])
        imp_train = pd.Series(rf_train.feature_importances_, index=feature_cols_A).sort_values(ascending=False)
        STYLE_WEIGHT_MAP[target]["training"] = list(imp_train.items())

        # Scoring
        features_for_score = formative_cols + [target, 'hitRate', 'totalPunchNum']
        rf_score = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_score.fit(df[features_for_score], df['score'])
        imp_score = pd.Series(rf_score.feature_importances_, index=features_for_score).sort_values(ascending=False)
        STYLE_WEIGHT_MAP[target]["scoring"] = list(imp_score.items())
        
        print(f"\n當前分析風格: {target}") #找出權重分析
        print(f"想提升 {target}，應該專注在:")
        for name, val in STYLE_WEIGHT_MAP[target]["training"]: 
            print(f"{name:<30} (權重: {val:.4f})")
            
        print(f"在 {target} 的風格中，影響 Score 的是:")
        for name, val in STYLE_WEIGHT_MAP[target]["scoring"]: 
            print(f"{name:<30} (權重: {val:.4f})")
previousboxstyle_totalpunchnum=0
# 送資料及正規劃給gpt
def calculate_detailed_stats_for_gpt(file_path):
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 取得基礎 list
    playerRHandPosLogs = data.get("playerRHandPosLogs", [])
    playerLHandPosLogs = data.get("playerLHandPosLogs", [])
    puncherIdx = data.get("puncherIdx", [])
    punchTimeCode = data.get("punchTimeCode", [])
    summary = data.get('summary', {})
    previousboxstyle_totalpunchnum = summary.get("totalPunchNum",0)
    reactionTime = data.get("reactionTime", [])
    punchSpeed = data.get("punchSpeed", [])
    punchPower = data.get("punchPower", [])
    
    # 判斷總時間 (30s / 60s)
    if not punchTimeCode: return None
    totaltime = math.ceil(punchTimeCode[-1])
    if totaltime not in [30, 60]:
        totaltime = 60 if abs(60-totaltime) < abs(30-totaltime) else 30
    
    sec_num = len(playerRHandPosLogs) // totaltime if totaltime > 0 else 30
    
    # 2. 計算平均值 )
    r_first, l_first = 0, 0
    l_xavg, l_yavg, l_zavg = [], [], []
    r_xavg, r_yavg, r_zavg = [], [], []
    
    # 這裡計算每拳的平均位置 
    for i in range(len(puncherIdx)):
        end_index = round(punchTimeCode[i] * sec_num)
        if end_index > len(playerRHandPosLogs): end_index = len(playerRHandPosLogs)
        
        # 左拳算左手 log，右拳算右手 log
        if puncherIdx[i] == 0: # Right Hand Punch
            temp_r = playerRHandPosLogs[r_first:end_index]
            if temp_r:
                r_xavg.append(sum(p["x"] for p in temp_r)/len(temp_r))
                r_yavg.append(sum(p["y"] for p in temp_r)/len(temp_r))
                r_zavg.append(sum(p["z"] for p in temp_r)/len(temp_r))
            else:
                r_xavg.append(0); r_yavg.append(0); r_zavg.append(0)
            r_first = end_index
            
            # 對齊長度補上0
            if l_xavg: l_xavg.append(l_xavg[-1])
            else: l_xavg.append(0)
            if l_yavg: l_yavg.append(l_yavg[-1])
            else: l_yavg.append(0)
            if l_zavg: l_zavg.append(l_zavg[-1])
            else: l_zavg.append(0)
            
        else: # Left Hand Punch
            temp_l = playerLHandPosLogs[l_first:end_index]
            if temp_l:
                l_xavg.append(sum(p["x"] for p in temp_l)/len(temp_l))
                l_yavg.append(sum(p["y"] for p in temp_l)/len(temp_l))
                l_zavg.append(sum(p["z"] for p in temp_l)/len(temp_l))
            else:
                l_xavg.append(0); l_yavg.append(0); l_zavg.append(0)
            l_first = end_index
            
            # 對齊長度
            if r_xavg: r_xavg.append(r_xavg[-1])
            else: r_xavg.append(0)
            if r_yavg: r_yavg.append(r_yavg[-1])
            else: r_yavg.append(0)
            if r_zavg: r_zavg.append(r_zavg[-1])
            else: r_zavg.append(0)

    # 計算 Stability (以 STD 代替 user 的 offset 計算，效果類似)
    # 若無數據則給 0
    std_r = np.std(r_xavg) + np.std(r_yavg) + np.std(r_zavg) if r_xavg else 0
    std_l = np.std(l_xavg) + np.std(l_yavg) + np.std(l_zavg) if l_xavg else 0
    
    # 建立要回傳的 raw data dict
    raw_data = {
        "reactionTime": reactionTime,
        "punchSpeed": punchSpeed,
        "punchPower": punchPower,
        "l_xavg": l_xavg, "l_yavg": l_yavg, "l_zavg": l_zavg, #左手平均值
        "r_xavg": r_xavg, "r_yavg": r_yavg, "r_zavg": r_zavg, #右手平均值
        # "total_r_stability": [std_r], # 模擬 user 的 list 結構
        # "total_l_stability": [std_l]
    }
    return raw_data

#準備 Prompt 需要的 Normalized Features 與 Percentage Series
def prepare_data_for_gpt(file_path): #第二種把資料分開的
    global boxing_df
    
    # 提取基礎特徵（Summary 用）
    rf_feats = extract_features_for_rf(file_path)
    if rf_feats is None: return None
    
    max_values = boxing_df.max(numeric_only=True)
    
    # 分類Summary Data百分比排名
    percentage_series = {}
    json_columns = ["totalPunchNum", "maxPunchSpeed", "hitRate", "minReactionTime", "maxPunchPower","avgPunchPower","avgPunchSpeed","avgReactionTime"]
    # for col in json_columns: #以前的計算
    #     val = rf_feats.get(col, 0)
    #     max_v = max_values.get(col, 1)
    #     if col == "minReactionTime":
    #          pct = max(0, (1 - val/max_v) * 100)
    #     else:
    #          pct = (val / max_v) * 100
    for col in json_columns:
        # 1. 取得當前使用者的數值
        val = rf_feats.get(col, 0)
        
        # 2. 取得歷史資料庫中該欄位的所有數據 (去除空值)
        # 這是關鍵：你的 val 要跟這群 ref_data 比排名
        if col in boxing_df.columns:
            ref_data = boxing_df[col].dropna()
        else:
            # 防呆：如果資料庫沒這個欄位，給個預設值 50 分
            percentage_series[col] = 50.0
            continue

        # 3. 計算 PR 值 (0 到 100)
        # percentileofscore 會告訴你，val 勝過了 ref_data 中多少百分比的人
        pr_score = stats.percentileofscore(ref_data, val, kind='weak')
        
        # 4. 特殊處理：反應時間 (越低越好)
        # 如果反應時間勝過 90% 的人(數值很大)，代表他很慢，這是不對的。
        # 所以對於反應時間，我們要用 100 去扣，或者直接取倒數排名。
        if col == "minReactionTime" or col == "avgReactionTime":
            # 邏輯反轉：數值越小(越快)，PR 應該越高
            # 這裡我們簡單用 100 - 原本的 PR (假設數值大代表慢)
            # 但更精確的做法是：數值越小，排名越前面
            pr_score = 100 - pr_score
        # percentage_series[col] = round(pct, 2)
        percentage_series[col] = round(pr_score, 2)

    #  分類Formative Data正規化細節 
    raw_data = calculate_detailed_stats_for_gpt(file_path)
    # scaler = MinMaxScaler()
    # formative_normalized = {}
    formative_analysis = {}
    
    keys_to_norm = {
        "reactionTime": "normalize_reactionTime",
        "punchSpeed": "normalize_punchspeed",
        "punchPower": "normalize_punchPower",
        # "total_r_stability": "normalize_rhand_stability",
        # "total_l_stability": "normalize_lhand_stability"
    }
    key_mapping = {
            "reactionTime": "avgReactionTime", # 對應到 avg
            "punchSpeed": "avgPunchSpeed",     # 對應到 avg
            "punchPower": "avgPunchPower",     # 對應到 avg
        }
    for key, new_key in keys_to_norm.items():
        user_arr = np.array(raw_data.get(key, []))
        if len(user_arr) > 1:# 1. 計算使用者自己的平均 (Intent) 與 穩定度 (Consistency)
            user_mean = np.mean(user_arr)
            user_std = np.std(user_arr)
            cv = user_std / user_mean if user_mean != 0 else 0 # 自身穩定度 (1 - CV) Coefficient of Variation
            consistency = max(0, min(1, 1 - cv)) # 限制在 0~1
            target_col = key_mapping.get(key)
            if target_col and target_col in boxing_df.columns:# 2. 取得全域(歷史資料)的統計數據來比較
                global_col = boxing_df[target_col].dropna()
                global_mean = global_col.mean()
                global_std = global_col.std()
            else:
                global_mean = user_mean # 防呆: 如果沒歷史資料，Z-score 就會是 0
                global_std = 1
            
            # 3. 計算 Z-Score (這就是他在全體中的強度定位)
            if global_std == 0: global_std = 1 # 避免除以零
            z_score = (user_mean - global_mean) / global_std
            
            # 特殊處理: 反應時間越低(越快)越好，所以 Z-score 要反轉
            if key == "reactionTime":
                z_score = -z_score
            
            # 4. 為了讓 Gemini 好讀，我們可以把 Z-Score 轉成一個 0-100 的分數概念
            # Z=0 -> 50分, Z=+2 -> 70分, Z=-2 -> 30分
            # 這樣 Gemini 就能懂: >50 是強項，<50 是弱項
            intent_score = 50 + (z_score * 10) 
            
            formative_analysis[f"{new_key}_intent_score"] = round(float(intent_score), 2)
            formative_analysis[f"{new_key}_consistency"] = round(float(consistency), 4)
            
        else:
            formative_analysis[f"{new_key}_intent_score"] = 0
            formative_analysis[f"{new_key}_consistency"] = 0
        # arr = np.array(raw_data.get(key, []))
        # if len(arr) > 0: #第一種normalize
        #     norm = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
        #     formative_normalized[new_key] = norm.tolist()
        #     # 額外加入平均值，幫助 GPT 快速抓到特徵
        #     formative_normalized[f"{new_key}_mean"] = round(float(np.mean(norm)), 4)
        
        # if len(arr) > 1: # 至少需要兩筆資料來計算標準#第一種normalize
        #     norm = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
        #     mean_val = np.mean(norm)
        #     std_val = np.std(norm)
        #     # 計算變異係數 CV = 標準差 / 平均值
        #     cv = std_val / mean_val if mean_val > 0 else 1
        #     # 一致性 C = 1 - CV (限制在 0-1 之間)
        #     consistency = max(0, min(1, 1 - cv))
            
        #     formative_analysis[f"{new_key}_intent"] = round(float(mean_val * 100), 2)
        #     formative_analysis[f"{new_key}_consistency"] = round(float(consistency), 4)
        # else:
        #     formative_analysis[f"{new_key}_intent"] = 0
        #     formative_analysis[f"{new_key}_consistency"] = 0
    return {    # 最終傳出的結構
        "summary_data": percentage_series,# 這是 PR 值
        "formative_data": formative_analysis# 這是 Z-Score 轉換分 + 一致性
    }
    
#  呼叫 GPT API 判斷風

def ask_gpt_for_style(structured_data):
    # 強化的 Prompt，明確區分 Summary 與 Formative
    # prompt = f"""
    # You are a data analyst. Your goal is to classify a boxer's style based on two distinct data types:
    
    # 1. Summary Data (Percentage Series): Overall performance compared to the dataset. HIGHER IS BETTER.
    # 2. Formative Data (Normalized Metrics): Detailed movement habits during the session.

    # Input Data:
    # {json.dumps(structured_data, indent=2)}

    # Classification Logic (Strict):
    # Priority 1: Look at 'summary_data'. The style MUST match the metric with the HIGHEST percentage rank.
    # If 'maxPunchPower' has the highest percentile (e.g., 90%), the style IS 'Dominant Knockout Artist'.
    # If 'maxPunchSpeed ' has the highest percentile (e.g., 90%), the style IS 'Agile Rapid Striker'.
    # If 'minReactionTime' has the highest percentile (e.g., 90%), the style IS 'Precision Timing Specialist'.
    
    # Use 'formative_data' ONLY to verify the consistency of movements.

    # Output Options (CHOOSE ONE):
    # Agile Rapid Striker 靈敏速功選手 (Speed)
    # Dominant Knockout Artist 壓迫KO藝術家 (Power)
    # Precision Timing Specialist 精準時機掌控專家 (Reaction)

    # Constraint: Return ONLY the exact style name. No explanation.
    # """
    
    
    #裡面            1. Agile Rapid Striker (Speed):
            # - High resonance in PunchSpeed and TotalPunchNum.
            # 2. Dominant Knockout Artist (Power):
            # - High resonance in PunchPower AND high stability in hand movements.
            # 3. Precision Timing Specialist (Reaction):
            # - High resonance in HitRate and ReactionTime.
    prompt = f"""
        Act as an expert boxing coach and data analyst. Determine the user's 'boxing style' by analyzing the correlation between their results (Summary) and their sport movements (Formative).

        Input Data:
        {json.dumps(structured_data, indent=2)}

        Archetype Definitions & Indicators:

        **Scoring Logic (Weighted Resonance):**
            For each metric (Speed, Power, Reaction), calculate a Resonance Score:
            Score = (Result_Score * 0.5 + Intent_Score * 0.5) * Consistency_Weight

            Where:
            - Result_Score: Percentile from 'summary_results'.
            - Intent_Score: Mean value from 'behavioral_habits'.
            - Consistency_Weight: Reliability multiplier (0.0 to 1.0) from 'behavioral_habits'.

            **Archetype Definitions:**
            1. Agile Rapid Striker (Speed):
            - High resonance in PunchSpeed .
            2. Dominant Knockout Artist (Power):
            - High resonance in PunchPower .
            3. Precision Timing Specialist (Reaction):
            - High resonance in  and ReactionTime.

            **Decision Process:**
            - If a trait has high 'summary_results' but low 'consistency', it is an outlier. DE-PRIORITIZE it.
            - If 'intent' and 'results' both align with high 'consistency', this is a Resonant Style. PRIORITIZE it.

            Output (CHOOSE ONE ONLY):
            Agile Rapid Striker 靈敏速功選手 (Speed)
            Dominant Knockout Artist 壓迫KO藝術家 (Power)
            Precision Timing Specialist 精準時機掌控專家 (Reaction)

            Constraint: Return ONLY the class label. No explanation.
        """
    
    # try:
    #     response = openai.ChatCompletion.create(
    #         # model="gpt-3.5-turbo-0125", 
    #         model="chatgpt-4o-latest",
    #         messages=[
    #             {"role": "system", "content": "You are a precise classification engine."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         max_tokens=100,
    #         temperature=0.2
    #     )
        # if hasattr(response, 'usage'):
        #             used = response.usage
        #             print(f" [GPT 用量] 輸入: {used.prompt_tokens} | 輸出: {used.completion_tokens} | 總計: {used.total_tokens} tokens")
        # result = response.choices[0].message["content"].strip()
    try:
        model = genai.GenerativeModel(
            # model_name="gemini-2.5-flash-lite", 
            model_name="gemini-2.5-flash", 
            system_instruction="You are a precise classification engine."
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3, #
                max_output_tokens=100,
            )
        )
        if response.usage_metadata:
                used = response.usage_metadata
                print(f"[Gemini 用量] 輸入: {used.prompt_token_count} | 輸出: {used.candidates_token_count} | 總計: {used.total_token_count}")
        result = response.text.strip()
        print(f"GEMINI 最終風格判定: {result}")
        return result
    except Exception as e:
        print(f"GEMINI  API 錯誤: {e}")
        return None
#解析 GPT 回傳的文字，對應到系統內部的 Style Key
def determine_style_from_gpt_result(gpt_text):
    global CURRENT_VOICE_FOLDER
    if not gpt_text: return "maxPunchPower" # Default
    
    text_lower = gpt_text.lower()
    
    if "speed" in text_lower or "rapid" in text_lower or "agile" in text_lower:
        play_quick_voice(os.path.join(CURRENT_VOICE_FOLDER, "maxPunchSpeed.wav"))
        return "maxPunchSpeed"
    elif "power" in text_lower or "knockout" in text_lower or "dominant" in text_lower:
        play_quick_voice(os.path.join(CURRENT_VOICE_FOLDER, "maxPunchPower.wav")) 
        return "maxPunchPower"
    elif "reaction" in text_lower or "timing" in text_lower or "precision" in text_lower:
        play_quick_voice(os.path.join(CURRENT_VOICE_FOLDER, "minReactionTime.wav"))
        return "minReactionTime"
    else:
        print("無法從 GPT 回覆中解析出明確風格，使用預設值 Power")
        return "maxPunchPower"


pygame.mixer.init()# 在全域先初始化就好

def play_quick_voice(file_path):
    try:
        if pygame.mixer.music.get_busy(): # 停止舊音樂，載入新音樂即可，不要再 init 了
            pygame.mixer.music.stop()
            
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print(f"播放中：{os.path.basename(file_path)}")
    except Exception as e:
        print(f"播放快速語音時發生錯誤：{e}")
def play_voice_sequence(file_paths): #連續撥放多個音檔
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
            
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            
        for path in file_paths:
            if os.path.exists(path):
                print(f"播放片段：{os.path.basename(path)}")
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                
                # 阻塞等待直到播放完畢 (Block until finished)
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10) 
            else:
                print(f"找不到片段：{path}")
                
    except Exception as e:
        print(f"序列播放錯誤：{e}")
# def play_quick_voice(file_path):# 音效播放
#     try:
#         if pygame.mixer.get_init():
#             pygame.mixer.music.stop()
#         pygame.mixer.init()
#         pygame.mixer.music.load(file_path)
#         pygame.mixer.music.play()
#         print(f"播放中：{os.path.basename(file_path)}")
#     except Exception as e:
#         print(f"播放快速語音時發生錯誤：{e}")

# 核心回饋邏輯 
LOWER_IS_BETTER_FEATURES = ["minReactionTime"]

def process_feedback_with_style_logic(current_file_path):
    global PREVIOUS_DATA, CURRENT_USER_STYLE, ROUND_PUNCH_ACCUMULATOR, CURRENT_VOICE_FOLDER
    
    try:
        current_data = extract_features_for_rf(current_file_path)
    except Exception as e:
        print(f"讀取檔案失敗: {e}")
        return
    
    if PREVIOUS_DATA is None:
        print("發生錯誤,無基準數據")
        PREVIOUS_DATA = current_data
        return

    print(f"當前風格目標: {CURRENT_USER_STYLE}")
    
    # 判斷主風格是否進步
    target_improved = False
    curr_val = current_data[CURRENT_USER_STYLE]
    prev_val = PREVIOUS_DATA[CURRENT_USER_STYLE]

    if CURRENT_USER_STYLE == "minReactionTime":
        # 反應時間：越小越好
        if curr_val < prev_val: target_improved = True 
    elif CURRENT_USER_STYLE == "maxPunchSpeed":
        # 速度：越快越好
        if curr_val > prev_val: target_improved = True
    elif CURRENT_USER_STYLE == "maxPunchPower":
        # 力量：越大越好
        if curr_val > prev_val: target_improved = True
    else:
        if curr_val > prev_val: target_improved = True 

    audio_to_play = None
    reason = ""
    feedback_mode = ""
    number_diff = 0
    current_punches = current_data.get('totalPunchNum', 0)
    ROUND_PUNCH_ACCUMULATOR += current_punches
    
    # 計算還差多少拳才達到目標
    remaining_punches = (previousboxstyle_totalpunchnum- ROUND_PUNCH_ACCUMULATOR)
    if remaining_punches>0:
        print(f"當前風格: {CURRENT_USER_STYLE} | 本次: {current_punches} | 累積: {ROUND_PUNCH_ACCUMULATOR} | 目標剩餘: {remaining_punches}")
    else:
        abs_remaining_punches=abs(remaining_punches)
        print(f"當前風格: {CURRENT_USER_STYLE} | 本次: {current_punches} | 累積: {ROUND_PUNCH_ACCUMULATOR} | 已超出: {abs_remaining_punches}")
    if not target_improved:
        feedback_mode = "Style"
        print(f"風格 {CURRENT_USER_STYLE}沒有提升, 與之前的差異 ({prev_val:.2f} -> {curr_val:.2f})")
        print("尋找合適提升風格的語音")
        weights = STYLE_WEIGHT_MAP[CURRENT_USER_STYLE]["training"]
        for feature, weight in weights:
            is_worse = False
            curr_feat_val = current_data.get(feature, 0)
            prev_feat_val = PREVIOUS_DATA.get(feature, 0)

            # 通用化判斷邏輯：根據特徵性質決定比較方向
            if feature in LOWER_IS_BETTER_FEATURES:
                # 越小越好，如果變大就是變差
                if curr_feat_val > prev_feat_val: is_worse = True
            else:
                # 越大越好，如果變小就是變差 (適用於 Power, Speed, Range, Movement 等)
                if curr_feat_val < prev_feat_val: is_worse = True
            
            if is_worse:
                audio_to_play = FEATURE_AUDIO_MAP.get(feature)
                if audio_to_play:
                    print(f"找到需加強{feature}因素: {prev_feat_val:.3f} -> {curr_feat_val:.3f}下降")
                    reason = f"{feature} 下降 (權重 {weight:.4f})"
                    break # 找到權重最高且變差的特徵，跳出迴圈
    else:
        feedback_mode = "Score"
        print(f"{CURRENT_USER_STYLE} 提升了非常多 ({prev_val:.2f} -> {curr_val:.2f})")
        print("尋找合適提升score的語音")
        thisround_punch=0
        weights = STYLE_WEIGHT_MAP[CURRENT_USER_STYLE]["scoring"]
        for feature, weight in weights:
            is_worse = False
            curr_feat_val = current_data.get(feature, 0)
            prev_feat_val = PREVIOUS_DATA.get(feature, 0)
            if feature == "minReactionTime":
                if current_data[feature] > PREVIOUS_DATA[feature]: is_worse = True
            else:
                if current_data[feature] < PREVIOUS_DATA[feature]: is_worse = True
            
            if is_worse:
                audio_to_play = FEATURE_AUDIO_MAP.get(feature)
                if audio_to_play:
                    reason = f"得分關鍵: {feature} 下降 (權重 {weight:.4f})"
                    if feature =="totalPunchNum":
                        number_diff = remaining_punches
                        print(f"偵測到出拳數減少：至少還需要 {number_diff} 拳")
                        
                    break
    if audio_to_play:
        play_path = os.path.join(CURRENT_VOICE_FOLDER, CURRENT_USER_STYLE,feedback_mode, audio_to_play)
        playlist=[play_path]
        if "totalPunchNum" in audio_to_play:
            num_file = None
            
            if remaining_punches < 0:
                # 情況 A: 已經超過目標 -> 播放 already_exceed.wav
                # 這時候可能不需要播原本的 "你需要更多拳..."，看你需求，這裡我設為「只播 exceed」或「接在後面」
                # 如果你想只播 exceed，就把 playlist 清空
                playlist = [] 
                num_file = "already_exceed.wav"
                print(f"狀況: 目標已達成 (超標 {abs(remaining_punches)})")
                
            elif remaining_punches > 70:
                # 情況 B: 還差很多 -> 播放 70over_.wav
                num_file = "70over_.wav"
                print(f"狀況: 還差很多 (>70)")
                
            else:
                # 情況 C: 0 < 剩餘 <= 70 -> 找最近的 10 的倍數
                # math.ceil(15/10)*10 = 20, math.ceil(11/10)*10 = 20
                # 這樣可以對應 10.wav, 20.wav ... 70.wav
                val = math.ceil(remaining_punches / 10.0) * 10
                num_file = f"{val}.wav"
                print(f"狀況: 剩餘 {remaining_punches} -> 播放 {num_file}")

            # 將數字音檔加入清單
            if num_file:
                # 假設數字檔案放在 quick_voice/numbers/ 資料夾
                num_path = os.path.join(CURRENT_VOICE_FOLDER, "numbers", num_file)
                if os.path.exists(num_path):
                    playlist.append(num_path)
                else:
                    print(f"缺少數字音檔: {num_path}")

        print(f"[播放序列] {playlist} | 原因: {reason}")
        
        # 3. 使用執行緒呼叫連續播放，避免卡住主程式
        threading.Thread(target=play_voice_sequence, args=(playlist,), daemon=True).start()

    else:
        print(f"表現優秀！所有數值都大幅提升！")
        best_file = "best.wav"
        play_path = os.path.join(CURRENT_VOICE_FOLDER, CURRENT_USER_STYLE, feedback_mode, best_file)
        if os.path.exists(play_path):
            print(f"[播放] 完美稱讚語音: {best_file}")
            play_quick_voice(play_path)
        else:
            print(f"找不到完美稱讚語音: {play_path}")
            print("請確認是否已在該風格資料夾中放入 best.wav")
    # if audio_to_play:
    #         play_path = os.path.join(QUICK_VOICE_FOLDER, CURRENT_USER_STYLE,feedback_mode, audio_to_play)
    #         print(f"[播放] 語音: {audio_to_play} | 原因: {reason}")
    #         print(f"路徑: {play_path}") # Debug
            
    #         if os.path.exists(play_path):
    #             play_quick_voice(play_path)
    #         else:
    #             print(f"找不到音檔: {play_path}")
    # else:
    #         print(f"表現優秀！關鍵指標都在進步！")
    #         best_file = "best.wav"
    #         play_path = os.path.join(QUICK_VOICE_FOLDER, CURRENT_USER_STYLE, feedback_mode, best_file)
    #         if os.path.exists(play_path):
    #             print(f"[播放] 完美稱讚語音: {best_file}")
    #             play_quick_voice(play_path)
    #         else:
    #             print(f"找不到完美稱讚語音: {play_path}")
    #             print("請確認是否已在該風格資料夾中放入 best.wav")

    PREVIOUS_DATA = current_data

# Watchdog 即時監控
def wait_for_file_release(file_path, timeout=5):
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)
            return True  
        except (PermissionError, OSError, json.JSONDecodeError):
            time.sleep(0.4)  
    return False

LAST_TRIGGER_TIME = 0
DELAY_SEC = 4

class JsonHandler(FileSystemEventHandler):
    def on_created(self, event):
        global LAST_TRIGGER_TIME, FILE_PROCESS_COUNT, stop_monitoring,ROUND_PUNCH_ACCUMULATOR 
        now = time.time()
        
        if now - LAST_TRIGGER_TIME < DELAY_SEC:
            print("冷卻時間內，略過觸發")
            return  

        LAST_TRIGGER_TIME = now 
        if not event.is_directory and event.src_path.endswith("json"):
            print(f"\n偵測到新 JSON 檔案：{event.src_path}")
            if wait_for_file_release(event.src_path):
                process_feedback_with_style_logic(event.src_path)
                
                # [新增] 計數與重置邏輯
                FILE_PROCESS_COUNT += 1
                print(f"目前已處理 {FILE_PROCESS_COUNT}/{MAX_FILES_BEFORE_RESET} 個檔案")
                
                if FILE_PROCESS_COUNT >= MAX_FILES_BEFORE_RESET:
                    print("\n>>> 已達 6 次一個round，換下一個新的user體驗")
                    ROUND_PUNCH_ACCUMULATOR=0
                    stop_monitoring = True # 通知主迴圈停止
                    
            else:
                print(f"無法讀取：{event.src_path}")
if __name__ == "__main__":
    print("\n開始處理原先資料庫並分配權重")
    load_all_json_files() 
    if not boxing_df.empty:
        training_data = boxing_df.to_dict('records')
        init_style_weights(training_data)
    else:
        print("沒有過去資料，無法建立權重比較。")
    if not pygame.mixer.get_init():
            pygame.mixer.init()
    while True:
        # 重置計數器與
        FILE_PROCESS_COUNT = 0
        stop_monitoring = False 
        
        print("\n正在重新載入最新資料庫")
        # 1. 將載入與訓練移入迴圈，確保讀取到新檔案
        load_all_json_files()
        if not STYLE_WEIGHT_MAP and not boxing_df.empty:
            print("檢測首次資料移失，重新建立風格權重系統.")
            training_data = boxing_df.to_dict('records')
            init_style_weights(training_data)

        # 2. 選擇基準檔案 (加入排序與檔名支援)
        files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".json")]
        # 依照修改時間排序，新的檔案會在最下面
        files.sort(key=lambda f: os.path.getmtime(os.path.join(FOLDER_PATH, f)))
        
        if not files:
            print("無檔案可選，程式結束")
            break 
            
        print("\n 請選擇一個檔案作為 User 過去資料風格 (可輸入編號或完整檔名)")
        # 顯示最後 10 筆就好，避免列表太長，或者您可以保留全部顯示
        start_idx = max(0, len(files) - 15)
        if start_idx > 0: print(f"... (略過前 {start_idx} 筆) ...")
        
        for i in range(start_idx, len(files)):
            print(f"{i}: {files[i]}")
        
        selected_file = None
        
        # 輸入判斷迴圈，直到輸入正確為止
        while selected_file is None:
            user_input = input(f"\n請輸入檔案編號 (0~{len(files)-1}) 或 完整檔名 (輸入 q 離開): ").strip()
            
            if user_input.lower() == 'q':
                print("程式結束")
                exit()
            
            # 情況 A: 輸入的是數字 (編號)
            if user_input.isdigit():
                idx = int(user_input)
                if 0 <= idx < len(files):
                    selected_file = os.path.join(FOLDER_PATH, files[idx])
                else:
                    print(f"編號錯誤！請輸入 0 到 {len(files)-1} 之間的數字。")
            
            # 情況 B: 輸入的是檔名 (字串)
            else:
                # 自動補齊 .json (如果使用者沒打)
                potential_name = user_input if user_input.endswith(".json") else user_input + ".json"
                if potential_name in files:
                    selected_file = os.path.join(FOLDER_PATH, potential_name)
                else:
                    print(f"找不到檔案：{user_input}，請檢查名稱是否正確。")
        while True:
            lang_input = input("\n請選擇語音語言 (C:Chinese / E:English / J:Japanese) : ").strip().upper()
            
            if lang_input == 'E':
                CURRENT_VOICE_FOLDER = ENG_QUICK_VOICE_FOLDER
                print(">> 已切換為：英文語音")
                break # 輸入正確，跳出迴圈
            elif lang_input == 'J':
                CURRENT_VOICE_FOLDER = JAP_QUICK_VOICE_FOLDER
                print(">> 已切換為：日文語音")
                break
            elif lang_input == 'C':
                CURRENT_VOICE_FOLDER = QUICK_VOICE_FOLDER
                print(">> 已切換為：中文語音")
                break
            else:
                print("輸入錯誤！請只輸入 C, E ,J")
        # 成功選取檔案後，繼續執行
        try:
            print(f"已選擇檔案：{os.path.basename(selected_file)}")
            print("正在建立 User 上一次基準數據...")
            PREVIOUS_DATA = extract_features_for_rf(selected_file)
            previousboxstyle_totalpunchnum= PREVIOUS_DATA.get('totalPunchNum', 0)
            print(f"基準數據已建立 (Score: {PREVIOUS_DATA['score']})")
            
            # 3. 準備資料給 GPT
            gpt_features = prepare_data_for_gpt(selected_file)
            print(f"這邊是傳入LLM的輸入{gpt_features}")
            if gpt_features:
                gpt_result_text = ask_gpt_for_style(gpt_features)
                CURRENT_USER_STYLE = determine_style_from_gpt_result(gpt_result_text)
                print(f"\n拳擊風格為: {CURRENT_USER_STYLE} <<<\n")
            else:
                print("從 GPT 獲取風格失敗，使用預設風格 Power")
                CURRENT_USER_STYLE = "maxPunchPower"
                
        except Exception as e:
            print(f"發生異常: {e}")
            traceback.print_exc()
            CURRENT_USER_STYLE = "maxPunchPower"
            print("使用預設風格 Power")

        # 5. 啟動監控
        event_handler = JsonHandler()
        observer = Observer()
        
        if not os.path.exists(MONITOR_FOLDER):
            os.makedirs(MONITOR_FOLDER)
            print(f"建立監控資料夾: {MONITOR_FOLDER}")
            
        observer.schedule(event_handler, MONITOR_FOLDER, recursive=False)
        print(f"開始監控資料夾：{MONITOR_FOLDER}")
        print(f"監控中... (蒐集滿 {MAX_FILES_BEFORE_RESET} 個檔案後將重置)")
        
        observer.start()
        
        try:
            while not stop_monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            observer.join()
            print("程式終止")
            exit() # 直接結束整個程式
            
        # 當 stop_monitoring 變成 True，會執行到這裡
        observer.stop()
        observer.join()
        print("\nround結束，準備讀取下一位user \n")
        print("等待 5 秒讓您拖入新檔案...")
        time.sleep(5)