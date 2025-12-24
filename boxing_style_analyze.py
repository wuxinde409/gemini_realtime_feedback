import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 設定 ---
FOLDER_PATH = "./processed_users1/"

def calculate_path_length(logs):
    """計算軌跡總長度 (歐幾里得距離總和)"""
    if not logs or len(logs) < 2:
        return 0.0
    coords = np.array([[p['x'], p['y'], p['z']] for p in logs])
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

def get_ranges(logs):
    """計算移動範圍 (最大值 - 最小值)"""
    if not logs: return None
    xs = [p['x'] for p in logs]
    ys = [p['y'] for p in logs]
    zs = [p['z'] for p in logs]
    return {
        'x_range': max(xs) - min(xs),
        'y_range': max(ys) - min(ys),
        'z_range': max(zs) - min(zs)
    }

def analyze_group_metrics(file_list, folder_path, group_name):
    """群組分析函式 - 回傳詳細數據供 RF 使用"""
    # 儲存該群組所有人的詳細數據 (用於 RF 分析)
    group_data_list = []
    hit_positions_r = []
    hit_positions_l = []

    # 用於計算平均值的暫存列表
    stats = {
        'total_user_body_movement': [],
        'r_hand_move_per_punch': [],
        'l_hand_move_per_punch': [],
        'total_hand_move_per_punch': [],
        'range_x': [], 'range_y': [], 'range_z': []
    }

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # 讀取基礎數據
            summary = data.get('summary', {})
            score = summary.get('score', 0)
            punch_num = summary.get('totalPunchNum', 1)
            if punch_num == 0: punch_num = 1
            
            # 讀取 Logs
            player_logs = data.get('playerPosLogs', [])
            r_hand_logs = data.get('playerRHandPosLogs', [])
            l_hand_logs = data.get('playerLHandPosLogs', [])
            
            # --- 計算新特徵 (Formative Features) ---
            body_dist = calculate_path_length(player_logs)
            r_dist = calculate_path_length(r_hand_logs)
            l_dist = calculate_path_length(l_hand_logs)
            
            r_move_per = r_dist / punch_num
            l_move_per = l_dist / punch_num
            total_move_per = (r_dist + l_dist) / punch_num
            
            rng = get_ranges(player_logs)
            rng_x = rng['x_range'] if rng else 0
            rng_y = rng['y_range'] if rng else 0
            rng_z = rng['z_range'] if rng else 0
            
            # 收集統計數據
            stats['total_user_body_movement'].append(body_dist)
            stats['r_hand_move_per_punch'].append(r_move_per)
            stats['l_hand_move_per_punch'].append(l_move_per)
            stats['total_hand_move_per_punch'].append(total_move_per)
            stats['range_x'].append(rng_x)
            stats['range_y'].append(rng_y)
            stats['range_z'].append(rng_z)

            record = {
                'score': score,
                'maxPunchPower': summary.get('maxPunchPower', 0),
                'maxPunchSpeed': summary.get('maxPunchSpeed', 0),
                'minReactionTime': summary.get('minReactionTime', 0),
                'hitRate': summary.get('hitRate', 0),
                # 新特徵
                'total_user_body_movement': body_dist,
                'total_hand_move_per_punch': total_move_per, 
                'totalPunchNum': punch_num,
                'range_x': rng_x,
                'range_y': rng_y,
                'range_z': rng_z
            }
            group_data_list.append(record)

            # --- 提取打擊落點 (維持原邏輯) ---
            punch_times = data.get('punchTimeCode', [])
            punch_indices = data.get('puncherIdx', [])
            total_time = punch_times[-1] if punch_times else 1
            if total_time <= 0: total_time = 1
            fps_r = len(r_hand_logs) / total_time if r_hand_logs else 30
            fps_l = len(l_hand_logs) / total_time if l_hand_logs else 30
            
            for i, t in enumerate(punch_times):
                hand_idx = punch_indices[i]
                target_log = r_hand_logs if hand_idx == 0 else l_hand_logs
                fps = fps_r if hand_idx == 0 else fps_l
                idx = int(t * fps)
                if idx < len(target_log):
                    pos = target_log[idx]
                    if hand_idx == 0: hit_positions_r.append((pos['x'], pos['y']))
                    else: hit_positions_l.append((pos['x'], pos['y']))

    # --- 輸出文字結果 ---
    print(f"分析結果 (針對 {group_name}):")
    print(f"1. 平均左手移動量 : {np.mean(stats['l_hand_move_per_punch']):.3f} m")
    print(f"1. 平均右手移動量 : {np.mean(stats['r_hand_move_per_punch']):.3f} m")
    print(f"1. 整場比賽雙手總移動量 (每拳): {np.mean(stats['total_hand_move_per_punch']):.3f} m")
    print(f"2. 整場比賽身體總移動距離: {np.mean(stats['total_user_body_movement']):.3f} m")
    print("3. 身體位移範圍 (平均):")
    print(f"左右移動範圍 (X Range): {np.mean(stats['range_x']):.3f} m")
    print(f"上下移動範圍 (Y Range): {np.mean(stats['range_y']):.3f} m")
    print(f" 前後移動範圍 (Z Range): {np.mean(stats['range_z']):.3f} m")
    print("")

    return hit_positions_r, hit_positions_l, group_data_list


def run_random_forest_analysis(all_data):
    """執行 Random Forest 分析特徵重要性"""
    if not all_data: return

    df = pd.DataFrame(all_data)
    
    # 定義特徵 X 與 目標 y
    # 我們把「新特徵」和「舊特徵」放在一起比較
    feature_cols = [
        'maxPunchPower', 'maxPunchSpeed', 'minReactionTime', 'hitRate', # 舊特徵
        'total_user_body_movement', 'total_hand_move_per_punch', 'range_x', 'range_y','range_z'        # 新特徵 (移動/範圍)
    ]
    summary_feature_cols =['maxPunchPower', 'maxPunchSpeed', 'minReactionTime', 'hitRate']
    formative_feature_cols=['total_user_body_movement', 'total_hand_move_per_punch', 'range_x', 'range_y','range_z']
    targets = ["maxPunchSpeed", "minReactionTime", "maxPunchPower"]
    # 確保資料乾淨
    df = df.dropna()
    X = df[feature_cols]
    y = df['score']

    # 訓練模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 評估準確度
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    


    print(f"(R2 Score): {r2:.3f}")

    
    # 輸出特徵重要性
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("影響score的因素排名 :")
    for name, val in importances.items():
        # 標記新特徵以便識別
        print(f" {name:<20}: {val:.4f}")
    
def run_style_specific_analysis(all_data): #有風格標示的

    if not all_data: return

    df = pd.DataFrame(all_data)
    df = df.dropna()

    # 定義基礎欄位
    # 動作特徵 (Formative): 玩家怎麼動
    formative_cols = [
        'total_user_body_movement', 'total_hand_move_per_punch', 
        'range_x', 'range_y', 'range_z'
    ]
    
    # 基礎數據 (Summary): 結果數據
    base_summary_cols = ['maxPunchPower', 'maxPunchSpeed', 'minReactionTime', 'hitRate']
    
    targets = ["maxPunchPower", "maxPunchSpeed", "minReactionTime"]

    # print(f"分析樣本數: {len(df)}")

    for target in targets:
        print(f"\n當前分析風格: {target}")

     
        # X_features = 動作數據 + 其他身體素質 (排除 Target 本身與 Score)
        other_summary_cols = [col for col in base_summary_cols if col != target]
        feature_cols = formative_cols + other_summary_cols
        
        X = df[feature_cols]
        y_target = df[target]
        rf_target = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_target.fit(X, y_target)
        
        print(f"想提升 {target}，應該專注在:")
        imp_target = pd.Series(rf_target.feature_importances_, index=feature_cols).sort_values(ascending=False)
        for name, val in imp_target.head(5).items():
            print(f"{name:<25} (權重: {val:.4f})")
            
     
        # 邏輯：把「Target 本身」加回去 X 裡面，看看它對 Score 的影響力
        # y = Score
        
        X_score = df[feature_cols + ['totalPunchNum']]
        y_score = df['score']
        
        rf_score = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_score.fit(X_score, y_score)
        
        print(f"在 {target} 的風格中，影響 Score 的是:")
        imp_score = pd.Series(rf_score.feature_importances_, index=X_score.columns).sort_values(ascending=False)
        for name, val in imp_score.head(5).items():
            # 使用文字標記當前 Target
            marker = " [Target]" if name == target else ""
            print(f"{name:<25} (權重: {val:.4f}){marker}")

        # 相關係數驗證 
        # corr = df[target].corr(df['score'])
        # print(f"(補充: {target} 與 Score 的直接相關係數 r = {corr:.4f})")



def calculate_distribution_stats(hit_list, label):
    """計算打擊分佈的中心與離散程度"""
    if not hit_list:
        return
    
    # 轉成 Numpy Array 方便計算
    data = np.array(hit_list)
    x = data[:, 0]
    y = data[:, 1]
    
    # 1. 中心點 (Mean)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 2. 離散程度 (Standard Deviation)
    std_x = np.std(x)
    std_y = np.std(y)

    print(f"[{label}]")
    print(f" 打擊Centroid: (X={mean_x:.3f}, Y={mean_y:.3f})")
    print(f" 離散程度: X軸={std_x:.3f}, Y軸={std_y:.3f}")
    # print(f"   - 綜合精準度 (Spread Radius): {spread_radius:.3f} (數值越小越準)")

    
    return mean_x, mean_y

def analyze_score_dispersion_correlation(file_list, folder_path):

    print("開始分析分數與(STD)關聯性 ")
    
    user_stats = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 取得分數
                score = data.get('summary', {}).get('score', 0)
                
                # --- 取得該用戶的打擊點 (複製原本的邏輯) ---
                r_hand_logs = data.get('playerRHandPosLogs', [])
                l_hand_logs = data.get('playerLHandPosLogs', [])
                punch_times = data.get('punchTimeCode', [])
                punch_indices = data.get('puncherIdx', [])
                
                total_time = punch_times[-1] if punch_times else 1
                if total_time <= 0: total_time = 1
                fps_r = len(r_hand_logs) / total_time if r_hand_logs else 30
                fps_l = len(l_hand_logs) / total_time if l_hand_logs else 30
                
                user_hits_x = []
                user_hits_y = []

                for i, t in enumerate(punch_times):
                    hand_idx = punch_indices[i]
                    target_log = r_hand_logs if hand_idx == 0 else l_hand_logs
                    fps = fps_r if hand_idx == 0 else fps_l
                    idx = int(t * fps)
                    
                    if idx < len(target_log):
                        pos = target_log[idx]
                        user_hits_x.append(pos['x'])
                        user_hits_y.append(pos['y'])
                
                # --- 計算該用戶的 STD (散度) ---
                # 如果打擊點太少，標準差無意義，跳過
                if len(user_hits_x) < 2:
                    continue
                
                # 計算 X 與 Y 的標準差，取平均作為該用戶的「總體散度」
                std_x = np.std(user_hits_x)
                std_y = np.std(user_hits_y)
                # avg_dispersion = (std_x + std_y) / 2
                
                user_stats.append({
                    'score': score,
                    'Xdispersion': std_x,
                    'Ydispersion': std_y
                })
                
        except Exception as e:
            continue

    # 轉成 DataFrame
    df = pd.DataFrame(user_stats)
    
    if df.empty:
        print("數據不足，無法計算相關性")
        return

    # 計算相關係數 (Pearson Correlation)
    corr_x = df['score'].corr(df['Xdispersion'])
    corr_y = df['score'].corr(df['Ydispersion'])
    
    print(f"分析樣本數: {len(df)} 人")
    print(f"X軸 散度相關係數 (左右穩定度): {corr_x:.4f}")
    print(f"Y軸 散度相關係數 (高度穩定度): {corr_y:.4f}")

    # --- 繪圖 (左右兩張圖並列) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 圖 1: Score vs X Dispersion (左右)
    axes[0].scatter(df['score'], df['Xdispersion'], alpha=0.5, color='teal', label='Users')
    z_x = np.polyfit(df['score'], df['Xdispersion'], 1)
    p_x = np.poly1d(z_x)
    axes[0].plot(df['score'], p_x(df['score']), "r--", linewidth=2, label=f'(r={corr_x:.2f})')
    axes[0].set_title("Score vs. X Dispersion (Left/Right)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(" Score")
    axes[0].set_ylabel("X STD ")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 圖 2: Score vs Y Dispersion (上下)
    axes[1].scatter(df['score'], df['Ydispersion'], alpha=0.5, color='orange', label='Users')
    z_y = np.polyfit(df['score'], df['Ydispersion'], 1)
    p_y = np.poly1d(z_y)
    axes[1].plot(df['score'], p_y(df['score']), "r--", linewidth=2, label=f'(r={corr_y:.2f})')
    axes[1].set_title("Score vs. Y Dispersion ", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Score")
    axes[1].set_ylabel("Y STD ")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("Comparison_XY_Dispersion.png")
    plt.show()


def analyze_top_score_players(folder_path):
    print(f"開始分析: {folder_path}")
    
    # --- 第一階段：讀取所有分數並計算門檻 ---
    all_files_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        if os.path.getsize(file_path) > 0:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    score = data.get("summary", {}).get("score", 0)
                    all_files_data.append({"filename": filename, "score": score})
            except:
                continue

    if not all_files_data:
        print("沒有找到有效的 JSON 檔案")
        return

    df_scores = pd.DataFrame(all_files_data)
    score_threshold = df_scores['score'].quantile(0.90)
    
    top_files_list = df_scores[df_scores['score'] >= score_threshold]['filename'].tolist()
    bottom_files_list = df_scores[df_scores['score'] < score_threshold]['filename'].tolist()
    

    all_files_list = top_files_list + bottom_files_list
    
    print(f"Top 10% 分數門檻: {score_threshold:.2f}")


    # 1. 分析 Top 10%
    hit_r_top, hit_l_top, data_top = analyze_group_metrics(top_files_list, folder_path, "score 前10% 人群")
    
    # 2. 分析 Bottom 90%
    hit_r_bottom, hit_l_bottom, data_bottom = analyze_group_metrics(bottom_files_list, folder_path, "score 後段90% 人群")
    
    # 3. 分析 全部人群 (使用正確的檔名列表)
    hit_r_all, hit_l_all, data_all = analyze_group_metrics(all_files_list, folder_path, "全部的人群")

    # --- 第三階段：AI 關聯性分析 ---
    # 使用所有人的詳細特徵數據進行訓練
    # run_random_forest_analysis(data_all)
    run_style_specific_analysis(data_all)
    analyze_score_dispersion_correlation(all_files_list, folder_path)

    print("打擊點分析")
    calculate_distribution_stats(hit_r_top, "前10% - 右手")
    calculate_distribution_stats(hit_l_top, "前10% - 左手")
    calculate_distribution_stats(hit_r_bottom, "後段90%- 右手")
    calculate_distribution_stats(hit_l_bottom, "後段90%- 左手")
    calculate_distribution_stats(hit_r_all, "All Users - 右手")
    calculate_distribution_stats(hit_l_all, "All Users - 左手")

    # --- 第五階段：繪製並列比較圖 ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    titles = ["Top 10% Hit Distribution", "Bottom 90% Hit Distribution" ,"ALL Users Hit Distribution"]
    datasets = [(hit_r_top, hit_l_top), (hit_r_bottom, hit_l_bottom), (hit_r_all, hit_l_all)]
    
    for i, ax in enumerate(axes):
        hit_r, hit_l = datasets[i]
        
        # 繪製右手 (藍色)
        if hit_r:
            xr = [p[0] for p in hit_r]
            yr = [p[1] for p in hit_r]
            ax.scatter(xr, yr, c='blue', s=10, alpha=0.3, label='Right Hand', marker='o') 
            

        # 繪製左手 (紅色)
        if hit_l:
            xl = [p[0] for p in hit_l]
            yl = [p[1] for p in hit_l]
            ax.scatter(xl, yl, c='red', s=10, alpha=0.3, label='Left Hand', marker='o') 
        

        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position (Left / Right)")
        ax.set_ylabel("Y Position (Height)")
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 固定軸範圍方便對比
        ax.set_xlim(17, 23)
        ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plt.savefig("Comparison_Hit_Distribution.png")
    plt.show()
# 執行分析
if __name__ == "__main__":
    analyze_top_score_players(FOLDER_PATH)