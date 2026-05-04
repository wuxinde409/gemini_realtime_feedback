import pandas as pd
import os

CACHE_FILE = "boxing_history_cache.pkl"

if os.path.exists(CACHE_FILE):
    print(f" 正在讀取 {CACHE_FILE} ...")
    
    #  使用 pandas 讀取 pkl
    df = pd.read_pickle(CACHE_FILE)
    
    print(f"\n 讀取成功: {df.shape}")
    
    # 印出前 5 筆資料 (看數值是否正常)
    print("前 100 筆資料預覽：")
    print(df.head())
    
    # 印出所有欄位名稱 (確認 avgPunchSpeed 等新欄位有無存進去)
    print("包含欄位：")
    print(df.columns.tolist())
    
    # 檢查是否有檔名欄位 (這是我們增量更新的關鍵)
    if 'filename' in df.columns:
        print("\n 'filename' 欄位存在，增量更新功能可正常運作。")
    else:
        print("\n 警告：缺少 'filename' 欄位，增量更新可能會失效！")

else:
    print(f" 找不到 {CACHE_FILE}，請先執行主程式生成快取。")