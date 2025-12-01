import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import re
import urllib3
from datetime import datetime, timedelta
import glob
import time
import zipfile
import altair as alt

# --- 1. 系統設定 ---
st.set_page_config(page_title="台彩數據中心 v19.0 (賓果分流版)", page_icon="🎱", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 定義獨立的賓果資料庫路徑
BINGO_DB_FILE = os.path.join(DATA_DIR, "bingo_history.csv")

# 自動解壓
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 2:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except: pass

# --- 3. 遊戲設定 ---
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "db_file": "db_539.csv",
        "num_count": 5, "num_range": (1, 39), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "db_file": "db_lotto649.csv",
        "num_count": 6, "num_range": (1, 49), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "db_file": "db_super.csv",
        "num_count": 6, "num_range": (1, 38), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    },
    # 賓果獨立處理，但仍保留在選單以便切換
    "賓果賓果": {
        "keywords": ["賓果賓果", "Bingo"],
        "db_file": "bingo_history.csv", # 指定獨立檔案
        "num_count": 20, "num_range": (1, 80), "has_special": True, "enable_predict": False,
        "cols": ["Date", "Period", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N09", "N10", 
                 "N11", "N12", "N13", "N14", "N15", "N16", "N17", "N18", "N19", "N20", "Super", "Source"]
    }
}

# --- 4. 賓果專用爬蟲 (慢速寫入) ---

def crawl_bingo_slowly(target_date=None):
    """
    賓果專用爬蟲：
    1. 針對官方 API 抓取單日所有期數 (一天約203期)
    2. 支援 '慢速寫入' 模式，避免被鎖
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
        
    # 轉換成官方 API 需要的月份格式 (YYYY-MM)
    # 官方 API 是按月查詢的，這很棒，一次抓一個月
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    month_str = dt.strftime("%Y-%m")
    
    api_url = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/BingoResult"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.taiwanlottery.com/'
    }
    
    new_data = []
    print(f"🐢 [賓果慢爬蟲] 正在抓取 {month_str} 資料...")
    
    try:
        res = requests.get(api_url, params={"month": month_str}, headers=headers, verify=False, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if "content" in data and "bingoBingoResulList" in data["content"]:
                items = data["content"]["bingoBingoResulList"]
                # 官方回傳是整個月的，我們需要過濾出「目標日期」或是「全部更新」
                # 這裡策略：既然抓了就全存，反正會去重
                
                for item in items:
                    try:
                        d_str = item["drawDate"][:10] # YYYY-MM-DD
                        period = str(item["period"])
                        
                        # 提取 20 個號碼 (r01~r20)
                        nums = [int(item[f"r{k:02d}"]) for k in range(1, 21)]
                        nums.sort() # 賓果通常看排序
                        
                        # 超級獎號
                        super_num = int(item["bullEye"]) if "bullEye" in item else 0
                        
                        entry = [d_str, period] + nums + [super_num, "Official_API"]
                        new_data.append(entry)
                    except: continue
        
        # 慢速模擬：抓完一次休息一下 (雖然 API 是一次給整月，但在連續抓多月時很有用)
        time.sleep(3) 
        
    except Exception as e:
        print(f"❌ 賓果爬蟲錯誤: {e}")
        return []

    return new_data

def update_bingo_db():
    """執行賓果更新並寫入獨立資料庫"""
    # 1. 嘗試抓取本月
    now = datetime.now()
    data_this_month = crawl_bingo_slowly(now.strftime("%Y-%m-%d"))
    
    # 2. 嘗試抓取上個月 (避免跨月時漏掉)
    last_month = now.replace(day=1) - timedelta(days=1)
    data_last_month = crawl_bingo_slowly(last_month.strftime("%Y-%m-%d"))
    
    all_new = data_this_month + data_last_month
    
    if all_new:
        cfg = GAME_CONFIG["賓果賓果"]
        df_new = pd.DataFrame(all_new, columns=cfg["cols"])
        
        # 讀取舊資料
        if os.path.exists(BINGO_DB_FILE):
            try:
                df_old = pd.read_csv(BINGO_DB_FILE)
                # 合併
                df_final = pd.concat([df_old, df_new], ignore_index=True)
            except:
                df_final = df_new
        else:
            df_final = df_new
            
        # 去重 (依據 日期 + 期別)
        df_final.drop_duplicates(subset=['Date', 'Period'], keep='last', inplace=True)
        df_final.sort_values(by=['Date', 'Period'], ascending=True, inplace=True)
        
        # 存檔
        df_final.to_csv(BINGO_DB_FILE, index=False)
        return len(df_new) # 回傳抓到的筆數(含重複)
    
    return 0

# --- 5. 通用讀取與其他彩種爬蟲 (維持不變) ---
# (為了節省篇幅，這裡僅列出關鍵修改，請保持您原有的 process_bulk_files 等函式)
# 關鍵：load_all_data 需增加對賓果獨立檔案的支援

@st.cache_data(show_spinner=False, ttl=60)
def load_all_data(game_name):
    cfg = GAME_CONFIG.get(game_name)
    if not cfg: return pd.DataFrame()
    
    # 特殊路徑：賓果
    if game_name == "賓果賓果":
        if os.path.exists(BINGO_DB_FILE):
            try: return