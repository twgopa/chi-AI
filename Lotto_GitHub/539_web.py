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
import collections

# --- 1. 系統設定 ---
st.set_page_config(
    page_title="台彩數據中心 v23.0", 
    page_icon="🔮", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. CSS 視覺美化 (淺綠水墨風 + 3D彩球) ---
st.markdown("""
<style>
    /* 全站背景：淺綠水墨紙質感 */
    .stApp {
        background-color: #f0f7f4;
        background-image: url("https://www.transparenttextures.com/patterns/rice-paper-2.png");
        color: #2c3e50;
    }
    
    /* 側邊欄美化 */
    section[data-testid="stSidebar"] {
        background-color: #e8f5e9;
        border-right: 1px solid #c8e6c9;
    }
    
    /* 標題字型 */
    h1, h2, h3 {
        font-family: "Microsoft JhengHei", sans-serif;
        color: #1b5e20;
        font-weight: bold;
    }

    /* 3D 彩球樣式基礎 */
    .lottery-ball {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
        font-family: Arial, sans-serif;
        margin: 4px;
        box-shadow: inset -5px -5px 10px rgba(0,0,0,0.3), 2px 2px 5px rgba(0,0,0,0.2);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* 顏色等級 (白綠藍黃紅金) */
    .ball-white { background: radial-gradient(circle at 30% 30%, #ffffff, #dcdcdc); color: #333; text-shadow: none; }
    .ball-green { background: radial-gradient(circle at 30% 30%, #66bb6a, #2e7d32); color: white; }
    .ball-blue  { background: radial-gradient(circle at 30% 30%, #42a5f5, #1565c0); color: white; }
    .ball-yellow{ background: radial-gradient(circle at 30% 30%, #ffee58, #fbc02d); color: #333; text-shadow: none;}
    .ball-red   { background: radial-gradient(circle at 30% 30%, #ef5350, #c62828); color: white; }
    .ball-gold  { background: radial-gradient(circle at 30% 30%, #ffd700, #ff8f00); color: white; border: 2px solid #fff; box-shadow: 0 0 10px #ffd700; }

    /* 區塊卡片化 */
    .stCard {
        background: rgba(255, 255, 255, 0.6);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c8e6c9;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 資料結構與路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 自動解壓 (容錯)
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 2:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except: pass

# 遊戲設定 (新增 pred_file 欄位，實現預測紀錄獨立)
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "db_file": os.path.join(DATA_DIR, "db_539.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_539.csv"), # 獨立預測檔
        "num_count": 5, "num_range": (1, 39), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "db_file": os.path.join(DATA_DIR, "db_lotto649.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_lotto649.csv"), # 獨立預測檔
        "num_count": 6, "num_range": (1, 49), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "db_file": os.path.join(DATA_DIR, "db_super.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_super.csv"), # 獨立預測檔
        "num_count": 6, "num_range": (1, 38), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    }
}

# --- 4. 核心函式庫 ---

def get_ball_html(num, count):
    """根據出現次數產生對應顏色的彩球 HTML"""
    if count >= 6: color_class = "ball-gold"
    elif count == 5: color_class = "ball-red"
    elif count == 4: color_class = "ball-yellow"
    elif count == 3: color_class = "ball-blue"
    elif count == 2: color_class = "ball-green"
    else: color_class = "ball-white"
    
    return f'<div class="lottery-ball {color_class}">{num:02d}</div>'

def render_prediction_row(nums, counts):
    """渲染整排彩球"""
    html = ""
    for n in nums:
        c = counts.get(n, 1)
        html += get_ball_html(n, c)
    return html

def detect_game_type(filename, df_head):
    filename = filename.lower()
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    if '遊戲名稱' in df_head.columns:
        val = str(df_head.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in val: return game
    return None

def process_bulk_files(uploaded_files):
    results = {g: 0 for g in GAME_CONFIG.keys()}
    temp_storage = {g: [] for g in GAME_CONFIG.keys()}
    
    for up_file in uploaded_files:
        try:
            if up_file.name.endswith('.zip'):
                with zipfile.ZipFile(up_file, 'r') as z: z.extractall(DATA_DIR)
                continue
            try: df = pd.read_csv(up_file, encoding='cp950', on_bad_lines='skip')
            except: 
                try: df = pd.read_csv(up_file, encoding='big5', on_bad_lines='skip')
                except: 
                    up_file.seek(0)
                    df = pd.read_csv(up_file, encoding='utf-8', on_bad_lines='skip')
            
            df.columns = [str(c).strip() for c in df.columns]
            game_type = detect_game_type(up_file.name, df.head(1))
            if not game_type: continue
            
            cfg = GAME_CONFIG[game_type]
            for _, row in df.iterrows():
                try:
                    d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                    nums = []
                    for k in range(1, cfg["num_count"] + 1):
                        col = f'獎號{k}'
                        if col in df.columns: nums.append(int(row[col]))
                    if len(nums) != cfg["num_count"]: continue
                    
                    sp = []
                    if cfg["has_special"]:
                        if "第二區" in df.columns: sp = [int(row['第二區'])]
                        elif "特別號" in df.columns: sp = [int(row['特別號'])]
                        else: sp = [0]
                    nums.sort()
                    entry = [d_str] + nums + sp + ["UserUpload"]
                    if len(entry) == len(cfg["cols"]): temp_storage[game_type].append(entry)
                except: continue
        except: continue

    for game, rows in temp_storage.items():
        if rows:
            cfg = GAME_CONFIG[game]
            new_filename = f"Upload_{game}_{int(time.time())}.csv"
            pd.DataFrame(rows, columns=cfg["cols"]).to_csv(os.path.join(DATA_DIR, new_filename), index=False)
            results[game] += len(rows)
    return results

@st.cache_data(show_spinner=False, ttl=60)
def load_all_data(game_name):
    if game_name not in GAME_CONFIG: return pd.DataFrame()
    cfg = GAME_CONFIG[game_name]
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    merged_data = []
    
    # 排除所有預測檔 (pred_*.csv) 和 日誌檔
    target_files = [f for f in all_files if "pred_" not in os.path.basename(f) and "log" not in os.path.basename(f)]

    for file_path in target_files:
        filename = os.path.basename(file_path)
        is_related = any(k in filename for k in cfg["keywords"])
        if not is_related and any(k in file_path for k in cfg["keywords"]): is_related = True
        
        if is_related:
            try:
                try: df = pd.read_csv(file_path, encoding='cp950', on_bad_lines='skip')
                except: 
                    try: df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    except: continue
                
                df.columns = [str(c).strip() for c in df.columns]
                
                if '開獎日期' in df.columns:
                    temp_rows = []
                    for _, row in df.iterrows():
                        try:
                            d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                            nums = [int(row[f'獎號{i}']) for i in range(1, cfg["num_count"] + 1)]
                            if len(nums) != cfg["num_count"]: continue
                            sp = []
                            if cfg["has_special"]:
                                if "第二區" in df.columns: sp = [int(row['第二區'])]
                                elif "特別號" in df.columns: sp = [int(row['特別號'])]
                                else: sp = [0]
                            nums.sort()
                            entry = [d_str] + nums + sp + ["Official"]
                            if len(entry) == len(cfg["cols"]): temp_rows.append(entry)
                        except: continue
                    merged_data.extend(temp_rows)

                elif 'Date' in df.columns:
                    valid_cols = [c for c in cfg["cols"] if c in df.columns]
                    temp_df = df[valid_cols].copy()
                    if "Source" not in temp_df.columns: temp_df["Source"] = "Auto"
                    if len(temp_df.columns) == len(cfg["cols"]):
                        merged_data.extend(temp_df.values.tolist())
            except: pass

    if merged_data:
        final_df = pd.DataFrame(merged_data, columns=cfg["cols"])
        final_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        final_df.sort_values(by='Date', ascending=True, inplace=True)
        return final_df
    return pd.DataFrame(columns=cfg["cols"])

def crawl_daily_web(game_name):
    if game_name not in GAME_CONFIG: return 0
    cfg = GAME_CONFIG[game_name]
    url = "https://i539.tw/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    new_rows = []
    try:
        res = requests.get(url, headers=headers, verify=False, timeout=5)
        res.encoding = 'utf-8'
        lines = res.text.split('\n')
        for line in lines:
            if len(line)
