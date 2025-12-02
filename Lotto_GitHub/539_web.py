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
    page_title="台彩數據中心 v24.0", 
    page_icon="🕰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. CSS 視覺美化 (淺綠水墨風 + 3D彩球) ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f7f4;
        background-image: url("https://www.transparenttextures.com/patterns/rice-paper-2.png");
        color: #2c3e50;
    }
    section[data-testid="stSidebar"] {
        background-color: #e8f5e9;
        border-right: 1px solid #c8e6c9;
    }
    h1, h2, h3 {
        font-family: "Microsoft JhengHei", sans-serif;
        color: #1b5e20;
        font-weight: bold;
    }
    .lottery-ball {
        display: inline-block; width: 40px; height: 40px; line-height: 40px;
        border-radius: 50%; text-align: center; font-weight: bold;
        margin: 4px; box-shadow: inset -5px -5px 10px rgba(0,0,0,0.3), 2px 2px 5px rgba(0,0,0,0.2);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5); border: 1px solid rgba(0,0,0,0.1);
    }
    .ball-white { background: radial-gradient(circle at 30% 30%, #ffffff, #dcdcdc); color: #333; text-shadow: none; }
    .ball-green { background: radial-gradient(circle at 30% 30%, #66bb6a, #2e7d32); color: white; }
    .ball-blue  { background: radial-gradient(circle at 30% 30%, #42a5f5, #1565c0); color: white; }
    .ball-yellow{ background: radial-gradient(circle at 30% 30%, #ffee58, #fbc02d); color: #333; text-shadow: none;}
    .ball-red   { background: radial-gradient(circle at 30% 30%, #ef5350, #c62828); color: white; }
    .ball-gold  { background: radial-gradient(circle at 30% 30%, #ffd700, #ff8f00); color: white; border: 2px solid #fff; box-shadow: 0 0 10px #ffd700; }
    .stCard {
        background: rgba(255, 255, 255, 0.7); padding: 15px;
        border-radius: 10px; border: 1px solid #c8e6c9; margin-bottom: 10px;
    }
    .mirror-alert {
        background: linear-gradient(90deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        padding: 15px; border-radius: 10px; border: 2px solid #ff6b6b;
        color: #880e4f; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 資料結構 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 自動解壓
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 2:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except: pass

# 遊戲設定
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "db_file": os.path.join(DATA_DIR, "db_539.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_539.csv"),
        "num_count": 5, "num_range": (1, 39), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "db_file": os.path.join(DATA_DIR, "db_lotto649.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_lotto649.csv"),
        "num_count": 6, "num_range": (1, 49), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "db_file": os.path.join(DATA_DIR, "db_super.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_super.csv"),
        "num_count": 6, "num_range": (1, 38), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    }
}

# --- 4. 核心函式 ---

def get_ball_html(num, count):
    if count >= 6: color_class = "ball-gold"
    elif count == 5: color_class = "ball-red"
    elif count == 4: color_class = "ball-yellow"
    elif count == 3: color_class = "ball-blue"
    elif count == 2: color_class = "ball-green"
    else: color_class = "ball-white"
    return f'<div class="lottery-ball {color_class}">{num:02d}</div>'

def render_prediction_row(nums, counts):
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

@st.cache_data(show_spinner=False, ttl=60)
def load_all_data(game_name):
    if game_name not in GAME_CONFIG: return pd.DataFrame()
    cfg = GAME_CONFIG[game_name]
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    merged_data = []
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
                    for _, row in df.iterrows():
                        try:
                            d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                            nums = sorted([int(row[f'獎號{i}']) for i in range(1, cfg["num_count"] + 1)])
                            if len(nums) != cfg["num_count"]: continue
                            sp = []
                            if cfg["has_special"]:
                                if "第二區" in df.columns: sp = [int(row['第二區'])]
                                elif "特別號" in df.columns: sp = [int(row['特別號'])]
                                else: sp = [0]
                            entry = [d_str] + nums + sp + ["Official"]
                            if len(entry) == len(cfg["cols"]): merged_data.append(entry)
                        except: continue
                elif 'Date' in df.columns:
                    valid_cols = [c for c in cfg["cols"] if c in df.columns]
                    temp_df = df[valid_cols].copy()
                    if "Source" not in temp_df.columns: temp_df["Source"] = "Auto"
                    if len(temp_df.columns) == len(cfg["cols"]): merged_data.extend(temp_df.values.tolist())
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
            if len(line) < 10: continue
            match = re.search(r'(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})', line)
            if not match: continue
            d_str = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            if d_str < "2025-01-01": continue
            clean = line.replace(match.group(0), "")
            all_n = [int(n) for n in re.findall(r'\b\d{1,2}\b', clean)]
            valid_n, sp_n = [], []
            if game_name == "今彩539": valid_n = sorted([n for n in all_n if 1<=n<=39])[:5]
            elif game_name == "大樂透":
                t = [n for n in all_n if 1<=n<=49]
                if len(t)>=7: valid_n = sorted(t[:6]); sp_n = [t[6]]
            elif game_name == "威力彩":
                if len(all_n)>=7: valid_n = sorted([n for n in all_n[:6] if 1<=n<=38]); sp_n = [all_n[6]] if 1<=all_n[6]<=8 else [1]
            if len(valid_n) == cfg["num_count"]:
                entry = [d_str] + valid_n + sp_n + ["Web_Crawl"]
                if len(entry) == len(cfg["cols"]): new_rows.append(entry)
    except: pass
    if new_rows:
        filename = f"Daily_Patch_{game_
