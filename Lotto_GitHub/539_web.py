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
import altair as alt

# --- 1. 系統設定 ---
st.set_page_config(
    page_title="台彩數據中心 v26.0 (資料庫重構版)", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. CSS 視覺美化 ---
css_code = """
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
    h1, h2, h3 { font-family: "Microsoft JhengHei", sans-serif; color: #1b5e20; font-weight: bold; }
    .lottery-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold;
        margin: 3px; box-shadow: inset -3px -3px 8px rgba(0,0,0,0.2), 1px 1px 3px rgba(0,0,0,0.2);
        border: 1px solid #bdc3c7; font-family: Arial, sans-serif;
    }
    .special-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold; color: white;
        margin: 3px; margin-left: 15px;
        background: radial-gradient(circle at 30% 30%, #ff5252, #b71c1c);
        box-shadow: 0 0 8px rgba(255, 82, 82, 0.6);
        border: 2px solid #ffcdd2; font-family: Arial, sans-serif;
    }
    .ball-white { background: radial-gradient(circle at 30% 30%, #ffffff, #e0e0e0); color: #333; }
    .ball-green { background: radial-gradient(circle at 30% 30%, #81c784, #388e3c); color: white; }
    .ball-blue  { background: radial-gradient(circle at 30% 30%, #64b5f6, #1976d2); color: white; }
    .ball-yellow{ background: radial-gradient(circle at 30% 30%, #fff176, #fbc02d); color: #333; }
    .ball-red   { background: radial-gradient(circle at 30% 30%, #e57373, #d32f2f); color: white; }
    .ball-gold  { background: radial-gradient(circle at 30% 30%, #ffd54f, #ffa000); color: white; border: 2px solid #fff; }
    .stCard {
        background: rgba(255, 255, 255, 0.85); padding: 15px;
        border-radius: 12px; border: 1px solid #c8e6c9; margin-bottom: 10px;
        text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .zone-label { font-size: 12px; color: #666; margin-bottom: 4px; display: block; width: 100%; }
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)

# --- 3. 資料結構與設定 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

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
        "num_count": 6, "num_range": (1, 38), "has_special": True, "special_is_zone2": True, "special_range": (1, 8), "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    }
}

# --- 4. 核心功能：日期處理與遊戲辨識 ---

def parse_date(date_val):
    """強力日期解析，支援民國年與西元年"""
    d_str = str(date_val).strip()
    try:
        # 嘗試標準格式 YYYY/MM/DD or YYYY-MM-DD
        return pd.to_datetime(d_str).strftime('%Y-%m-%d')
    except:
        # 嘗試民國年 (e.g., 96/01/01)
        match = re.match(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', d_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if year < 1911: year += 1911
            return f"{year}-{month:02d}-{day:02d}"
    return None

def detect_game_type(filename, df_head):
    """判斷檔案屬於哪種遊戲"""
    filename = filename.lower()
    # 1. 優先看內容 (有些檔名是亂碼)
    if '遊戲名稱' in df_head.columns:
        val = str(df_head.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in val: return game
            
    # 2. 再看檔名
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    return None

# --- 5. 資料庫重整引擎 (核心修復邏輯) ---

def rebuild_databases():
    """
    暴力掃描 data 資料夾下所有 CSV，重新歸類並建立主資料庫
    """
    st.toast("🏗️ 開始掃描並重整所有資料庫...")
    
    # 準備暫存區
    storage = {g: [] for g in GAME_CONFIG.keys()}
    file_count = 0
    
    # 1. 解壓所有 ZIP
    zip_files = glob.glob(os.path.join(DATA_DIR, "*.zip")) + glob.glob("*.zip")
    for z in zip_files:
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        except: pass

    # 2. 遞迴搜尋所有 CSV (包含 data/2007/, data/2008/ 等子目錄)
    all_csv = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    
    # 排除主資料庫本身，避免重複讀取
    db_filenames = [os.path.basename(cfg['db_file']) for cfg in GAME_CONFIG.values()]
    target_files = [f for f in all_csv if os.path.basename(f) not in db_filenames and "pred_" not in f]
    
    progress_bar = st.progress(0)
    
    for i, file_path in enumerate(target_files):
        progress_bar.progress((i + 1) / len(target_files), text=f"處理中: {os.path.basename(file_path)}")
        
        try:
            # 嘗試讀取
            try: df = pd.read_csv(file_path, encoding='cp950', on_bad_lines='skip')
            except: 
                try: df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                except: continue
            
            # 欄位清理
            df.columns = [str(c).strip() for c in df.columns]
            
            # 判斷遊戲
            game_type = detect_game_type(os.path.basename(file_path), df.head(1))
            if not game_type: continue
            
            cfg = GAME_CONFIG[game_type]
            
            # 解析資料
            if '開獎日期' in df.columns:
                for _, row in df.iterrows():
                    try:
                        d_str = parse_date(row['開獎日期'])
                        if not d_str: continue
                        
                        nums = []
                        for k in range(1, cfg["num_count"] + 1):
                            col = f'獎號{k}'
                            if col in df.columns: nums.append(int(row[col]))
                        if len(nums) != cfg["num_count"]: continue
                        
                        sp = []
                        if cfg["has_
