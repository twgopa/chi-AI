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
st.set_page_config(page_title="台彩數據中心 v19.2 (語法修正版)", page_icon="🎱", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 定義獨立的賓果資料庫路徑
BINGO_DB_FILE = os.path.join(DATA_DIR, "bingo_history.csv")
LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# 自動解壓 ZIP
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            # 簡單判斷：若 CSV 很少，就解壓
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 2:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except:
        pass

# --- 3. 遊戲設定 ---
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "num_count": 5, "num_range": (1, 39), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "num_count": 6, "num_range": (1, 49), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "num_count": 6, "num_range": (1, 38), "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    },
    "賓果賓果": {
        "keywords": ["賓果賓果", "Bingo"],
        "num_count": 20, "num_range": (1, 80), "has_special": True, "enable_predict": False,
        "cols": ["Date", "Period", "N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N09", "N10", 
                 "N11", "N12", "N13", "N14", "N15", "N16", "N17", "N18", "N19", "N20", "Super", "Source"]
    },
    "3星彩": {
        "keywords": ["3星彩", "3 Star"],
        "num_count": 3, "num_range": (0, 9), "has_special": False, "enable_predict": False,
        "cols": ["Date", "D1", "D2", "D3", "Source"]
    },
    "4星彩": {
        "keywords": ["4星彩", "4 Star"],
        "num_count": 4, "num_range": (0, 9), "has_special": False, "enable_predict": False,
        "cols": ["Date", "D1", "D2", "D3", "D4", "Source"]
    }
}

# --- 4. 核心讀取與爬蟲 ---

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

def process_bulk_files(uploaded_files, progress_bar):
    results = {g: 0 for g in GAME_CONFIG.keys()}
    temp_storage = {g: [] for g in GAME_CONFIG.keys()}
    
    total = len(uploaded_files)
    for i, up_file in enumerate(uploaded_files):
        if progress_bar:
            progress_bar.progress((i + 1) / total, text=f"處理中: {up_file.name}")
        try:
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
            
            if game_type == "賓果賓果":
                for _, row in df.iterrows():
                    try:
                        d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                        period = str(row['期別'])
                        nums = [int(row[f'獎號{k}']) for k in range(1, 21)]
                        sp = [int(row['超級獎號'])] if '超級獎號' in df.columns else [0]
                        entry = [d_str, period] + nums + sp + ["UserUpload"]
                        if len(entry) == len(cfg["cols"]): temp_storage[game_type].append(entry)
                    except: continue
            else:
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
                        if cfg["enable_predict"]: nums.sort()
                        entry = [d_str] + nums + sp + ["UserUpload"]
                        if len(entry) == len(cfg["cols"]): temp_storage[game_type].append(entry)
                    except: continue
        except: continue

    for game, rows in temp_storage.items():
        if rows:
            cfg = GAME_CONFIG[game]
            if game == "賓果賓果":
                new_df = pd.DataFrame(rows, columns=cfg["cols"])
                if os.path.exists(BINGO_DB_FILE):
                    try:
                        old_df = pd.read_csv(BINGO_DB_FILE)
                        final = pd.concat([old_df, new_df], ignore_index=True)
                    except: final = new_df
                else: final = new_df
                final.drop_duplicates(subset=['Date', 'Period'], keep='last', inplace=True)
                final.sort_values(by=['Date', 'Period'], inplace=True)