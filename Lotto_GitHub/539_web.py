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
    page_title="台彩數據中心 v25.0", 
    page_icon="🎯", 
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
    h1, h2, h3 {
        font-family: "Microsoft JhengHei", sans-serif;
        color: #1b5e20;
        font-weight: bold;
    }
    /* 一般號碼球 (第一區) */
    .lottery-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold;
        margin: 3px; box-shadow: inset -3px -3px 8px rgba(0,0,0,0.2), 1px 1px 3px rgba(0,0,0,0.2);
        border: 1px solid #bdc3c7; font-family: Arial;
    }
    /* 第二區專用球 (紅球) */
    .special-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold; color: white;
        margin: 3px; margin-left: 10px; /* 與第一區隔開 */
        background: radial-gradient(circle at 30% 30%, #ff5252, #b71c1c);
        box-shadow: 0 0 8px rgba(255, 82, 82, 0.6);
        border: 2px solid #ffcdd2;
    }
    .ball-white { background: radial-gradient(circle at 30% 30%, #ffffff, #e0e0e0); color: #333; }
    .ball-green { background: radial-gradient(circle at 30% 30%, #81c784, #388e3c); color: white; }
    .ball-blue  { background: radial-gradient(circle at 30% 30%, #64b5f6, #1976d2); color: white; }
    .ball-yellow{ background: radial-gradient(circle at 30% 30%, #fff176, #fbc02d); color: #333; }
    .ball-red   { background: radial-gradient(circle at 30% 30%, #e57373, #d32f2f); color: white; }
    .ball-gold  { background: radial-gradient(circle at 30% 30%, #ffd54f, #ffa000); color: white; border: 2px solid #fff; }
    
    .stCard {
        background: rgba(255, 255, 255, 0.8); padding: 15px;
        border-radius: 12px; border: 1px solid #c8e6c9; margin-bottom: 10px;
        text-align: center;
    }
    .zone-label { font-size: 12px; color: #888; margin-bottom: 2px; }
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)

# --- 3. 資料結構 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# 自動解壓
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 2:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except: pass

# 遊戲設定 (更新威力彩設定)
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "db_file": os.path.join(DATA_DIR, "db_539.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_539.csv"),
        "num_count": 5, "num_range": (1, 39), 
        "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "db_file": os.path.join(DATA_DIR, "db_lotto649.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_lotto649.csv"),
        "num_count": 6, "num_range": (1, 49), 
        "has_special": True, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "db_file": os.path.join(DATA_DIR, "db_super.csv"),
        "pred_file": os.path.join(DATA_DIR, "pred_super.csv"),
        "num_count": 6, "num_range": (1, 38), 
        "has_special": True, 
        "special_is_zone2": True, # 標記這是第二區
        "special_range": (1, 8),  # 第二區範圍
        "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    }
}

# --- 4. 核心讀取與爬蟲 ---

def get_ball_html(num, count, is_special=False):
    if is_special:
        return f'<div class="special-ball">{num:02d}</div>'
        
    if count >= 6: color_class = "ball-gold"
    elif count == 5: color_class = "ball-red"
    elif count == 4: color_class = "ball-yellow"
    elif count == 3: color_class = "ball-blue"
    elif count == 2: color_class = "ball-green"
    else: color_class = "ball-white"
    return f'<div class="lottery-ball {color_class}">{num:02
