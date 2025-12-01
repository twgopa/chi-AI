import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import re
import urllib3
from datetime import datetime
import glob
import time
import zipfile
import altair as alt

# --- 1. 系統設定 ---
st.set_page_config(page_title="台彩數據中心 v18.0 (全能資料庫)", page_icon="🏢", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 自動解壓 ZIP (如果有)
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
for z_file in zip_files:
    try:
        if zipfile.is_zipfile(z_file):
            # 檢查 data 內是否已有 csv，若太少則解壓
            if len(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)) < 5:
                with zipfile.ZipFile(z_file, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
    except: pass

LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# --- 3. 遊戲設定 (擴充至全彩種) ---
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
    "3星彩": {
        "keywords": ["3星彩", "3 Star"],
        "db_file": "db_3star.csv",
        "num_count": 3, "num_range": (0, 9), "has_special": False, "enable_predict": False,
        "cols": ["Date", "D1", "D2", "D3", "Source"]
    },
    "4星彩": {
        "keywords": ["4星彩", "4 Star"],
        "db_file": "db_4star.csv",
        "num_count": 4, "num_range": (0, 9), "has_special": False, "enable_predict": False,
        "cols": ["Date", "D1", "D2", "D3", "D4", "Source"]
    },
    "39樂合彩": {
        "keywords": ["39樂合彩"],
        "db_file": "db_39lotto.csv",
        "num_count": 5, "num_range": (1, 39), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "49樂合彩": {
        "keywords": ["49樂合彩"],
        "db_file": "db_49lotto.csv",
        "num_count": 6, "num_range": (1, 49), "has_special": False, "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Source"]
    },
    "賓果賓果": {
        "keywords": ["賓果賓果", "Bingo"],
        "db_file": "db_bingo.csv",
        "num_count": 20, "num_range": (1, 80), "has_special": True, "enable_predict": False, # 賓果還有超級獎號
        "cols": ["Date", "Period"] + [f"N{i}" for i in range(1, 21)] + ["Super", "Source"]
    },
    "大樂透加開": {
        "keywords": ["加開獎項", "Big Red"],
        "db_file": "db_lotto649_extra.csv",
        "num_count": 6, "num_range": (1, 49), "has_special": False, "enable_predict": False,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Source"] # 簡化版
    }
}

# --- 4. 智慧分類與讀取引擎 ---

def detect_game_type(filename, df_head):
    """判斷檔案屬於哪種遊戲"""
    filename = filename.lower()
    # 1. 檔名優先
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    # 2. 內容判斷
    if '遊戲名稱' in df_head.columns:
        val = str(df_head.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in val: return game
    return None

def process_bulk_files(uploaded_files, progress_bar):
    """處理上傳的檔案，自動歸檔"""
    results = {g: 0 for g in GAME_CONFIG.keys()}
    temp_storage = {g: [] for g in GAME_CONFIG.keys()}
    
    total = len(uploaded_files)
    for i, up_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / total, text=f"處理中: {up_file.name}")
        
        try:
            # 讀取 CSV
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
            
            # 針對賓果做特殊處理 (欄位多)
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
                    
            # 一般彩種處理
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
                            else: sp = [0] # 防呆
                            
                        if cfg["enable_predict"]: nums.sort()
                        
                        entry = [d_str] + nums + sp + ["UserUpload"]
                        if len(entry) == len(cfg["cols"]): temp_storage[game_type].append(entry)
                    except: continue
        except: continue

    # 存檔
    for game, rows in temp_storage.items():
        if rows:
            cfg = GAME_CONFIG[game]
            # 存成獨立檔案，避免跟爬蟲檔打架
            new_filename = f"Upload_{game}_{int(time.time())}.csv"
            pd.DataFrame(rows, columns=cfg["cols"]).to_csv(os.path.join(DATA_DIR, new_filename), index=False)
            results[game] += len(rows)
            
    return results

@st.cache_data(show_spinner=False, ttl=60)
def load_all_data(game_name):
    """讀取該遊戲的所有資料"""
    if game_name not in GAME_CONFIG: return pd.DataFrame()
    cfg = GAME_CONFIG[game_name]
    
    # 遞迴搜尋所有 CSV
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    target_files = [f for f in all_files if "prediction_log.csv" not in f]
    
    merged_data = []
    
    for file_path in target_files:
        filename = os.path.basename(file_path)
        # 關鍵字篩選
        if any(k in filename for k in cfg["keywords"]):
            # 若選的是賓果，就只讀賓果；若選其他，排除賓果以省資源
            if game_name != "賓果賓果" and "賓果" in filename: continue
            if game_name == "賓果賓果" and "賓果" not in filename: continue

            try:
                try: df = pd.read_csv(file_path, encoding='cp950', on_bad_lines='skip')
                except: 
                    try: df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    except: continue
                
                df.columns = [str(c).strip() for c in df.columns]
                
                # A. 官方格式 CSV (含中文)
                if '開獎日期' in df.columns:
                    # 使用前面 process_bulk_files 類似邏輯解析，這裡簡化直接讀取已轉好的 Upload 檔優先
                    # 如果使用者直接放官方原始檔在 data 裡，這裡即時解析
                    for _, row in df.iterrows():
                        try:
                            d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                            # ... (省略重複解析邏輯，建議使用者透過匯入功能轉成標準格式)
                            # 這裡僅支援標準格式讀取，若為原始檔建議先匯入
                            pass 
                        except: continue

                # B. 標準格式 (我們轉存後的)
                elif 'Date' in df.columns:
                    # 檢查欄位是否吻合
                    if len(df.columns) == len(cfg["cols"]):
                        merged_data.extend(df.values.tolist())
                    else:
                        # 欄位不合嘗試修正 (例如缺 Source)
                        valid_cols = [c for c in cfg["cols"] if c in df.columns]
                        temp_df = df[valid_cols].copy()
                        if "Source" not in temp_df.columns: temp_df["Source"] = "Auto"
                        if len(temp_df.columns) == len(cfg["cols"]):
                            merged_data.extend(temp_df.values.tolist())
            except: continue

    if merged_data:
        final_df = pd.DataFrame(merged_data, columns=cfg["cols"])
        # 賓果資料量大，去重較慢，可視情況優化
        final_df.drop_duplicates(subset=['Date'] if game_name!="賓果賓果" else ['Date', 'Period'], keep='last', inplace=True)
        final_df.sort_values(by='Date', ascending=True, inplace=True)
        return final_df
    return pd.DataFrame(columns=cfg["cols"])

# --- 5. 爬蟲更新 ---
def crawl_daily_web(game_name):
    if game_name not in ["今彩539", "大樂透", "威力彩"]: return 0
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
            if d_str < "2025-01-01": continue # 只抓今年的
            
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
        filename = f"Daily_Patch_{game_name}.csv"
        path = os.path.join(DATA_DIR, filename)
        pd.DataFrame(new_rows, columns=cfg["cols"]).to_csv(path, index=False)
        return len(new_rows)
    return 0

# --- 6. 介面 ---

with st.sidebar:
    st.title("🎛️ 全能總控台")
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), index=0)
    
    st.markdown("---")
    st.subheader("📂 批次匯入資料庫")
    st.info("將您所有的 CSV 檔 (包含賓果、3星、4星...) 全部拖進來，系統會自動分類。")
    
    uploaded_files = st.file_uploader("拖曳檔案至此", accept_multiple_files=True, type=['csv'])
    if uploaded_files:
        if st.button("📥 開始智慧歸檔"):
            bar = st.progress(0, text="啟動中...")
            res = process_bulk_files(uploaded_files, bar)
            bar.empty()
            st.success("✅ 歸檔完成！")
            for g, c in res.items():
                if c > 0: st.write(f"- {g}: +{c} 筆")
            load_all_data.clear()
            time.sleep(3)
            st.rerun()
            
    st.markdown("---")
    if selected_game in ["今彩539", "大樂透", "威力彩"]:
        if st.button(f"🚀 更新 {selected_game}"):
            with st.spinner("爬取中..."):
                c = crawl_daily_web(selected_game)
                if c>0: 
                    load_all_data.clear()
                    st.success(f"更新 {c} 筆！")
                    st.rerun()
                else: st.info("無新資料")

# 主畫面
cfg = GAME_CONFIG[selected_game]
df = load_all_data(selected_game)

st.header(f"📊 {selected_game} 資料庫")

if df.empty:
    st.warning("尚無資料。請使用左側匯入功能。")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("總筆數", len(df))
    c2.metric("起", df.iloc[0]['Date'])
    c3.metric("訖", df.iloc[-1]['Date'])
    
    tab1, tab2 = st.tabs(["📋 數據列表", "🔮 統計預測"])
    
    with tab1:
        st.dataframe(df, use_container_width=True, height=600)
        
    with tab2:
        if not cfg["enable_predict"]:
            st.info("此遊戲為數字排列型，不提供預測功能。")
        else:
            st.subheader("下期預測 (基於歷史數據)")
            if st.button("🎲 運算"):
                num_cols = [c for c in cfg["cols"] if c.startswith("N")]
                df_nums = df[num_cols].apply(pd.to_numeric)
                
                # 簡單統計權重
                vals = df_nums.values.flatten()
                freq = pd.Series(vals).value_counts().sort_index()
                mn, mx = cfg["num_range"]
                for i in range(mn, mx+1): 
                    if i not in freq: freq[i] = 0
                
                w = freq.values / freq.values.sum()
                nums = freq.index.tolist()
                
                res = []
                for _ in range(5):
                    s = sorted(np.random.choice(nums, cfg["num_count"], replace=False, p=w))
                    res.append(s)
                
                cols = st.columns(5)
                for i, (c, r) in enumerate(zip(cols, res)):
                    c.success(f"第 {i+1} 組")
                    c.code(str(r))