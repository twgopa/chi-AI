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
    page_title="台彩數據中心 v24.1", 
    page_icon="🕰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. CSS 視覺美化 ---
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
                    temp_rows = []
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
        filename = f"Daily_Patch_{game_name}.csv"
        path = os.path.join(DATA_DIR, filename)
        pd.DataFrame(new_rows, columns=cfg["cols"]).to_csv(path, index=False)
        return len(new_rows)
    return 0

def save_prediction(game_name, candidates):
    cfg = GAME_CONFIG[game_name]
    file_path = cfg["pred_file"]
    log_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, cand in enumerate(candidates):
        nums_str = ",".join([str(n) for n in cand['n']])
        row = {"Timestamp": timestamp, "Game": game_name, "Set_ID": cand['type'], "Numbers": nums_str, "Error": f"{cand['e']:.4f}"}
        log_data.append(row)
    df_log = pd.DataFrame(log_data)
    if os.path.exists(file_path):
        try: old_log = pd.read_csv(file_path); df_final = pd.concat([old_log, df_log], ignore_index=True)
        except: df_final = df_log
    else: df_final = df_log
    df_final.to_csv(file_path, index=False, encoding='utf-8-sig')

def load_predictions(game_name):
    cfg = GAME_CONFIG[game_name]
    if os.path.exists(cfg["pred_file"]):
        try: return pd.read_csv(cfg["pred_file"])
        except: return pd.DataFrame()
    return pd.DataFrame()

# --- 5. 新增：歷史鏡像搜尋 ---

def find_historical_mirror(df, cfg):
    if len(df) < 10: return False, None, None
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    last_draw = set(df_nums.iloc[-1].values)
    matches = []
    for i in range(len(df)-2):
        hist_draw = set(df_nums.iloc[i].values)
        if hist_draw == last_draw:
            match_date = df.iloc[i]['Date']
            next_draw_nums = sorted(df_nums.iloc[i+1].values.tolist())
            matches.append( (match_date, next_draw_nums) )
    if matches:
        return True, matches[-1][0], matches[-1][1]
    return False, None, None

def calculate_weights(df, cfg, mode="balanced"):
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    mn, mx = cfg["num_range"]
    all_range = list(range(mn, mx+1))
    df_hist = df.iloc[:-1]
    freq_hist = pd.Series(df_hist[num_cols].values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
    prob_hist = freq_hist / freq_hist.sum()
    df_recent = df.tail(30)
    freq_recent = pd.Series(df_recent[num_cols].values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
    prob_recent = (freq_recent + 0.1) / (freq_recent.sum() + 1)
    last_draw = df.iloc[-1][num_cols].values
    drag_counts = pd.Series(0.0, index=all_range)
    data_matrix = df_hist[num_cols].values
    for target_num in last_draw:
        indices = np.where(data_matrix[:-1] == target_num)[0]
        next_indices = indices + 1
        valid_idx = next_indices[next_indices < len(data_matrix)]
        if len(valid_idx) > 0:
            next_draws = data_matrix[valid_idx]
            counts = pd.Series(next_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
            drag_counts = drag_counts.add(counts, fill_value=0)
    prob_banlu = drag_counts / drag_counts.sum() if drag_counts.sum() > 0 else pd.Series(1/len(all_range), index=all_range)
    if mode == "trend": final = (prob_recent * 0.7) + (prob_hist * 0.2) + (prob_banlu * 0.1)
    elif mode == "banlu": final = (prob_banlu * 0.6) + (prob_recent * 0.3) + (prob_hist * 0.1)
    else: final = (prob_recent * 0.4) + (prob_hist * 0.3) + (prob_banlu * 0.3)
    return final / final.sum()

def find_sniper_strategy(df, cfg, search_depth=60):
    if len(df) < search_depth + 10: return []
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    strategies = [("trend", 0.12), ("trend", 0.15), ("banlu", 0.12), ("banlu", 0.15), ("balanced", 0.12)]
    valid_strategies = []
    for i in range(len(df)-1, len(df)-search_depth-1, -1):
        target_draw = set(df_nums.iloc[i].values)
        train_df = df.iloc[:i]
        if len(train_df) < 50: continue
        avg_std = train_df[num_cols].std(axis=1).mean()
        for mode, tol in strategies:
            probs = calculate_weights(train_df, cfg, mode)
            numbers = probs.index.tolist()
            p_vals = probs.values
            hit_max = 0
            for _ in range(100):
                sel = sorted(np.random.choice(numbers, cfg["num_count"], replace=False, p=p_vals))
                curr_std = np.std(sel, ddof=1)
                if abs(curr_std - avg_std) <= tol:
                    hits = len(set(sel).intersection(target_draw))
                    if hits > hit_max: hit_max = hits
            if hit_max >= 4:
                valid_strategies.append({"date": df.iloc[i]['Date'], "hits": hit_max, "mode": mode, "tol": tol})
                break
    return valid_strategies

# --- 7. 介面主程式 ---

with st.sidebar:
    st.title("🎛️ 總控中心 v24.1")
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), index=0)
    now = datetime.now()
    if now.hour >= 17: st.info(f"🔔 17:00 後請按下方補單")
    st.markdown("---")
    uploaded_files = st.file_uploader("上傳資料 (CSV/ZIP)", accept_multiple_files=True, type=['csv', 'zip'])
    if uploaded_files:
        if st.button("📥 匯入資料"):
            with st.spinner("處理中..."):
                process_bulk_files(uploaded_files)
                load_all_data.clear()
                st.success("匯入完成！")
                time.sleep(1)
                st.rerun()
    st.markdown("---")
    if st.button(f"🚀 每日補單 ({selected_game})"):
        with st.spinner("連線中..."):
            c = crawl_daily_web(selected_game)
            load_all_data.clear()
            if c>0: st.success(f"更新 {c} 筆！")
            else: st.info("無新資料")

cfg = GAME_CONFIG[selected_game]
df = load_all_data(selected_game)

st.title(f"🔮 {selected_game} 鏡像預測系統")

if df.empty:
    st.error("⚠️ 資料庫空白，請先匯入檔案。")
else:
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    avg_std = df_nums.std(axis=1).mean()
    
    with st.spinner("正在進行 18 年歷史大數據比對..."):
        has_mirror, mirror_date, mirror_next_nums = find_historical_mirror(df, cfg)
    
    if has_mirror:
        st.markdown(f"""
        <div class="mirror-alert">
            <h3>⚡ 驚人發現！歷史重現！</h3>
            <p>本期開出的號碼，與 <b>{mirror_date}</b> 的開獎結果 <b>完全相同</b>！<br>
            根據歷史紀錄，當年的下一期開出了以下號碼，具有極高的參考價值：
