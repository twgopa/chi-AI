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
    page_title="台彩數據中心 v25.1", 
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
    /* 第一區號碼球 */
    .lottery-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold;
        margin: 3px; box-shadow: inset -3px -3px 8px rgba(0,0,0,0.2), 1px 1px 3px rgba(0,0,0,0.2);
        border: 1px solid #bdc3c7; font-family: Arial, sans-serif;
    }
    /* 第二區專用球 (紅球) */
    .special-ball {
        display: inline-block; width: 38px; height: 38px; line-height: 38px;
        border-radius: 50%; text-align: center; font-weight: bold; color: white;
        margin: 3px; margin-left: 15px; /* 與第一區隔開 */
        background: radial-gradient(circle at 30% 30%, #ff5252, #b71c1c);
        box-shadow: 0 0 8px rgba(255, 82, 82, 0.6);
        border: 2px solid #ffcdd2; font-family: Arial, sans-serif;
    }
    /* 顏色等級 */
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

# 遊戲設定
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

# --- 4. 核心讀取與 HTML 生成 (修復語法錯誤) ---

def get_ball_html(num, count, is_special=False):
    # 使用 f-string 時避免換行，或是改用 format
    if is_special:
        return f'<div class="special-ball">{num:02d}</div>'
        
    color_class = "ball-white"
    if count >= 6: color_class = "ball-gold"
    elif count == 5: color_class = "ball-red"
    elif count == 4: color_class = "ball-yellow"
    elif count == 3: color_class = "ball-blue"
    elif count == 2: color_class = "ball-green"
    
    return f'<div class="lottery-ball {color_class}">{num:02d}</div>'

def render_prediction_row(nums, counts, special_num=None):
    # 使用列表來構建 HTML，最後再 join，避免 f-string 過長出錯
    html_parts = ['<div style="display:flex; justify-content:center; align-items:center; flex-wrap:wrap;">']
    
    # 第一區
    for n in nums:
        c = counts.get(n, 1)
        html_parts.append(get_ball_html(n, c))
    
    # 第二區 (如果有)
    if special_num is not None:
        html_parts.append(get_ball_html(special_num, 1, is_special=True))
        
    html_parts.append('</div>')
    return "".join(html_parts)

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
            if d_str < "2025-01-01": continue
            
            clean = line.replace(match.group(0), "")
            all_n = [int(n) for n in re.findall(r'\b\d{1,2}\b', clean)]
            valid_n, sp_n = [], []
            
            if game_name == "今彩539": 
                valid_n = sorted([n for n in all_n if 1<=n<=39])[:5]
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
        if 's' in cand and cand['s'] is not None:
            nums_str += f" + {cand['s']}"
            
        row = {
            "Timestamp": timestamp, 
            "Game": game_name, 
            "Set_ID": cand['type'], 
            "Numbers": nums_str, 
            "Error": f"{cand['e']:.4f}"
        }
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

# --- 5. 運算邏輯 (含雙區獨立運算) ---

def calculate_weights(df, cfg, target_cols, num_range, mode="balanced"):
    """通用權重計算"""
    mn, mx = num_range
    all_range = list(range(mn, mx+1))
    
    df_hist = df.iloc[:-1]
    freq_hist = pd.Series(df_hist[target_cols].values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
    prob_hist = freq_hist / freq_hist.sum()

    df_recent = df.tail(30)
    freq_recent = pd.Series(df_recent[target_cols].values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
    prob_recent = (freq_recent + 0.1) / (freq_recent.sum() + 1)
    
    last_draw = df.iloc[-1][target_cols].values
    drag_counts = pd.Series(0.0, index=all_range)
    data_matrix = df_hist[target_cols].values
    
    for target_num in last_draw:
        indices = np.where(data_matrix[:-1] == target_num)[0]
        next_indices = indices + 1
        valid_idx = next_indices[next_indices < len(data_matrix)]
        if len(valid_idx) > 0:
            next_draws = data_matrix[valid_idx]
            counts = pd.Series(next_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
            drag_counts = drag_counts.add(counts, fill_value=0)
    
    prob_banlu = drag_counts / drag_counts.sum() if drag_counts.sum() > 0 else pd.Series(1/len(all_range), index=all_range)

    if mode == "trend": final = (prob_recent * 0.8) + (prob_hist * 0.2)
    elif mode == "banlu": final = (prob_recent * 0.3) + (prob_hist * 0.7)
    else: final = (prob_recent * 0.5) + (prob_hist * 0.5)
        
    return final / final.sum()

def find_sniper_strategy(df, cfg, search_depth=60):
    # 簡化版回測
    if len(df) < search_depth + 10: return []
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    
    valid_strategies = []
    i = len(df) - 1
    # 確保資料足夠
    if i < 0: return []
    
    # 轉數值
    try:
        target_vals = df.iloc[i][num_cols].astype(int).values
        target_draw = set(target_vals)
    except: return []
    
    train_df = df.iloc[:i]
    train_df_nums = train_df[num_cols].apply(pd.to_numeric)
    avg_std = train_df_nums.std(axis=1).mean()
    
    probs = calculate_weights(train_df, cfg, num_cols, cfg["num_range"], "trend")
    p_vals = probs.values
    numbers = probs.index.tolist()
    
    hit_max = 0
    for _ in range(50):
        sel = sorted(np.random.choice(numbers, cfg["num_count"], replace=False, p=p_vals))
        curr_std = np.std(sel, ddof=1)
        if abs(curr_std - avg_std) <= 0.15:
            hits = len(set(sel).intersection(target_draw))
            if hits > hit_max: hit_max = hits
            
    if hit_max >= 4:
        valid_strategies.append({"date": df.iloc[i]['Date'], "hits": hit_max, "mode": "trend", "tol": 0.15})
    
    return valid_strategies

# --- 6. 介面主程式 ---

with st.sidebar:
    st.title("🎛️ 總控中心 v25.1")
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

st.title(f"🎯 {selected_game} 雙區特化系統")

if df.empty:
    st.error("⚠️ 資料庫空白，請先匯入檔案。")
else:
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    avg_std = df_nums.std(axis=1).mean()
    
    # 威力彩/大樂透 專屬：第二區分析
    has_zone2 = cfg.get("special_is_zone2", False)
    zone2_col = "Zw" if has_zone2 else ("SP" if cfg["has_special"] else None)
    
    snipers = find_sniper_strategy(df, cfg, search_depth=50)
    sniper_mode = "balanced"
    if snipers:
        best_s = snipers[0]
        sniper_mode = best_s['mode']
        st.success(f"🎯 **狙擊參數**：[{best_s['mode'].upper()}] 策略曾在 {best_s['date']} 命中 {best_s['hits']} 星！")

    tab1, tab2, tab3 = st.tabs(["🎲 彩球預測", "📜 預測紀錄", "📂 歷史資料"])

    with tab1:
        c1, c2 = st.columns(2)
        tol = c1.slider("第一區誤差", 0.01, 0.5, 0.15, 0.01)
        repeater = c2.checkbox("第一區連莊", value=True)

        if st.button("🎲 啟動雙軌運算", type="primary"):
            candidates = []
            
            templates = [
                {"name": f"狙擊 ({sniper_mode})", "mode": sniper_mode},
                {"name": "順勢 (Trend)", "mode": "trend"},
                {"name": "版路 (Drag)", "mode": "banlu"}
            ]
            
            # 計算第二區權重 (如果有的話)
            sp_probs = None
            if zone2_col:
                sp_range = cfg.get("special_range", cfg["num_range"])
                sp_probs = calculate_weights(df, cfg, [zone2_col], sp_range, "trend")
            
            bar = st.progress(0)
            
            # 產生 6 組預測 (3種模式 x 2)
            for i, temp in enumerate(templates * 2):
                probs = calculate_weights(df, cfg, [c for c in cfg["cols"] if c.startswith("N")], cfg["num_range"], temp["mode"])
                numbers = probs.index.tolist()
                p_vals = probs.values
                
                att = 0
                found = False
                while not found and att < 5000:
                    sel = sorted(np.random.choice(numbers, cfg["num_count"], replace=False, p=p_vals))
                    if repeater:
                        last_draw = df_nums.iloc[-1].values
                        rep_num = np.random.choice(last_draw)
                        if rep_num not in sel:
                            sel[0] = rep_num
                            sel.sort()
                    
                    curr_std = np.std(sel, ddof=1)
                    if abs(curr_std - avg_std) <= tol:
                        sp_num = None
                        if sp_probs is not None:
                            sp_num = np.random.choice(sp_probs.index.tolist(), p=sp_probs.values)
                        
                        candidates.append({'n': sel, 'e': abs(curr_std - avg_std), 'type': temp["name"], 's': sp_num})
                        found = True
                    att += 1
                bar.progress((i + 1) / 6)
            bar.empty()
            
            st.session_state[f'last_candidates_{selected_game}'] = candidates

        # 顯示結果
        sess_key = f'last_candidates_{selected_game}'
        if sess_key in st.session_state and st.session_state[sess_key]:
            results = st.session_state[sess_key]
            
            # 統計熱度
            all_nums = []
            for res in results: all_nums.extend(res['n'])
            counter = collections.Counter(all_nums)
            
            st.markdown("### 🎨 預測結果 (紅球為第二區)")
            
            cols = st.columns(3)
            for i, res in enumerate(results):
                with cols[i % 3]:
                    html = f'<div class="stCard"><h5>{res["type"]}</h5>'
                    html += '<div class="zone-label">第一區</div>'
                    html += render_prediction_row(res['n'], counter, None)
                    
                    if res['s'] is not None:
                        html += '<div class="zone-label" style="margin-top:8px; color:#d32f2f;">第二區</div>'
                        html += render_prediction_row([], {}, res['s'])
                        
                    html += f'<div style="font-size:12px;color:#888;margin-top:5px">結構誤差: {res["e"]:.4f}</div></div>'
                    st.markdown(html, unsafe_allow_html=True)
            
            st.divider()
            if st.button("💾 存入獨立紀錄"):
                save_prediction(selected_game, results)
                st.success(f"已存入 {cfg['pred_file']}")

    with tab2:
        df_pred = load_predictions(selected_game)
        if not df_pred.empty:
            df_pred = df_pred.sort_index(ascending=False)
            st.dataframe(df_pred, use_container_width=True)
            if st.button("🗑️ 清空此遊戲紀錄"):
                if os.path.exists(cfg["pred_file"]): os.remove(cfg["pred_file"])
                st.rerun()
        else: st.info("無紀錄")

    with tab3:
        st.dataframe(df, use_container_width=True)
