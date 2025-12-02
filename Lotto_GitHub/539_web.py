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
import collections

# --- 1. 系統設定 ---
st.set_page_config(
    page_title="台彩數據中心 v27.0 (絕對歸檔版)", 
    page_icon="🗄️", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 預測紀錄檔
LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# --- 3. 遊戲設定 (只保留三大天王) ---
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "db_file": os.path.join(DATA_DIR, "db_539.csv"),
        "num_count": 5, "num_range": (1, 39), "has_special": False, 
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "db_file": os.path.join(DATA_DIR, "db_lotto649.csv"),
        "num_count": 6, "num_range": (1, 49), "has_special": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "db_file": os.path.join(DATA_DIR, "db_super.csv"),
        "num_count": 6, "num_range": (1, 38), "has_special": True, "special_is_zone2": True, "special_range": (1, 8),
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    }
}

# --- 4. 核心功能：強力解析與歸檔 ---

def parse_date_strict(date_val):
    """強力日期解析：支援 民國年/西元年/斜線/橫線"""
    d_str = str(date_val).strip()
    # 1. 嘗試標準 YYYY-MM-DD
    try:
        return pd.to_datetime(d_str).strftime('%Y-%m-%d')
    except:
        pass
    
    # 2. 嘗試民國年 (e.g. 112/01/01, 96/1/1)
    # 抓取 2-3位年, 1-2位月, 1-2位日
    match = re.match(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', d_str)
    if match:
        y = int(match.group(1))
        m = int(match.group(2))
        d = int(match.group(3))
        # 轉西元
        if y < 1911: 
            y += 1911
        return f"{y}-{m:02d}-{d:02d}"
    
    return None

def detect_game_type(filename, df_columns, df_first_row):
    """嚴格判斷遊戲類型"""
    filename = filename.lower()
    
    # A. 內容判斷 (最準)
    if '遊戲名稱' in df_columns:
        game_name = str(df_first_row['遊戲名稱'])
        if "539" in game_name: return "今彩539"
        if "大樂透" in game_name: return "大樂透"
        if "威力彩" in game_name: return "威力彩"
    
    # B. 檔名判斷
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
            
    return None

def process_and_merge_files(uploaded_files, progress_bar):
    """
    核心引擎：讀取 -> 解析 -> 立即合併至主資料庫
    """
    # 用來暫存讀到的所有資料
    new_data_buffer = {g: [] for g in GAME_CONFIG.keys()}
    
    total_files = len(uploaded_files)
    
    # 1. 遍歷上傳的檔案
    for i, file_obj in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / total_files, text=f"正在解析: {file_obj.name}")
        
        # 處理 ZIP
        if file_obj.name.endswith('.zip'):
            with zipfile.ZipFile(file_obj, 'r') as z:
                z.extractall(DATA_DIR)
            continue # ZIP 解壓後，下次掃描會讀到 CSV，這裡先跳過

        # 處理 CSV
        try:
            # 多編碼嘗試
            try: df = pd.read_csv(file_obj, encoding='cp950', on_bad_lines='skip')
            except: 
                try: df = pd.read_csv(file_obj, encoding='big5', on_bad_lines='skip')
                except: 
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding='utf-8', on_bad_lines='skip')
            
            # 清理欄位名稱 (去除空格)
            df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
            
            if df.empty: continue

            # 判斷類型
            game_type = detect_game_type(file_obj.name, df.columns, df.iloc[0])
            
            # 只處理三大彩種，其他跳過
            if not game_type: continue
            
            cfg = GAME_CONFIG[game_type]
            
            # 開始解析每一行
            if '開獎日期' in df.columns:
                for _, row in df.iterrows():
                    try:
                        # 日期
                        d_str = parse_date_strict(row['開獎日期'])
                        if not d_str: continue
                        
                        # 號碼
                        nums = []
                        for k in range(1, cfg["num_count"] + 1):
                            col_name = f'獎號{k}'
                            if col_name in df.columns:
                                nums.append(int(row[col_name]))
                        
                        if len(nums) != cfg["num_count"]: continue
                        
                        # 特別號 / 第二區
                        sp = []
                        if cfg["has_special"]:
                            if "第二區" in df.columns: sp = [int(row['第二區'])]
                            elif "特別號" in df.columns: sp = [int(row['特別號'])]
                            else: sp = [0]
                        
                        # 排序主號碼
                        nums.sort()
                        
                        # 組合 (Date, N1...N5/6, SP, Source)
                        entry = [d_str] + nums + sp + ["History_Import"]
                        
                        if len(entry) == len(cfg["cols"]):
                            new_data_buffer[game_type].append(entry)
                            
                    except: continue
                    
        except Exception as e:
            print(f"Error reading {file_obj.name}: {e}")
            continue

    # 2. 立即合併至主資料庫
    updated_counts = {}
    
    for game, rows in new_data_buffer.items():
        if not rows:
            updated_counts[game] = 0
            continue
            
        cfg = GAME_CONFIG[game]
        db_path = cfg["db_file"]
        
        # 載入現有 DB
        if os.path.exists(db_path):
            try:
                old_df = pd.read_csv(db_path)
            except:
                old_df = pd.DataFrame(columns=cfg["cols"])
        else:
            old_df = pd.DataFrame(columns=cfg["cols"])
            
        # 轉成 DF
        new_df = pd.DataFrame(rows, columns=cfg["cols"])
        
        # 合併
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        
        # 關鍵：去重 (以日期為準，保留最後一筆) 與 排序
        final_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        final_df.sort_values(by='Date', ascending=True, inplace=True)
        
        # 存檔
        final_df.to_csv(db_path, index=False)
        updated_counts[game] = len(final_df)
        
    return updated_counts

# --- 5. 讀取 DB (顯示用) ---
def load_db(game_name):
    cfg = GAME_CONFIG[game_name]
    if os.path.exists(cfg["db_file"]):
        return pd.read_csv(cfg["db_file"])
    return pd.DataFrame()

# --- 6. 爬蟲補單 ---
def crawl_daily(game_name):
    cfg = GAME_CONFIG[game_name]
    url = "https://i539.tw/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    new_rows = []
    try:
        res = requests.get(url, headers=headers, verify=False, timeout=5)
        res.encoding = 'utf-8'
        lines = res.text.split('\n')
        for line in lines:
            if len(line)<10: continue
            match = re.search(r'(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})', line)
            if not match: continue
            d_str = f"{match.group(1)}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)}"
            if d_str < "2025-01-01": continue
            
            clean = line.replace(match.group(0), "")
            all_n = [int(n) for n in re.findall(r'\b\d{1,2}\b', clean)]
            valid_n, sp_n = [], []
            
            if game_name == "今彩539": valid_n = sorted([n for n in all_n if 1<=n<=39])[:5]
            elif game_name == "大樂透":
                if len(all_n)>=7: 
                    valid_n = sorted([n for n in all_n if 1<=n<=49][:6])
                    sp_n = [all_n[6]] if 1<=all_n[6]<=49 else [0]
            elif game_name == "威力彩":
                 if len(all_n)>=7:
                     valid_n = sorted([n for n in all_n[:6] if 1<=n<=38])
                     sp_n = [all_n[6]] if 1<=all_n[6]<=8 else [1]

            if len(valid_n) == cfg["num_count"]:
                entry = [d_str] + valid_n + sp_n + ["Web_Crawl"]
                if len(entry) == len(cfg["cols"]): new_rows.append(entry)
    except: pass
    
    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=cfg["cols"])
        if os.path.exists(cfg["db_file"]):
            df_old = pd.read_csv(cfg["db_file"])
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else: df_final = df_new
        
        df_final.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        df_final.sort_values(by='Date', ascending=True, inplace=True)
        df_final.to_csv(cfg["db_file"], index=False)
        return len(new_rows)
    return 0

# --- 7. 預測與視覺化 ---
def get_ball_html(num, count, is_special=False):
    if is_special: return f'<div class="special-ball">{num:02d}</div>'
    color = "ball-white"
    if count >= 6: color = "ball-gold"
    elif count == 5: color = "ball-red"
    elif count == 4: color = "ball-yellow"
    elif count == 3: color = "ball-blue"
    elif count == 2: color = "ball-green"
    return f'<div class="lottery-ball {color}">{num:02d}</div>'

def render_row(nums, counts, sp=None):
    html = '<div style="display:flex;justify-content:center;flex-wrap:wrap;">'
    for n in nums: html += get_ball_html(n, counts.get(n, 1))
    if sp is not None: html += get_ball_html(sp, 1, True)
    html += '</div>'
    return html

def calc_weights(df, cfg, cols, rng, mode="balanced"):
    mn, mx = rng
    all_range = list(range(mn, mx+1))
    df_hist = df.iloc[:-1]
    freq = pd.Series(df_hist[cols].values.flatten()).value_counts().reindex(all_range, fill_value=0)
    prob_hist = freq / freq.sum()
    
    df_rec = df.tail(30)
    freq_rec = pd.Series(df_rec[cols].values.flatten()).value_counts().reindex(all_range, fill_value=0)
    prob_rec = (freq_rec + 0.1) / (freq_rec.sum() + 1)
    
    # 版路
    last_draw = df.iloc[-1][cols].values
    drag = pd.Series(0.0, index=all_range)
    mat = df_hist[cols].values
    for t in last_draw:
        idxs = np.where(mat[:-1] == t)[0]
        n_idxs = idxs + 1
        if len(n_idxs) > 0:
            n_draws = mat[n_idxs]
            c = pd.Series(n_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
            drag = drag.add(c, fill_value=0)
    prob_banlu = drag / drag.sum() if drag.sum() > 0 else pd.Series(1/len(all_range), index=all_range)
    
    if mode == "trend": final = prob_rec * 0.8 + prob_hist * 0.2
    elif mode == "banlu": final = prob_banlu * 0.7 + prob_rec * 0.3
    else: final = prob_rec * 0.4 + prob_hist * 0.3 + prob_banlu * 0.3
    return final / final.sum()

# --- 8. 介面 ---
with st.sidebar:
    st.title("🗄️ 絕對歸檔中心")
    game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()))
    
    st.markdown("---")
    st.subheader("📂 1. 匯入歷年資料 (2007-2025)")
    st.info("請將所有 CSV 全選拖入下方 (系統會自動過濾非相關檔案)")
    uploaded_files = st.file_uploader("CSV/ZIP", accept_multiple_files=True, type=['csv', 'zip'])
    
    if uploaded_files:
        if st.button("📥 強制合併歸檔"):
            bar = st.progress(0)
            stats = process_and_merge_files(uploaded_files, bar)
            bar.empty()
            st.success("歸檔完成！")
            for g, c in stats.items():
                st.write(f"- **{g}**: 目前共 {c} 筆")
            time.sleep(2)
            st.rerun()

    st.markdown("---")
    if st.button(f"🚀 2. 每日補單 ({game})"):
        with st.spinner("連線中..."):
            c = crawl_daily_web(game)
            if c>0: st.success(f"更新 {c} 筆")
            else: st.info("無新資料")
            st.rerun()

# 主畫面
cfg = GAME_CONFIG[game]
df = load_db(game)

# CSS
st.markdown("""
<style>
.stApp { background-color: #f0f7f4; background-image: url("https://www.transparenttextures.com/patterns/rice-paper-2.png"); }
</style>
""", unsafe_allow_html=True)

st.title(f"🎯 {game} 操盤室")

if df.empty:
    st.error(f"⚠️ {game} 資料庫為空。")
    st.warning("請在左側 **步驟 1** 匯入歷年 CSV 檔案。")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("總期數", len(df))
    c2.metric("最早日期", df.iloc[0]['Date'])
    c3.metric("最新日期", df.iloc[-1]['Date'])
    
    tab1, tab2 = st.tabs(["🔮 預測", "📋 數據"])
    
    with tab1:
        c_l, c_r = st.columns(2)
        tol = c_l.slider("誤差值", 0.01, 0.5, 0.15, 0.01)
        repeater = c_r.checkbox("連莊", value=True)
        
        if st.button("🎲 運算", type="primary"):
            cands = []
            # 預測參數
            num_cols = [c for c in cfg["cols"] if c.startswith("N")]
            df_n = df[num_cols].apply(pd.to_numeric)
            avg_std = df_n.std(axis=1).mean()
            
            # 第二區
            sp_probs = None
            has_z2 = cfg.get("special_is_zone2", False)
            z2_col = "Zw" if has_z2 else ("SP" if cfg["has_special"] else None)
            if z2_col:
                sp_range = cfg.get("special_range", cfg["num_range"])
                sp_probs = calc_weights(df, cfg, [z2_col], sp_range, "trend")
            
            temps = ["balanced", "trend", "banlu"]
            bar = st.progress(0)
            
            for idx, mode in enumerate(temps * 2):
                probs = calc_weights(df, cfg, num_cols, cfg["num_range"], mode)
                nums = probs.index.tolist()
                p_vals = probs.values
                
                found = False
                att = 0
                while not found and att < 5000:
                    sel = sorted(np.random.choice(nums, cfg["num_count"], replace=False, p=p_vals))
                    if repeater:
                        last = df_n.iloc[-1].values
                        r_n = np.random.choice(last)
                        if r_n not in sel: sel[0] = r_n; sel.sort()
                    
                    curr_std = np.std(sel, ddof=1)
                    if abs(curr_std - avg_std) <= tol:
                        sp = None
                        if sp_probs is not None: sp = np.random.choice(sp_probs.index.tolist(), p=sp_probs.values)
                        cands.append({'n': sel, 's': sp, 'e': abs(curr_std - avg_std), 't': mode})
                        found = True
                    att += 1
                bar.progress((idx+1)/6)
            bar.empty()
            st.session_state['res'] = cands
            
        if 'res' in st.session_state:
            res_list = st.session_state['res']
            all_n = []
            for r in res_list: all_n.extend(r['n'])
            ctr = collections.Counter(all_n)
            
            cols = st.columns(3)
            for i, r in enumerate(res_list):
                with cols[i%3]:
                    html = f'<div class="stCard"><b>{r["t"]}</b>'
                    html += '<div class="zone-label">第一區</div>'
                    html += render_row(r['n'], ctr, None)
                    if r['s']:
                        html += '<div class="zone-label" style="color:red">第二區</div>'
                        html += render_row([], {}, r['s'])
                    html += f'<small>誤差: {r["e"]:.4f}</small></div>'
                    st.markdown(html, unsafe_allow_html=True)

    with tab2:
        st.dataframe(df.sort_values(by='Date', ascending=False), use_container_width=True)
