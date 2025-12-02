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
import altair as alt

# --- 1. 系統設定 ---
st.set_page_config(
    page_title="台彩數據中心 v28.0 (核彈級重構)", 
    page_icon="☢️", 
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
""", unsafe_allow_html=True)

# --- 3. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# --- 4. 遊戲設定 ---
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

# --- 5. 核心功能 ---

def parse_date_strict(date_val):
    """強力日期解析"""
    s = str(date_val).strip()
    # 處理 2007/1/1
    try: return pd.to_datetime(s).strftime('%Y-%m-%d')
    except: pass
    
    # 處理 96/1/1 (民國年)
    match = re.match(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', s)
    if match:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if y < 1911: y += 1911
        return f"{y}-{m:02d}-{d:02d}"
    return None

def detect_game_type(df):
    """只依賴 CSV 內容判斷"""
    if '遊戲名稱' in df.columns:
        val = str(df.iloc[0]['遊戲名稱'])
        if "539" in val: return "今彩539"
        if "大樂透" in val: return "大樂透"
        if "威力彩" in val: return "威力彩"
    return None

def rebuild_databases_nuclear():
    """核彈級重整：掃描所有 CSV 並重新建立 DB"""
    st.toast("☢️ 核彈級重整啟動！正在掃描硬碟...")
    
    # 1. 解壓所有 ZIP
    zips = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z, 'r') as zf: zf.extractall(DATA_DIR)
        except: pass

    # 2. 搜尋所有 CSV
    all_csv = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    # 排除我們自己產生的 DB 檔
    db_files = [os.path.abspath(cfg['db_file']) for cfg in GAME_CONFIG.values()]
    # 也排除 log
    skip_list = db_files + [os.path.abspath(LOG_FILE)]
    
    target_files = []
    for f in all_csv:
        if os.path.abspath(f) not in skip_list and "pred_" not in f:
            target_files.append(f)

    # 3. 讀取並分類
    storage = {g: [] for g in GAME_CONFIG.keys()}
    prog = st.progress(0)
    
    for i, fpath in enumerate(target_files):
        prog.progress((i+1)/len(target_files), text=f"解析: {os.path.basename(fpath)}")
        try:
            # 多編碼讀取
            try: df = pd.read_csv(fpath, encoding='cp950', on_bad_lines='skip')
            except:
                try: df = pd.read_csv(fpath, encoding='big5', on_bad_lines='skip')
                except: df = pd.read_csv(fpath, encoding='utf-8', on_bad_lines='skip')
            
            # 清理欄位
            df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
            
            # 判斷遊戲
            gtype = detect_game_type(df)
            if not gtype: continue
            
            cfg = GAME_CONFIG[gtype]
            
            if '開獎日期' in df.columns:
                for _, row in df.iterrows():
                    try:
                        d = parse_date_strict(row['開獎日期'])
                        if not d: continue
                        
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
                        entry = [d] + nums + sp + ["History"]
                        if len(entry) == len(cfg["cols"]): storage[gtype].append(entry)
                    except: continue
        except: continue
    
    prog.empty()
    
    # 4. 存檔
    stats = {}
    for g, rows in storage.items():
        if rows:
            cfg = GAME_CONFIG[g]
            new_df = pd.DataFrame(rows, columns=cfg["cols"])
            new_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
            new_df.sort_values(by='Date', ascending=True, inplace=True)
            new_df.to_csv(cfg["db_file"], index=False)
            stats[g] = len(new_df)
            
    return stats

@st.cache_data(show_spinner=False, ttl=60)
def load_db_data(game_name):
    cfg = GAME_CONFIG[game_name]
    if os.path.exists(cfg["db_file"]):
        try: return pd.read_csv(cfg["db_file"])
        except: return pd.DataFrame()
    return pd.DataFrame()

def crawl_daily_web(game_name):
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

# --- 6. 運算邏輯 ---

def get_ball_html(num, count, is_special=False):
    if is_special: return f'<div class="special-ball">{num:02d}</div>'
    color = "ball-white"
    if count >= 6: color = "ball-gold"
    elif count == 5: color = "ball-red"
    elif count == 4: color = "ball-yellow"
    elif count == 3: color = "ball-blue"
    elif count == 2: color = "ball-green"
    return f'<div class="lottery-ball {color}">{num:02d}</div>'

def render_prediction_row(nums, counts, special_num=None):
    html = '<div style="display:flex;justify-content:center;flex-wrap:wrap;">'
    for n in nums: html += get_ball_html(n, counts.get(n, 1))
    if special_num is not None: html += get_ball_html(special_num, 1, True)
    html += '</div>'
    return html

def calculate_weights(df, cfg, target_cols, num_range, mode="balanced"):
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
    if len(df) < search_depth + 10: return []
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    valid_strategies = []
    i = len(df) - 1
    try: target_draw = set(df.iloc[i][num_cols].astype(int).values)
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

def save_prediction(game_name, candidates):
    cfg = GAME_CONFIG[game_name]
    file_path = cfg["pred_file"]
    log_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, cand in enumerate(candidates):
        nums_str = ",".join([str(n) for n in cand['n']])
        if 's' in cand and cand['s']: nums_str += f" + {cand['s']}"
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

# --- 7. 介面 ---
with st.sidebar:
    st.title("🎛️ 總控中心 v28.0")
    
    # === 核彈按鈕 ===
    st.header("1. 資料庫維護")
    if st.button("🔄 執行全域重整 (Rebuild DB)", type="primary"):
        with st.spinner("正在暴力掃描硬碟所有 CSV..."):
            stats = rebuild_databases_nuclear()
            load_db_data.clear()
            st.success("重整完成！")
            for g, c in stats.items():
                st.write(f"- **{g}**: {c} 筆")
            time.sleep(2)
            st.rerun()
            
    st.markdown("---")
    st.header("2. 檔案上傳")
    uploaded_files = st.file_uploader("CSV/ZIP", accept_multiple_files=True, type=['csv', 'zip'])
    if uploaded_files:
        if st.button("📥 儲存到 Data 資料夾"):
            for uf in uploaded_files:
                with open(os.path.join(DATA_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            st.success("檔案已存入，請按上方「🔄 執行全域重整」來寫入資料庫！")

    st.markdown("---")
    st.header("3. 遊戲與補單")
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), index=0)
    if st.button(f"🚀 每日補單 ({selected_game})"):
        with st.spinner("連線中..."):
            c = crawl_daily_web(selected_game)
            load_db_data.clear()
            if c>0: st.success(f"更新 {c} 筆")
            else: st.error("爬取失敗")
            st.rerun()

cfg = GAME_CONFIG[selected_game]
df = load_db_data(selected_game)

st.title(f"🎯 {selected_game} 操盤室")

if df.empty:
    st.warning(f"⚠️ {selected_game} 資料庫為空。")
    st.info("1. 請在左側上傳歷年 CSV。\n2. 按下「📥 儲存」。\n3. 務必按下 **「🔄 執行全域重整」**！")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("總期數", len(df))
    c2.metric("起", df.iloc[0]['Date'])
    c3.metric("訖", df.iloc[-1]['Date'])
    
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    avg_std = df_nums.std(axis=1).mean()
    snipers = find_sniper_strategy(df, cfg, search_depth=50)
    sniper_mode = "balanced"
    if snipers:
        best_s = snipers[0]
        sniper_mode = best_s['mode']
        st.success(f"🎯 **狙擊參數**：[{best_s['mode'].upper()}] 策略曾在 {best_s['date']} 命中 {best_s['hits']} 星！")
    
    tab1, tab2, tab3 = st.tabs(["🎲 預測", "📜 紀錄", "📂 資料"])
    with tab1:
        c1, c2 = st.columns(2)
        tol = c1.slider("誤差值", 0.01, 0.5, 0.15, 0.01)
        repeater = c2.checkbox("連莊", value=True)
        if st.button("🎲 運算", type="primary"):
            candidates = []
            templates = [{"name": f"狙擊 ({sniper_mode})", "mode": sniper_mode}, {"name": "順勢", "mode": "trend"}, {"name": "版路", "mode": "banlu"}]
            sp_probs = None
            has_zone2 = cfg.get("special_is_zone2", False)
            z2_col = "Zw" if has_zone2 else ("SP" if cfg["has_special"] else None)
            if z2_col:
                sp_range = cfg.get("special_range", cfg["num_range"])
                sp_probs = calculate_weights(df, cfg, [z2_col], sp_range, "trend")
            bar = st.progress(0)
            for i, temp in enumerate(templates * 2):
                probs = calculate_weights(df, cfg, num_cols, cfg["num_range"], temp["mode"])
                numbers = probs.index.tolist()
                p_vals = probs.values
                att = 0
                found = False
                while not found and att < 5000:
                    sel = sorted(np.random.choice(numbers, cfg["num_count"], replace=False, p=p_vals))
                    if repeater:
                        last = df_nums.iloc[-1].values
                        rep_num = np.random.choice(last)
                        if rep_num not in sel: sel[0] = rep_num; sel.sort()
                    curr_std = np.std(sel, ddof=1)
                    if abs(curr_std - avg_std) <= tol:
                        sp = None
                        if sp_probs is not None: sp = np.random.choice(sp_probs.index.tolist(), p=sp_probs.values)
                        candidates.append({'n': sel, 'e': abs(curr_std - avg_std), 'type': temp["name"], 's': sp})
                        found = True
                    att += 1
                bar.progress((i+1)/6)
            bar.empty()
            st.session_state[f'last_{selected_game}'] = candidates
        if f'last_{selected_game}' in st.session_state:
            results = st.session_state[f'last_{selected_game}']
            if results:
                all_n = []
                for r in results: all_n.extend(r['n'])
                ctr = collections.Counter(all_n)
                cols = st.columns(3)
                for i, r in enumerate(results):
                    with cols[i%3]:
                        html = f'<div class="stCard"><h5>{r["type"]}</h5>'
                        html += '<div class="zone-label">第一區</div>'
                        html += render_prediction_row(r['n'], ctr, None)
                        if r['s']:
                            html += '<div class="zone-label" style="color:red">第二區</div>'
                            html += render_prediction_row([], {}, r['s'])
                        html += f'<small>誤差: {r["e"]:.4f}</small></div>'
                        st.markdown(html, unsafe_allow_html=True)
                if st.button("💾 存檔"):
                    save_prediction(selected_game, results)
                    st.success("已存檔")
    with tab2:
        df_pred = load_predictions(selected_game)
        if not df_pred.empty:
            st.dataframe(df_pred.sort_index(ascending=False), use_container_width=True)
            if st.button("🗑️ 清空"):
                if os.path.exists(cfg["pred_file"]): os.remove(cfg["pred_file"])
                st.rerun()
        else: st.info("無紀錄")
    with tab3:
        st.dataframe(df, use_container_width=True)
