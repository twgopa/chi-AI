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
st.set_page_config(page_title="台彩數據中心 v22.0 (熱點聚焦版)", page_icon="🔥", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

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

def process_bulk_files(uploaded_files, progress_bar=None):
    results = {g: 0 for g in GAME_CONFIG.keys()}
    temp_storage = {g: [] for g in GAME_CONFIG.keys()}
    total = len(uploaded_files)
    
    for i, up_file in enumerate(uploaded_files):
        if progress_bar: progress_bar.progress((i + 1) / total, text=f"處理中: {up_file.name}")
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
    if game_name not in GAME_CONFIG: return pd.DataFrame(), []
    cfg = GAME_CONFIG[game_name]
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    merged_data = []
    debug_log = []
    target_files = [f for f in all_files if "prediction_log.csv" not in f]

    for file_path in target_files:
        filename = os.path.basename(file_path)
        is_related = any(k in filename for k in cfg["keywords"])
        if not is_related and any(k in file_path for k in cfg["keywords"]): is_related = True
        
        if is_related:
            file_status = {"name": filename, "status": "OK", "count": 0}
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
                    file_status["count"] = len(temp_rows)

                elif 'Date' in df.columns:
                    valid_cols = [c for c in cfg["cols"] if c in df.columns]
                    temp_df = df[valid_cols].copy()
                    if "Source" not in temp_df.columns: temp_df["Source"] = "Auto"
                    if len(temp_df.columns) == len(cfg["cols"]):
                        merged_data.extend(temp_df.values.tolist())
                        file_status["count"] = len(temp_df)
            except: pass
            debug_log.append(file_status)

    if merged_data:
        final_df = pd.DataFrame(merged_data, columns=cfg["cols"])
        final_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        final_df.sort_values(by='Date', ascending=True, inplace=True)
        return final_df, debug_log
    return pd.DataFrame(columns=cfg["cols"]), debug_log

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

def save_prediction_log(game_name, candidates):
    log_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, cand in enumerate(candidates):
        nums_str = ", ".join([f"{n:02d}" for n in cand['n']])
        row = {
            "Timestamp": timestamp, "Game": game_name, "Set_ID": cand['type'],
            "Numbers": nums_str, "Error": f"{cand['e']:.4f}", 
            "Hit_Repeats": str(cand.get('r', [])), "Note": cand.get('note', '')
        }
        log_data.append(row)
    df_log = pd.DataFrame(log_data)
    if os.path.exists(LOG_FILE):
        try: old = pd.read_csv(LOG_FILE); df_final = pd.concat([old, df_log], ignore_index=True)
        except: df_final = df_log
    else: df_final = df_log
    df_final.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

def load_prediction_log():
    if os.path.exists(LOG_FILE):
        try: return pd.read_csv(LOG_FILE)
        except: return pd.DataFrame()
    return pd.DataFrame()

# --- 5. 演算邏輯 ---

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
    st.toast("🎯 搜尋最佳參數...")
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

# --- 6. 介面主程式 ---

with st.sidebar:
    st.title("🎛️ 菁英總控 v22.0")
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), index=0)
    
    now = datetime.now()
    if now.hour >= 17: st.warning("🔔 下午 5 點後，請按下方補單！")
    
    st.markdown("---")
    uploaded_files = st.file_uploader("上傳 CSV/ZIP", accept_multiple_files=True, type=['csv', 'zip'])
    if uploaded_files:
        if st.button("📥 匯入資料"):
            with st.spinner("處理中..."):
                process_bulk_files(uploaded_files, None)
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
df, logs = load_all_data(selected_game)

st.header(f"📊 {selected_game} 菁英操盤室")

if df.empty:
    st.error("⚠️ 資料庫空白，請先匯入檔案。")
else:
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    avg_std = df_nums.std(axis=1).mean()
    
    snipers = find_sniper_strategy(df, cfg, search_depth=50)
    sniper_mode = "balanced"
    sniper_tol = 0.15
    
    if snipers:
        best_s = snipers[0]
        sniper_mode = best_s['mode']
        sniper_tol = best_s['tol']
        st.markdown(f"<div style='background:#d4edda;color:#155724;padding:15px;border-radius:10px;margin-bottom:10px;'><b>🎯 鎖定參數：</b>[{best_s['mode']}] + 誤差 {best_s['tol']} (歷史命中 {best_s['hits']} 星)</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔮 熱點預測", "📋 紀錄", "📂 資料"])

    with tab1:
        st.subheader("運算設定")
        c1, c2 = st.columns(2)
        tol = c1.slider("標準差誤差", 0.01, 0.5, sniper_tol, 0.01)
        repeater = c2.checkbox("啟用連莊", value=True)

        if st.button("🎲 啟動多模組運算", type="primary"):
            candidates = []
            templates = [{"name": f"🌟 狙擊 ({sniper_mode})", "mode": sniper_mode}, {"name": "🔥 順勢", "mode": "trend"}, {"name": "🐲 版路", "mode": "banlu"}]
            bar = st.progress(0)
            
            for i, temp in enumerate(templates):
                probs = calculate_weights(df, cfg, temp["mode"])
                numbers = probs.index.tolist()
                p_vals = probs.values
                count = 0
                att = 0
                while count < 2 and att < 10000:
                    sel = sorted(np.random.choice(numbers, cfg["num_count"], replace=False, p=p_vals))
                    if repeater:
                        last_draw = df_nums.iloc[-1].values
                        rep_num = np.random.choice(last_draw)
                        if rep_num not in sel:
                            sel[0] = rep_num
                            sel.sort()
                    curr_std = np.std(sel, ddof=1)
                    if abs(curr_std - avg_std) <= tol:
                        if not any(x['n'] == sel for x in candidates):
                            candidates.append({'n': sel, 'e': abs(curr_std - avg_std), 'type': temp["name"]})
                            count += 1
                    att += 1
                bar.progress((i + 1) / len(templates))
            bar.empty()
            st.session_state['last_candidates'] = candidates
            st.session_state['last_game'] = selected_game

        if 'last_candidates' in st.session_state and st.session_state['last_candidates']:
            results = st.session_state['last_candidates']
            
            # --- 核心新功能：計算熱點號碼 ---
            all_pred_nums = []
            for res in results:
                all_pred_nums.extend(res['n'])
            
            freq_counter = pd.Series(all_pred_nums).value_counts()
            # 出現 2 次以上視為熱點
            hot_nums = freq_counter[freq_counter >= 2].index.tolist()

            # 顯示熱點提示
            if hot_nums:
                hot_html = "".join([f"<span style='background:#ff4b4b;color:white;padding:3px 8px;border-radius:15px;margin:3px;font-weight:bold'>{n:02d}</span>" for n in sorted(hot_nums)])
                st.markdown(f"🔥 **共同推薦熱點 (重複出現)：** {hot_html}", unsafe_allow_html=True)
                st.markdown("---")

            # 顯示預測結果 (套用紅色標記)
            for i in range(0, len(results), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        res = results[i+j]
                        with col:
                            st.markdown(f"**{res['type']}**")
                            
                            # 產生彩色號碼 HTML
                            num_htmls = []
                            for n in res['n']:
                                n_str = f"{n:02d}"
                                if n in hot_nums:
                                    # 紅色熱點樣式
                                    style = "background:#ff4b4b; color:white; padding:2px 8px; border-radius:5px; font-weight:bold; margin:1px; box-shadow:0 0 4px rgba(255,0,0,0.4);"
                                else:
                                    # 一般樣式
                                    style = "background:#f0f2f6; color:#333; padding:2px 8px; border-radius:5px; margin:1px; border:1px solid #ddd;"
                                num_htmls.append(f"<span style='{style}'>{n_str}</span>")
                            
                            st.markdown(" ".join(num_htmls), unsafe_allow_html=True)
                            st.caption(f"誤差: {res['e']:.4f}")
            
            st.divider()
            if st.button("💾 儲存"):
                save_prediction_log(st.session_state['last_game'], results)
                st.success("已存檔")

    with tab2:
        df_log = load_prediction_log()
        if not df_log.empty:
            st.dataframe(df_log.sort_index(ascending=False), use_container_width=True)
            if st.button("🗑️ 清空"):
                if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
                st.rerun()
        else: st.info("無紀錄")

    with tab3:
        st.dataframe(df, use_container_width=True)
