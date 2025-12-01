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
import altair as alt
import zipfile

# --- 1. 系統設定 ---
st.set_page_config(page_title="台彩數據中心 v16.1 (強力讀取版)", page_icon="📂", layout="wide")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑與自動解壓 (v16.1 升級) ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 核心升級：自動搜尋目錄下所有的 .zip 檔案並解壓縮
# 這樣您可以上傳 data1.zip, data2.zip 分開傳，避開 25MB 限制
zip_files = glob.glob("*.zip") + glob.glob(os.path.join(DATA_DIR, "*.zip"))

for z_file in zip_files:
    try:
        # 檢查是否已經解壓過 (簡單判斷：如果 zip 很大，但資料夾是空的)
        # 這裡我們採取「每次啟動嘗試解壓」策略，因為 Streamlit Cloud 重啟後是乾淨的
        with zipfile.ZipFile(z_file, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print(f"成功解壓縮: {z_file}")
    except Exception as e:
        print(f"解壓縮失敗 {z_file}: {e}")

LOG_FILE = os.path.join(DATA_DIR, "prediction_log.csv")

# --- 3. 遊戲設定 ---
GAME_CONFIG = {
    "今彩539": {
        "keywords": ["今彩539", "539"],
        "num_count": 5,
        "num_range": (1, 39),
        "has_special": False,
        "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "Source"]
    },
    "大樂透": {
        "keywords": ["大樂透", "Lotto649"],
        "num_count": 6,
        "num_range": (1, 49),
        "has_special": True,
        "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "SP", "Source"]
    },
    "威力彩": {
        "keywords": ["威力彩", "SuperLotto"],
        "num_count": 6,
        "num_range": (1, 38),
        "has_special": True, 
        "enable_predict": True,
        "cols": ["Date", "N1", "N2", "N3", "N4", "N5", "N6", "Zw", "Source"]
    },
}

# --- 4. 基礎資料讀取與爬蟲 ---

def detect_game_type(filename, df_head):
    filename = filename.lower()
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    if '遊戲名稱' in df_head.columns:
        game_name_in_csv = str(df_head.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in game_name_in_csv: return game
    return None

def process_bulk_files(uploaded_files):
    results = {g: 0 for g in GAME_CONFIG.keys()}
    temp_storage = {g: [] for g in GAME_CONFIG.keys()}
    
    for up_file in uploaded_files:
        try:
            try: df = pd.read_csv(up_file, encoding='cp950')
            except: 
                try: df = pd.read_csv(up_file, encoding='big5')
                except: 
                    up_file.seek(0)
                    df = pd.read_csv(up_file, encoding='utf-8')
            
            df.columns = [c.strip() for c in df.columns]
            game_type = detect_game_type(up_file.name, df.head(1))
            if not game_type: continue
            
            cfg = GAME_CONFIG[game_type]
            for _, row in df.iterrows():
                try:
                    date_raw = str(row['開獎日期']).strip()
                    date_str = pd.to_datetime(date_raw).strftime('%Y-%m-%d')
                    nums = [int(row[f'獎號{k}']) for k in range(1, cfg["num_count"] + 1)]
                    sp_val = []
                    if cfg["has_special"]:
                        if "第二區" in df.columns: sp_val = [int(row['第二區'])]
                        elif "特別號" in df.columns: sp_val = [int(row['特別號'])]
                    if cfg["enable_predict"]: nums.sort()
                    entry = [date_str] + nums + sp_val + ["UserUpload"]
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
    if game_name not in GAME_CONFIG: return pd.DataFrame()
    cfg = GAME_CONFIG[game_name]
    
    # 遞迴讀取所有 CSV
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    merged_data = []
    
    target_files = [f for f in all_files if "prediction_log.csv" not in f]

    for file_path in target_files:
        filename = os.path.basename(file_path)
        # 簡單過濾：檔名包含關鍵字，且不包含 "賓果" (排除大檔案)
        if any(k in filename for k in cfg["keywords"]) and "賓果" not in filename:
            try:
                try: df = pd.read_csv(file_path, encoding='cp950')
                except: 
                    try: df = pd.read_csv(file_path, encoding='utf-8')
                    except: continue
                
                df.columns = [c.strip() for c in df.columns]
                
                if '開獎日期' in df.columns:
                    for _, row in df.iterrows():
                        try:
                            d_str = pd.to_datetime(str(row['開獎日期']).strip()).strftime('%Y-%m-%d')
                            nums = [int(row[f'獎號{i}']) for i in range(1, cfg["num_count"] + 1)]
                            sp = []
                            if cfg["has_special"]:
                                if "第二區" in df.columns: sp = [int(row['第二區'])]
                                elif "特別號" in df.columns: sp = [int(row['特別號'])]
                            if cfg["enable_predict"]: nums.sort()
                            entry = [d_str] + nums + sp + ["Official"]
                            if len(entry) == len(cfg["cols"]): merged_data.append(entry)
                        except: continue
                elif 'Date' in df.columns:
                    valid_cols = [c for c in cfg["cols"] if c in df.columns]
                    temp_df = df[valid_cols].copy()
                    if "Source" not in temp_df.columns: temp_df["Source"] = "Auto"
                    for _, row in temp_df.iterrows():
                        if len(row) == len(cfg["cols"]): merged_data.append(row.tolist())
            except: continue

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
            if d_str < "2024-01-01": continue
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
                entry = [d_str] + valid_n + sp_n + ["Daily_Web"]
                if len(entry) == len(cfg["cols"]): new_rows.append(entry)
    except: pass
    if new_rows:
        filename = f"Daily_Patch_{game_name}.csv"
        path = os.path.join(DATA_DIR, filename)
        df_new = pd.DataFrame(new_rows, columns=cfg["cols"])
        if os.path.exists(path):
            old = pd.read_csv(path)
            final = pd.concat([old, df_new]).drop_duplicates(subset=['Date'], keep='last')
        else: final = df_new
        final.to_csv(path, index=False)
        return len(new_rows)
    return 0

# --- 5. 紀錄存檔模組 ---

def save_prediction_log(game_name, candidates):
    log_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, cand in enumerate(candidates):
        nums_str = ", ".join([f"{n:02d}" for n in cand['n']])
        row = {
            "Timestamp": timestamp,
            "Game": game_name,
            "Set_ID": f"第 {i+1} 組",
            "Numbers": nums_str,
            "Error": f"{cand['e']:.4f}",
            "Hit_Repeats": str(cand.get('r', [])),
            "Note": cand.get('note', '')
        }
        log_data.append(row)
    
    df_log = pd.DataFrame(log_data)
    if os.path.exists(LOG_FILE):
        try:
            old_log = pd.read_csv(LOG_FILE)
            df_final = pd.concat([old_log, df_log], ignore_index=True)
        except: df_final = df_log
    else: df_final = df_log
    df_final.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')

def load_prediction_log():
    if os.path.exists(LOG_FILE):
        try: return pd.read_csv(LOG_FILE)
        except: return pd.DataFrame()
    return pd.DataFrame()

# --- 6. AI 演算法 ---

def search_for_miracle_strategy(df, cfg, search_depth=50):
    if len(df) < search_depth + 10: return False, None, None, None
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_nums = df[num_cols].apply(pd.to_numeric)
    param_space = [(0.8, 0.2, 0.12), (0.5, 0.5, 0.15), (0.2, 0.8, 0.18), (0.9, 0.1, 0.10)]
    mn, mx = cfg["num_range"]
    all_range = list(range(mn, mx+1))
    
    for i in range(len(df)-1, len(df)-search_depth-1, -1):
        target_draw = set(df_nums.iloc[i].values)
        train_data = df_nums.iloc[:i]
        if len(train_data) < 30: continue
        last_draw = train_data.iloc[-1].values
        drag_counts = pd.Series(0.0, index=all_range)
        data_matrix = train_data.values
        for target_num in last_draw:
            indices = np.where(data_matrix[:-1] == target_num)[0]
            next_indices = indices + 1
            if len(next_indices) > 0:
                next_draws = data_matrix[next_indices]
                counts = pd.Series(next_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
                drag_counts = drag_counts.add(counts, fill_value=0)
        prob_banlu = drag_counts / drag_counts.sum() if drag_counts.sum() > 0 else pd.Series(1/len(all_range), index=all_range)
        freq_recent = pd.Series(train_data.tail(30).values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
        prob_recent = (freq_recent + 0.1) / (freq_recent.sum() + 1)
        
        for params in param_space:
            w_banlu, w_recent, tol = params
            final_prob = (prob_banlu * w_banlu) + (prob_recent * w_recent)
            final_prob = final_prob / final_prob.sum()
            top_candidates = final_prob.nlargest(8).index.tolist()
            if target_draw.issubset(set(top_candidates)):
                curr_last = df_nums.iloc[-1].values
                d_cts = pd.Series(0.0, index=all_range)
                d_mtx = df_nums.values
                for t_num in curr_last:
                    idxs = np.where(d_mtx[:-1] == t_num)[0]
                    n_idxs = idxs + 1
                    if len(n_idxs) > 0:
                        n_draws = d_mtx[n_idxs]
                        cts = pd.Series(n_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
                        d_cts = d_cts.add(cts, fill_value=0)
                p_b = d_cts / d_cts.sum() if d_cts.sum() > 0 else pd.Series(1/len(all_range), index=all_range)
                f_rec = pd.Series(df_nums.tail(30).values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
                p_r = (f_rec + 0.1) / (f_rec.sum() + 1)
                f_prob = (p_b * w_banlu) + (p_r * w_recent)
                f_prob = f_prob / f_prob.sum()
                pred = sorted(f_prob.nlargest(cfg["num_count"]).index.tolist())
                return True, params, pred, df.iloc[i]['Date']
    return False, None, None, None

def analyze_stats(df, cfg):
    if df.empty: return 0
    num_cols = [c for c in cfg["cols"] if c.startswith("N")]
    df_calc = df[num_cols].apply(pd.to_numeric)
    return df_calc.std(axis=1).mean()

# --- 7. 介面主程式 ---

with st.sidebar:
    st.title("🎛️ 總控中心 v16.1")
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), index=0)
    st.markdown("---")
    
    st.subheader("📂 補檔案")
    uploaded_files = st.file_uploader("拖曳 ZIP/CSV 至此", accept_multiple_files=True, type=['csv', 'zip'])
    
    if uploaded_files:
        if st.button("📥 確認匯入"):
            with st.spinner("處理中..."):
                processed_count = 0
                for up_file in uploaded_files:
                    if up_file.name.endswith('.zip'):
                        with zipfile.ZipFile(up_file, 'r') as zip_ref:
                            zip_ref.extractall(DATA_DIR)
                        processed_count += 1
                    else:
                        process_bulk_files([up_file])
                        processed_count += 1
                load_all_data.clear()
                st.success(f"處理完成！共處理 {processed_count} 個檔案/壓縮包。")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    if st.button("🚀 每日補單 (i539)"):
        with st.spinner("更新中..."):
            c = crawl_daily_web(selected_game)
            if c>0: 
                load_all_data.clear()
                st.success(f"更新 {c} 筆！")
                st.rerun()
            else: st.info("無新資料")

df = load_all_data(selected_game)
cfg = GAME_CONFIG[selected_game]

st.header(f"📊 {selected_game} 戰情紀錄系統")

if df.empty:
    st.error("資料庫空白。請在 GitHub 上傳 data.zip 或 CSV。")
else:
    avg_std = analyze_stats(df, cfg)
    last_draw = df.iloc[-1][cfg["cols"][1:cfg["num_count"]+1]].tolist()
    last_draw = [int(x) for x in last_draw]
    
    has_miracle, m_params, m_pred, m_date = search_for_miracle_strategy(df, cfg)

    if has_miracle:
        st.markdown(f"""
        <div style="background:linear-gradient(90deg, #FFD700, #FF8C00);padding:20px;border-radius:15px;color:black;text-align:center;margin-bottom:25px;">
            <div style="font-size:24px;font-weight:bold;">⚡ 天選號碼 (The Chosen One) ⚡</div>
            <div style="font-size:40px;font-weight:bold;color:white;text-shadow:2px 2px 4px black;letter-spacing:5px;">{m_pred}</div>
            <div style="font-size:14px;">此策略曾在 <b>{m_date}</b> 完全命中頭獎！</div>
        </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔮 AI 預測", "📝 預測歷史", "📋 資料庫"])

    with tab1:
        st.subheader("預測參數設定")
        c1, c2, c3 = st.columns(3)
        tol = c1.slider("標準差誤差", 0.01, 0.5, 0.12, 0.01)
        repeater_mode = c2.checkbox("啟用連莊", value=True)
        w_ratio = c3.slider("版路/近期權重", 0.0, 1.0, 0.6)

        if 'last_candidates' not in st.session_state: st.session_state['last_candidates'] = []
        if 'last_game' not in st.session_state: st.session_state['last_game'] = ""

        if st.button("🎲 啟動運算", type="primary"):
            st.session_state['last_game'] = selected_game
            num_cols = [c for c in cfg["cols"] if c.startswith("N")]
            df_nums = df[num_cols].apply(pd.to_numeric)
            mn, mx = cfg["num_range"]
            all_range = list(range(mn, mx+1))
            
            drag_counts = pd.Series(0.0, index=all_range)
            data_matrix = df_nums.values
            for target_num in last_draw:
                indices = np.where(data_matrix[:-1] == target_num)[0]
                next_indices = indices + 1
                if len(next_indices) > 0:
                    next_draws = data_matrix[next_indices]
                    counts = pd.Series(next_draws.flatten()).value_counts().reindex(all_range, fill_value=0)
                    drag_counts = drag_counts.add(counts, fill_value=0)
            prob_banlu = drag_counts / drag_counts.sum() if drag_counts.sum() > 0 else pd.Series(1/len(all_range), index=all_range)
            
            freq_recent = pd.Series(df_nums.tail(30).values.flatten()).value_counts().sort_index().reindex(all_range, fill_value=0)
            prob_recent = (freq_recent + 0.1) / (freq_recent.sum() + 1)
            
            final_prob = (prob_banlu * w_ratio) + (prob_recent * (1-w_ratio))
            final_prob = final_prob / final_prob.sum()
            
            numbers = final_prob.index.tolist()
            probs = final_prob.values
            
            candidates = []
            attempts = 0
            bar = st.progress(0)
            last_draw_set = list(last_draw)
            
            if has_miracle:
                hit_rep_m = set(m_pred).intersection(set(last_draw_set))
                curr_std_m = np.std(m_pred, ddof=1)
                err_m = abs(curr_std_m - avg_std)
                candidates.append({'n': m_pred, 'e': err_m, 'r': list(hit_rep_m), 'note': '天選號碼'})

            while len(candidates) < 5 and attempts < 50000:
                selection = []
                if repeater_mode:
                    rep = np.random.choice(last_draw_set, 1, replace=False).tolist()
                    selection.extend(rep)
                needed = cfg["num_count"] - len(selection)
                temp_pool = [n for n in numbers if n not in selection]
                temp_probs = [final_prob[n] for n in temp_pool]
                temp_probs = np.array(temp_probs) / sum(temp_probs)
                others = np.random.choice(temp_pool, needed, replace=False, p=temp_probs).tolist()
                selection.extend(others)
                selection.sort()
                curr_std = np.std(selection, ddof=1)
                if abs(curr_std - avg_std) <= tol:
                    if not any(x['n'] == selection for x in candidates):
                        hit_rep = set(selection).intersection(set(last_draw_set))
                        candidates.append({'n': selection, 'e': abs(curr_std - avg_std), 'r': list(hit_rep), 'note': ''})
                        bar.progress(len(candidates)/5)
                attempts += 1
            bar.empty()
            st.session_state['last_candidates'] = candidates

        if st.session_state['last_candidates']:
            res_candidates = st.session_state['last_candidates']
            cols_ui = st.columns(len(res_candidates))
            for i, (col, res) in enumerate(zip(cols_ui, res_candidates)):
                with col:
                    if res.get('note') == '天選號碼':
                        st.markdown(f"**⚡ 天選號碼**")
                        st.markdown(f"<span style='color:#FFD700;font-weight:bold;font-size:18px'>{res['n']}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**第 {i+1} 組**")
                        st.code(str(res['n']))
                    st.caption(f"誤差: {res['e']:.3f}")
                    if res['r']: st.caption(f"連莊: {res['r']}")
            st.divider()
            if st.button("💾 儲存本次預測紀錄", type="primary"):
                save_prediction_log(st.session_state['last_game'], res_candidates)
                st.success("✅ 已儲存至「預測歷史」分頁！")

    with tab2:
        st.subheader("📝 您的歷史預測紀錄")
        df_log = load_prediction_log()
        if not df_log.empty:
            df_log = df_log.sort_index(ascending=False)
            filter_game = st.checkbox("只顯示目前選擇的彩種", value=True)
            if filter_game: df_show = df_log[df_log["Game"] == selected_game]
            else: df_show = df_log
            st.dataframe(df_show, use_container_width=True, height=500)
            if st.button("🗑️ 清空所有紀錄"):
                if os.path.exists(LOG_FILE):
                    os.remove(LOG_FILE)
                    st.success("紀錄已清空")
                    st.rerun()
        else: st.info("目前沒有儲存的紀錄。")

    with tab3:
        st.dataframe(df, use_container_width=True)