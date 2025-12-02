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
    page_title="台彩數據中心 v29.0", 
    page_icon="🏢", 
    layout="wide",
    initial_sidebar_state="expanded"
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. 資料路徑 ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# --- 3. 遊戲設定 ---
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

# --- 4. 核心工具函式 ---

def parse_date_strict(date_val):
    """強力日期解析"""
    s = str(date_val).strip()
    # 1. 替換分隔符
    s = s.replace('/', '-').replace('.', '-')
    # 2. 嘗試標準 YYYY-MM-DD
    try: return pd.to_datetime(s).strftime('%Y-%m-%d')
    except: pass
    # 3. 嘗試民國年 (96-1-1)
    match = re.match(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', s)
    if match:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if y < 1911: y += 1911
        return f"{y}-{m:02d}-{d:02d}"
    return None

def detect_game_type(filename, df_head):
    filename = filename.lower()
    if '遊戲名稱' in df_head.columns:
        val = str(df_head.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in val: return game
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    return None

@st.cache_data(show_spinner=False, ttl=10)
def load_db_data(game_name):
    cfg = GAME_CONFIG[game_name]
    if os.path.exists(cfg["db_file"]):
        try: return pd.read_csv(cfg["db_file"])
        except: return pd.DataFrame(columns=cfg["cols"])
    return pd.DataFrame(columns=cfg["cols"])

def save_db_data(game_name, df):
    cfg = GAME_CONFIG[game_name]
    if not df.empty:
        # 確保格式
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values(by='Date', ascending=True)
        df.to_csv(cfg["db_file"], index=False)
        load_db_data.clear() # 清快取
        return True
    return False

# --- 5. 頁面 1：資料庫管理 (Admin) ---

def render_admin_page():
    st.title("🗄️ 資料庫管理專區")
    
    # 1. 選擇要管理的資料庫
    game_list = list(GAME_CONFIG.keys())
    selected_game = st.selectbox("選擇要維護的資料庫", game_list)
    cfg = GAME_CONFIG[selected_game]
    
    # 載入目前資料
    df_current = load_db_data(selected_game)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(f"📊 **{selected_game}** 目前狀態")
        if not df_current.empty:
            st.metric("總筆數", len(df_current))
            st.text(f"起：{df_current.iloc[0]['Date']}")
            st.text(f"迄：{df_current.iloc[-1]['Date']}")
        else:
            st.warning("資料庫是空的")

        st.markdown("---")
        st.subheader("📥 匯入歷年資料")
        st.caption("支援拖曳多個 CSV 檔，系統會自動過濾")
        
        uploaded_files = st.file_uploader("上傳檔案", accept_multiple_files=True, type=['csv'])
        if uploaded_files:
            if st.button("開始分析並合併"):
                new_rows = []
                logs = []
                
                progress = st.progress(0)
                for i, up_file in enumerate(uploaded_files):
                    progress.progress((i+1)/len(uploaded_files))
                    try:
                        # 讀取
                        try: df_up = pd.read_csv(up_file, encoding='cp950', on_bad_lines='skip')
                        except: df_up = pd.read_csv(up_file, encoding='utf-8', on_bad_lines='skip')
                        
                        # 檢查是否為該遊戲
                        gtype = detect_game_type(up_file.name, df_up.head(1))
                        if gtype != selected_game:
                            logs.append(f"⚠️ 跳過 {up_file.name} (非 {selected_game})")
                            continue
                            
                        # 解析
                        count = 0
                        df_up.columns = [c.strip() for c in df_up.columns]
                        if '開獎日期' in df_up.columns:
                            for _, row in df_up.iterrows():
                                try:
                                    d = parse_date_strict(row['開獎日期'])
                                    if not d: continue
                                    
                                    nums = []
                                    for k in range(1, cfg["num_count"] + 1):
                                        nums.append(int(row[f'獎號{k}']))
                                    
                                    sp = []
                                    if cfg["has_special"]:
                                        if "第二區" in df_up.columns: sp = [int(row['第二區'])]
                                        elif "特別號" in df_up.columns: sp = [int(row['特別號'])]
                                        else: sp = [0]
                                    
                                    nums.sort()
                                    entry = [d] + nums + sp + ["Admin_Import"]
                                    if len(entry) == len(cfg["cols"]):
                                        new_rows.append(entry)
                                        count += 1
                                except: continue
                        logs.append(f"✅ {up_file.name}: 成功讀取 {count} 筆")
                    except Exception as e:
                        logs.append(f"❌ {up_file.name}: 錯誤 {e}")
                
                # 合併
                if new_rows:
                    df_new = pd.DataFrame(new_rows, columns=cfg["cols"])
                    df_final = pd.concat([df_current, df_new], ignore_index=True)
                    df_final.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                    df_final.sort_values(by='Date', ascending=True, inplace=True)
                    save_db_data(selected_game, df_final)
                    st.success(f"匯入成功！資料庫現有 {len(df_final)} 筆。")
                    st.rerun()
                
                # 顯示報告
                with st.expander("匯入詳細報告"):
                    for l in logs: st.write(l)

    with col2:
        st.subheader("✏️ 資料庫編輯器")
        st.caption("您可以直接修改下表內容，勾選刪除，或在最下方新增資料。修改後請按右上角「Save」")
        
        if not df_current.empty:
            # 使用 data_editor 進行 CRUD
            edited_df = st.data_editor(
                df_current,
                num_rows="dynamic", # 允許新增/刪除
                use_container_width=True,
                height=600,
                key=f"editor_{selected_game}"
            )
            
            # 儲存按鈕 (雖然 data_editor 會自動更新 state，但寫入檔案需手動)
            if st.button("💾 儲存變更至資料庫"):
                save_db_data(selected_game, edited_df)
                st.success("資料庫已更新！")
                
            st.download_button(
                "📥 下載完整資料庫備份 (CSV)",
                edited_df.to_csv(index=False).encode('utf-8-sig'),
                f"{selected_game}_backup.csv",
                "text/csv"
            )
        else:
            st.info("請先從左側匯入資料。")

# --- 6. 頁面 2：預測主頁 (Main Page) ---
# (這裡保留原本 v26 的核心預測邏輯，但簡化顯示)

def render_main_page():
    st.title("🔮 戰情預測主頁")
    
    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), key="main_select")
    cfg = GAME_CONFIG[selected_game]
    df = load_db_data(selected_game)
    
    if df.empty:
        st.error(f"⚠️ {selected_game} 資料庫空白，請切換至「資料庫管理」頁面匯入資料。")
        return

    # 顯示最新一期
    last = df.iloc[-1]
    st.info(f"📅 最新開獎: **{last['Date']}** | 號碼: **{last[cfg['cols'][1:cfg['num_count']+1]].tolist()}**")

    # 預測核心
    st.subheader("AI 運算")
    col_op, col_res = st.columns([1, 2])
    
    with col_op:
        tol = st.slider("誤差值", 0.01, 0.5, 0.15, 0.01)
        repeater = st.checkbox("連莊", value=True)
        
        if st.button("🎲 立即預測"):
            # (這裡沿用原本的運算邏輯，簡化展示)
            num_cols = [c for c in cfg["cols"] if c.startswith("N")]
            df_nums = df[num_cols].apply(pd.to_numeric)
            avg_std = df_nums.std(axis=1).mean()
            
            # 權重
            mn, mx = cfg["num_range"]
            vals = df_nums.values.flatten()
            freq = pd.Series(vals).value_counts().sort_index().reindex(range(mn, mx+1), fill_value=0)
            w = freq.values / freq.values.sum()
            nums = freq.index.tolist()
            
            res = []
            att = 0
            while len(res) < 5 and att < 10000:
                sel = sorted(np.random.choice(nums, cfg["num_count"], replace=False, p=w))
                # 連莊
                if repeater:
                    last_n = df_nums.iloc[-1].values
                    r = np.random.choice(last_n)
                    if r not in sel: sel[0] = r; sel.sort()
                
                curr_std = np.std(sel, ddof=1)
                if abs(curr_std - avg_std) <= tol:
                    # 第二區
                    sp = None
                    if cfg["has_special"]:
                        z2_col = "Zw" if "Zw" in df.columns else "SP"
                        sp_vals = df[z2_col].value_counts().sort_index().index.tolist()
                        if sp_vals: sp = np.random.choice(sp_vals)
                    
                    res.append({'n': sel, 's': sp, 'e': abs(curr_std - avg_std)})
                att += 1
            
            st.session_state['pred_res'] = res

    with col_res:
        if 'pred_res' in st.session_state:
            results = st.session_state['pred_res']
            for i, r in enumerate(results):
                txt = f"**第 {i+1} 組**: {r['n']}"
                if r['s']: txt += f" + <span style='color:red'>[{r['s']}]</span>"
                st.markdown(txt, unsafe_allow_html=True)

# --- 7. 導航控制 ---

# CSS 美化
st.markdown("""
<style>
.stApp { background-color: #f0f7f4; }
.lottery-ball { display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%; text-align: center; background: #fff; border: 1px solid #ccc; margin: 2px; }
</style>
""", unsafe_allow_html=True)

# 側邊欄導航
page = st.sidebar.radio("功能選單", ["🔮 戰情預測主頁", "🗄️ 資料庫管理專區"])

if page == "🔮 戰情預測主頁":
    render_main_page()
else:
    render_admin_page()
