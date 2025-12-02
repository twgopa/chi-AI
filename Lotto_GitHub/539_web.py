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
import io

# --- 1. 系統設定 ---
st.set_page_config(
    page_title="台彩數據中心 v30.0", 
    page_icon="🛠️", 
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

# --- 4. 核心工具函式 (智慧修復) ---

def parse_date_strict(date_val):
    """強力日期解析"""
    s = str(date_val).strip()
    s = s.replace('/', '-').replace('.', '-')
    try: return pd.to_datetime(s).strftime('%Y-%m-%d')
    except: pass
    match = re.match(r'(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})', s)
    if match:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if y < 1911: y += 1911
        return f"{y}-{m:02d}-{d:02d}"
    return None

def detect_game_type(filename, df):
    filename = filename.lower()
    # 內容判斷
    if '遊戲名稱' in df.columns and not df.empty:
        val = str(df.iloc[0]['遊戲名稱'])
        for game in GAME_CONFIG.keys():
            if game in val: return game
    # 檔名判斷
    for game, cfg in GAME_CONFIG.items():
        for kw in cfg["keywords"]:
            if kw.lower() in filename: return game
    return None

def smart_read_csv(uploaded_file):
    """
    v30 核心：智慧讀取器
    自動尋找「期別」或「開獎日期」所在的行數，跳過標題行
    """
    try:
        # 1. 嘗試讀取前 20 行來分析
        content = uploaded_file.getvalue()
        
        # 嘗試解碼
        try: text = content.decode('cp950')
        except: 
            try: text = content.decode('big5')
            except: text = content.decode('utf-8')
            
        lines = text.splitlines()
        header_row = 0
        found_header = False
        
        # 尋找標題行
        for i, line in enumerate(lines[:20]):
            if "期別" in line or "開獎日期" in line:
                header_row = i
                found_header = True
                break
        
        # 2. 使用正確的 header row 重新讀取
        # 使用 io.StringIO 模擬檔案
        from io import StringIO
        df = pd.read_csv(StringIO(text), header=header_row)
        
        # 清理欄位
        df.columns = [str(c).strip().replace(" ", "") for c in df.columns]
        
        return df, "OK"
        
    except Exception as e:
        return None, str(e)

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
        try:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        except: pass
        df = df.sort_values(by='Date', ascending=True)
        df.to_csv(cfg["db_file"], index=False)
        load_db_data.clear()
        return True
    return False

# --- 5. 頁面 1：資料庫管理 ---

def render_admin_page():
    st.title("🗄️ 資料庫管理專區 v30")
    
    game_list = list(GAME_CONFIG.keys())
    selected_game = st.selectbox("選擇要維護的資料庫", game_list)
    cfg = GAME_CONFIG[selected_game]
    df_current = load_db_data(selected_game)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info(f"📊 **{selected_game}** 現況")
        if not df_current.empty:
            st.metric("總筆數", len(df_current))
            st.text(f"起：{df_current.iloc[0]['Date']}")
            st.text(f"迄：{df_current.iloc[-1]['Date']}")
        else:
            st.warning("資料庫是空的")

        st.markdown("---")
        st.subheader("📥 匯入歷年資料 (強力修復)")
        st.caption("系統會自動跳過標題行，解決「No columns」錯誤")
        
        uploaded_files = st.file_uploader("上傳檔案", accept_multiple_files=True, type=['csv'])
        if uploaded_files:
            if st.button("開始分析並合併"):
                new_rows = []
                logs = []
                progress = st.progress(0)
                
                for i, up_file in enumerate(uploaded_files):
                    progress.progress((i+1)/len(uploaded_files))
                    
                    # v30 使用智慧讀取
                    df_up, status = smart_read_csv(up_file)
                    
                    if df_up is None:
                        logs.append(f"❌ {up_file.name}: {status}")
                        continue
                        
                    # 檢查遊戲類型
                    gtype = detect_game_type(up_file.name, df_up.head(1))
                    if gtype != selected_game:
                        # 嘗試寬鬆判斷 (若檔名包含遊戲名)
                        if selected_game in up_file.name:
                             pass # 強制通過
                        else:
                             logs.append(f"⚠️ 跳過 {up_file.name} (非 {selected_game})")
                             continue

                    # 解析
                    count = 0
                    if '開獎日期' in df_up.columns:
                        for _, row in df_up.iterrows():
                            try:
                                d = parse_date_strict(row['開獎日期'])
                                if not d: continue
                                
                                nums = []
                                for k in range(1, cfg["num_count"] + 1):
                                    # 容錯：有時候欄位叫 '獎號1' 有時候叫 '第一區1'
                                    val = None
                                    if f'獎號{k}' in df_up.columns: val = row[f'獎號{k}']
                                    elif f'第一區{k}' in df_up.columns: val = row[f'第一區{k}']
                                    
                                    if val is not None: nums.append(int(val))
                                
                                if len(nums) != cfg["num_count"]: continue
                                
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
                    logs.append(f"✅ {up_file.name}: 讀取 {count} 筆")
                
                if new_rows:
                    df_new = pd.DataFrame(new_rows, columns=cfg["cols"])
                    df_final = pd.concat([df_current, df_new], ignore_index=True)
                    df_final.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                    df_final.sort_values(by='Date', ascending=True, inplace=True)
                    save_db_data(selected_game, df_final)
                    st.success(f"匯入成功！資料庫現有 {len(df_final)} 筆。")
                    time.sleep(1)
                    st.rerun()
                
                with st.expander("匯入報告", expanded=True):
                    for l in logs: st.write(l)

    with col2:
        st.subheader("✏️ 資料庫編輯器")
        if not df_current.empty:
            edited_df = st.data_editor(
                df_current, num_rows="dynamic", use_container_width=True, height=600, key=f"ed_{selected_game}"
            )
            if st.button("💾 儲存變更"):
                save_db_data(selected_game, edited_df)
                st.success("已更新！")
            st.download_button("📥 下載備份", edited_df.to_csv(index=False).encode('utf-8-sig'), f"{selected_game}_backup.csv", "text/csv")

# --- 6. 頁面 2：預測主頁 ---

def render_main_page():
    st.title("🔮 戰情預測主頁")
    
    # 側邊欄爬蟲
    with st.sidebar:
        st.markdown("---")
        if st.button("🚀 執行每日補單"):
            st.toast("連線中...")
            # 這裡簡化，直接用簡單邏輯示範，實際請用完整爬蟲
            st.info("請使用資料庫管理區匯入 CSV")

    selected_game = st.selectbox("選擇彩種", list(GAME_CONFIG.keys()), key="main_gm")
    cfg = GAME_CONFIG[selected_game]
    df = load_db_data(selected_game)
    
    if df.empty:
        st.error("⚠️ 資料庫空白，請至「資料庫管理專區」匯入檔案。")
        return

    last = df.iloc[-1]
    nums_show = last[cfg['cols'][1:cfg['num_count']+1]].tolist()
    st.info(f"📅 最新開獎: **{last['Date']}** | 號碼: **{nums_show}**")

    # 運算區
    c1, c2 = st.columns(2)
    tol = c1.slider("誤差值", 0.01, 0.5, 0.15, 0.01)
    repeater = c2.checkbox("連莊", value=True)
    
    if st.button("🎲 立即預測", type="primary"):
        num_cols = [c for c in cfg["cols"] if c.startswith("N")]
        df_nums = df[num_cols].apply(pd.to_numeric)
        avg_std = df_nums.std(axis=1).mean()
        
        mn, mx = cfg["num_range"]
        vals = df_nums.values.flatten()
        freq = pd.Series(vals).value_counts().sort_index().reindex(range(mn, mx+1), fill_value=0)
        w = freq.values / freq.values.sum()
        nums = freq.index.tolist()
        
        res = []
        att = 0
        while len(res) < 5 and att < 10000:
            sel = sorted(np.random.choice(nums, cfg["num_count"], replace=False, p=w))
            if repeater:
                last_n = df_nums.iloc[-1].values
                r = np.random.choice(last_n)
                if r not in sel: sel[0] = r; sel.sort()
            
            curr_std = np.std(sel, ddof=1)
            if abs(curr_std - avg_std) <= tol:
                sp = None
                if cfg["has_special"]:
                    z2_col = "Zw" if "Zw" in df.columns else "SP"
                    sp_vals = df[z2_col].value_counts().sort_index().index.tolist()
                    if sp_vals: sp = np.random.choice(sp_vals)
                res.append({'n': sel, 's': sp, 'e': abs(curr_std - avg_std)})
            att += 1
        
        st.session_state['pred_res'] = res

    if 'pred_res' in st.session_state:
        results = st.session_state['pred_res']
        cols = st.columns(3)
        for i, r in enumerate(results):
            with cols[i%3]:
                txt = f"**第 {i+1} 組**: {r['n']}"
                if r['s']: txt += f" + <span style='color:red'>[{r['s']}]</span>"
                st.markdown(txt, unsafe_allow_html=True)
                st.caption(f"誤差: {r['e']:.4f}")

# --- 7. 導航 ---
st.markdown("""
<style>
.lottery-ball { display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%; text-align: center; background: #fff; border: 1px solid #ccc; margin: 2px; }
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("功能選單", ["🗄️ 資料庫管理專區", "🔮 戰情預測主頁"])

if page == "🔮 戰情預測主頁":
    render_main_page()
else:
    render_admin_page()
