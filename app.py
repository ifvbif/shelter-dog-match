"""
收容犬認養媒合推薦系統（期末報告示範）
=================================================
依使用者的居住縣市、住宅型態、每日可陪伴外出活動時間、能否接受成犬，
從農業部「動物認領養」開放資料中推薦適合的待認養犬，並讓等待較久的狗被優先看見。

本系統是「認養前的自我評估工具」，不是認養審核工具；
實際認養資格與流程以各收容所公告為準。

第三方套件：gradio（網頁介面）、pandas（資料整理，需 2.0 以上）、requests（下載開放資料）
安裝：在終端機輸入 pip install gradio pandas requests
執行：python app.py，再用瀏覽器開啟終端機顯示的網址（http://127.0.0.1:7860）
"""

import html
import json
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import requests

# ============================================================
# Part 1. 系統設定
# ============================================================
API_URL = "https://data.moa.gov.tw/Service/OpenData/TransService.aspx?UnitId=QcbUEzN6E6DL"
# 上次成功下載的資料，API 連不上時使用；存在程式檔旁邊，不受終端機所在資料夾影響
CACHE_FILE = Path(__file__).with_name("dogs_cache.csv")

# 體型適配：住宅型態與活動時間各分 1～3 級
HOME_LEVEL = {"套房": 1, "公寓大樓": 2, "有院子的住家": 3}
ACTIVITY_LEVEL = {"30 分鐘以下": 1, "30～60 分鐘": 2, "1 小時以上": 3}

FIT_SCORE = {"適合": 90, "尚可": 60}   # 「不建議」的狗不推薦
WAIT_BONUS_MAX = 10                    # 等待加分上限（小於兩級適配分的差距 30 分，不會蓋過適配度）
WAIT_DAYS_FULL = 365                   # 等待滿一年拿到全部加分

# 開放資料的代碼 → 中文（若資料本身已是中文，也一併接受）
SIZE_MAP = {"SMALL": "小型", "MEDIUM": "中型", "BIG": "大型",
            "小型": "小型", "中型": "中型", "大型": "大型"}
AGE_MAP = {"CHILD": "幼犬", "ADULT": "成犬", "幼年": "幼犬", "成年": "成犬"}
YES_NO_MAP = {"T": "是", "F": "否", "N": "未知"}
SEX_MAP = {"M": "公", "F": "母", "N": "未知"}


# ============================================================
# Part 2. 取得資料：呼叫 API，失敗時改讀快取
# ============================================================
def download_records(max_seconds=60):
    """下載 API 資料；超過 max_seconds 秒還沒下載完就放棄（避免連線很慢時一直卡住）。"""
    start = time.time()
    chunks = []
    with requests.get(API_URL, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()                 # HTTP 錯誤碼（404、500…）會丟出例外
        for chunk in response.iter_content(chunk_size=65536):
            chunks.append(chunk)
            if time.time() - start > max_seconds:
                raise TimeoutError(f"超過 {max_seconds} 秒仍未下載完成")
    return json.loads(b"".join(chunks))             # 內容不是 JSON 會丟出 ValueError


def load_data():
    """回傳 (原始資料表, 資料來源說明)。"""
    print("下載開放資料中…", flush=True)            # flush=True：雲端平台的紀錄才會立刻顯示
    try:
        records = download_records()
        if not records:
            raise ValueError("API 回傳空資料")
        raw = pd.DataFrame(records)
        raw.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")   # 成功就更新快取
        print(f"下載完成，共 {len(raw)} 筆", flush=True)
        return raw, f"農業部動物認領養開放資料（{datetime.now():%Y/%m/%d %H:%M} 下載）"
    except (requests.RequestException, ValueError, TimeoutError) as error:
        print("API 下載失敗：", error, flush=True)
        if not CACHE_FILE.exists():
            raise RuntimeError("無法連線開放資料，也沒有快取資料，請確認網路後重新執行。")
        raw = pd.read_csv(CACHE_FILE, dtype=str, encoding="utf-8-sig")
        updated = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        print(f"改用快取資料，共 {len(raw)} 筆", flush=True)
        return raw, f"⚠️ 無法連線開放資料，改用快取資料（{updated:%Y/%m/%d %H:%M} 更新）"


# ============================================================
# Part 3. 整理資料：只留待認養的狗，算出縣市與等待天數
# ============================================================
def clean_data(raw, today=None):
    today = pd.Timestamp(today or datetime.now().date())
    df = raw.copy()

    # 只留「狗」，且狀態為開放認養、今天在開放認養期間內的動物。
    # 依《動物保護法》第 14 條，動物入所後要先等原飼主認領（逾七日未領回才可公告認養），
    # 所以開放認養日還沒到、或認養期限已過的狗，現在不能認養，不顯示。
    opened = pd.to_datetime(df["animal_opendate"], errors="coerce", format="mixed")
    closed = pd.to_datetime(df.get("animal_closeddate", pd.Series(index=df.index, dtype=str)),
                            errors="coerce", format="mixed")
    is_dog = df["animal_kind"].isin(["狗", "犬"])
    is_open = df["animal_status"].isin(["OPEN", "開放認養"])
    in_period = (opened.isna() | (opened <= today)) & (closed.isna() | (closed >= today))
    df = df[is_dog & is_open & in_period].copy()

    df["品種"] = df["animal_Variety"].fillna("").str.strip()      # 原始資料尾端有多餘空白
    df["體型"] = df["animal_bodytype"].map(SIZE_MAP).fillna("未知")
    df["年齡"] = df["animal_age"].map(AGE_MAP).fillna("未知")
    df["性別"] = df["animal_sex"].map(SEX_MAP).fillna("未知")
    df["已絕育"] = df["animal_sterilization"].map(YES_NO_MAP).fillna("未知")
    df["狂犬病疫苗"] = df["animal_bacterin"].map(YES_NO_MAP).fillna("未知")

    # 縣市：收容所地址前 3 個字，「台」統一成「臺」
    df["縣市"] = df["shelter_address"].fillna("").str.strip().str[:3].str.replace("台", "臺")

    # 等待天數：今天 − 資料建立日（約等於入所日）。
    # 不用開放認養日，因為它可能被重新設定，會把等很久的狗算成剛進來；
    # 資料建立日無法辨識時，才改用開放認養日。日期在未來時記為空值。
    created = pd.to_datetime(df["animal_createtime"], errors="coerce", format="mixed")
    days = (today - created.fillna(opened[df.index])).dt.days
    df["等待天數"] = days.where(days >= 0)
    return df


# ============================================================
# Part 4. 媒合計分（自訂函數）
# ============================================================
def size_fit(size, home, activity):
    """依體型、住宅型態、活動時間判斷適配程度：適合／尚可／不建議。"""
    home_lv = HOME_LEVEL[home]
    act_lv = ACTIVITY_LEVEL[activity]
    if size == "小型":
        return "適合"
    if size == "中型":
        if home_lv >= 2 or act_lv >= 2:
            return "適合"
        return "尚可"                               # 住套房且活動時間少於 30 分鐘
    if size == "大型":
        if home_lv == 3 or act_lv == 3:
            return "適合"                           # 有院子，或每天能陪伴 1 小時以上
        if home_lv == 2 and act_lv == 2:
            return "尚可"
        return "不建議"
    return "尚可"                                   # 體型未知：不排除，但不給最高分


def wait_bonus(wait_days):
    """等待越久加分越多，滿一年拿到上限；天數未知不加分。"""
    if pd.isna(wait_days):
        return 0.0
    return round(min(wait_days, WAIT_DAYS_FULL) / WAIT_DAYS_FULL * WAIT_BONUS_MAX, 1)


def score_dogs(candidates, home, activity):
    """用 pandas 一次算出所有候選犬的總分 = 適配分 + 等待加分，回傳依總分由高至低排序的推薦清單。"""
    ranked = candidates.copy()
    # ① 每隻狗的適配：apply 會把「體型」欄的每個值丟進 size_fit()
    ranked["適配"] = ranked["體型"].apply(lambda size: size_fit(size, home, activity))
    # ② 排除「不建議」的狗
    ranked = ranked[ranked["適配"] != "不建議"].copy()
    # ③ 總分 = 適配分（map 對照 90／60）＋ 等待加分
    ranked["等待加分"] = ranked["等待天數"].apply(wait_bonus)
    ranked["總分"] = ranked["適配"].map(FIT_SCORE) + ranked["等待加分"]
    # ④ 全部列出：總分高的在前；同分時等待較久的在前
    return ranked.sort_values(["總分", "等待天數"], ascending=False)


# ============================================================
# Part 5. 呈現結果
# ============================================================
def percent_yes(series):
    """「是」的比例（%），未知不列入分母；全部未知回傳 None。"""
    known = series[series.isin(["是", "否"])]
    return None if known.empty else round((known == "是").mean() * 100)


def make_summary(ranked, county):
    if ranked.empty:   # 候選犬全都「不建議」時，給使用者調整方向
        return (f"### {county}目前沒有適合您住宅與活動時間的狗\n"
                "可以試著增加每日活動時間，或考慮體型較小的狗。")
    vaccine = percent_yes(ranked["狂犬病疫苗"])
    sterile = percent_yes(ranked["已絕育"])
    show = lambda v: "資料不足" if v is None else f"{v}%"
    return (f"### {county}共推薦 {len(ranked)} 隻狗\n"
            f"推薦名單中，狂犬病疫苗施打率 **{show(vaccine)}**、絕育率 **{show(sterile)}**。"
            f"未施打或未絕育的狗，認養後請與收容所或獸醫師確認補做時程。\n\n"
            f"> 認養前請確認租約或社區規約允許飼養犬隻。")


def make_cards(ranked):
    """把推薦清單做成 HTML 卡片；資料來自外部，一律 escape 避免被插入網頁程式碼。
    分數只用來排序，不顯示給使用者；卡片只呈現名次與適配程度（適合／尚可）。"""
    cards = []
    for rank, (_, dog) in enumerate(ranked.iterrows(), start=1):
        e = lambda key: html.escape(str(dog.get(key, "") or ""))
        photo = str(dog.get("album_file", "") or "")
        # 照片網址讀不到時，隱藏圖片，露出底下的「照片無法載入」
        img = (f'<div class="photo"><span>照片無法載入</span>'
               f'<img src="{html.escape(photo)}" alt="" loading="lazy" '
               f'onerror="this.style.display=\'none\'"></div>'
               if photo.startswith("https://") else '<div class="photo"><span>無照片</span></div>')
        days = "未知" if pd.isna(dog["等待天數"]) else f"{int(dog['等待天數'])} 天"
        cards.append(f"""
        <div class="card">
          {img}
          <div class="info">
            <b>推薦第 {rank} 名｜{e('animal_subid')}</b><span class="fit">{e('適配')}</span><br>
            {e('品種')}・{e('體型')}・{e('年齡')}・{e('性別')}・{e('animal_colour')}<br>
            已在收容所等待 {days}<br>
            狂犬病疫苗：{e('狂犬病疫苗')}｜絕育：{e('已絕育')}<br>
            <small>{e('shelter_name')}｜{e('shelter_tel')}</small>
          </div>
        </div>""")
    return CARD_CSS + '<div class="grid">' + "".join(cards) + "</div>"


CARD_CSS = """<style>
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{border:1px solid #ddd;border-radius:10px;overflow:hidden}
.photo{position:relative;height:180px;background:#eee;
  display:flex;align-items:center;justify-content:center;color:#888}
.photo img{position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover}
.info{padding:8px 10px;line-height:1.6}
.fit{float:right;font-size:0.85em;padding:0 8px;border-radius:10px;background:#e5f4e4;color:#2e6b30}
</style>"""


# ============================================================
# 主流程：按下「開始媒合」後執行
# ============================================================
def recommend(county, home, activity, accept_adult):
    # 檢查輸入是否完整
    if not county or not home or not activity or not accept_adult:
        raise gr.Error("請把四個條件都選好再按「開始媒合」。")

    # 篩選：同縣市；不接受成犬時只留幼犬
    candidates = DOGS[DOGS["縣市"] == county]
    if accept_adult == "否":
        candidates = candidates[candidates["年齡"] == "幼犬"]
    if candidates.empty:
        return f"### {county}目前沒有符合條件的待認養犬", ""

    ranked = score_dogs(candidates, home, activity)
    return make_summary(ranked, county), make_cards(ranked)


# 程式啟動時先載入一次資料
RAW, SOURCE = load_data()
DOGS = clean_data(RAW)
COUNTIES = sorted(c for c in DOGS["縣市"].unique() if c)

# ============================================================
# 網頁介面
# ============================================================
# Gradio 預設跟著電腦的深色／淺色模式；這段 JavaScript 在網址加上 ?__theme=light，固定白色底
FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.href = url.href;
  }
}
"""
# Gradio 6 以後 js 參數改放在 launch()，舊版放在 Blocks()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

with gr.Blocks(title="收容犬認養媒合推薦", **({} if GRADIO_6 else {"js": FORCE_LIGHT})) as demo:
    gr.Markdown("# 🐕 收容犬認養媒合推薦\n"
                f"資料來源：{SOURCE}｜目前待認養犬 {len(DOGS)} 隻\n\n"
                "本系統是認養前的自我評估工具，實際認養資格以各收容所公告為準。")
    with gr.Row():
        county_in = gr.Dropdown(COUNTIES, label="居住縣市")
        home_in = gr.Radio(list(HOME_LEVEL), label="住宅型態")
        activity_in = gr.Radio(list(ACTIVITY_LEVEL), label="每日可陪伴外出活動時間")
        adult_in = gr.Radio(["是", "否"], label="能否接受成犬")
    button = gr.Button("開始媒合", variant="primary")
    summary_out = gr.Markdown()
    cards_out = gr.HTML()
    button.click(recommend, [county_in, home_in, activity_in, adult_in], [summary_out, cards_out])

if __name__ == "__main__":
    launch_options = {"js": FORCE_LIGHT} if GRADIO_6 else {}
    # 部署到 Render 等雲端平台時，平台會用環境變數 PORT 指定連接埠，
    # 並需要開放外部連線（0.0.0.0）；在自己電腦執行時沒有 PORT，維持預設
    if "PORT" in os.environ:
        launch_options.update(server_name="0.0.0.0", server_port=int(os.environ["PORT"]))
    print("啟動網頁…", flush=True)
    demo.launch(**launch_options)
