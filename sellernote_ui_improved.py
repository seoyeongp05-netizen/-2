import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 페이지 설정 및 SELLERNOTE 딥그린 테마 스타일
# ---------------------------------------------------------
st.set_page_config(
    page_title="셀러노트 | 발주 프로그램",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .top-banner {
        background-color: #153D34;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    .brand-title { font-size: 1.35rem; font-weight: 800; letter-spacing: 0.04em; }
    .demo-badge {
        font-size: 0.8rem;
        border: 1px solid #78958A;
        border-radius: 20px;
        padding: 4px 12px;
        color: #E1EFE8;
    }
    .exec-badge {
        display: inline-block;
        background-color: #E1EFE8;
        color: #096351;
        border: 1px solid #A9C4BA;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .step-header {
        background-color: #F0F6F3;
        border-left: 4px solid #096351;
        padding: 0.55rem 0.9rem;
        font-weight: 800;
        color: #153D34;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0 0.6rem 0;
    }
    .date-preview {
        background-color: #E6F2ED;
        color: #096351;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 2px;
    }
    .alert-card {
        padding: 0.75rem 0.95rem;
        border-radius: 8px;
        margin-bottom: 0.55rem;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .alert-red { background-color: #FCE5E4; border-left: 5px solid #D93025; color: #7F1D1D; }
    .alert-orange { background-color: #FFEAD8; border-left: 5px solid #F59E0B; color: #78350F; }
    .alert-yellow { background-color: #FFF3CA; border-left: 5px solid #EAB308; color: #713F12; }
    .alert-blue { background-color: #E2ECFF; border-left: 5px solid #2563EB; color: #1E3A8A; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D9E5DF;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

BASE_DATE = datetime.date(2026, 11, 16)

def fmt_md(days: int) -> str:
    target = BASE_DATE + datetime.timedelta(days=int(days))
    return target.strftime("%m.%d")

# ---------------------------------------------------------
# 2. 초기 예시 데이터 구성
# ---------------------------------------------------------
import os

FILE_FUTURE = "SKU별_관측이후_미래수요예측표_A.csv"
FILE_WEEKLY = "SKU별_주차별_수요예측량_A.csv"

def read_csv_safe(path):
    for enc in ["utf-8-sig", "cp949", "utf-8", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
            return df
        except Exception:
            continue
    raise ValueError(f"파일 인코딩을 읽을 수 없습니다: {path}")

@st.cache_data
def load_real_sku_data():
    if not os.path.exists(FILE_FUTURE) or not os.path.exists(FILE_WEEKLY):
        st.error(f"'{FILE_FUTURE}' 또는 '{FILE_WEEKLY}' 파일을 찾을 수 없습니다. GitHub 저장소에 2개 CSV 파일을 업로드해 주세요.")
        st.stop()

    df_fut = read_csv_safe(FILE_FUTURE)
    df_wk = read_csv_safe(FILE_WEEKLY)

    # 1) 미래수요예측표에서 4주 기준 주당 평균 예측 수요 추출
    fut_4w = df_fut[df_fut["예측 기간"].astype(str).str.contains("4")].copy()
    if fut_4w.empty:
        fut_4w = df_fut.groupby("SKU ID", as_index=False).first()
    fut_map = dict(zip(
        fut_4w["SKU ID"].astype(str).str.strip(),
        fut_4w["주당 평균 예측 수요(PCS)"].astype(float)
    ))

    # 2) 주차별 수요예측량에서 52주 실적 및 예측 오차(잔차) 기반 P20/P95 산출
    df_wk["SKU ID"] = df_wk["SKU ID"].astype(str).str.strip()
    df_wk = df_wk.sort_values(["SKU ID", "주 시작일"])

    sku_list = []
    all_skus = list(dict.fromkeys(df_fut["SKU ID"].astype(str).str.strip().tolist()))

    for idx, sid in enumerate(all_skus):
        grp = df_wk[df_wk["SKU ID"] == sid]
        ships = grp["총출고PCS"].fillna(0).tolist() if not grp.empty else [10, 10]
        prev_ship = int(ships[-2]) if len(ships) >= 2 else int(ships[-1])
        curr_ship = int(ships[-1]) if len(ships) >= 1 else 0

        wk_mean = fut_map.get(sid, float(np.mean(ships[-4:])) if ships else 10.0)

        # 실적과 맞춤형 예측치의 잔차 또는 최근 12주 출고 표준편차로 분위수(P20, P95) 계산
        if not grp.empty and "선택_맞춤형예측_PCS" in grp.columns:
            residuals = (grp["총출고PCS"] - grp["선택_맞춤형예측_PCS"]).dropna().tolist()
            wk_std = float(np.std(residuals[-12:])) if len(residuals) >= 2 else wk_mean * 0.3
        else:
            wk_std = float(np.std(ships[-12:])) if len(ships) >= 2 else wk_mean * 0.3

        mean_w = max(1, int(round(wk_mean)))
        p20_w = max(0, int(round(mean_w - 0.84 * wk_std)))
        p95_w = max(mean_w + 1, int(round(mean_w + 1.65 * wk_std)))

        forecast = [[p20_w, mean_w, p95_w] for _ in range(4)]

        # 재고·소비기한·리드타임 시뮬레이션 파라미터
        cur_stock = max(mean_w * 2, curr_ship * 3)
        shelf_life = 120 if idx % 3 != 1 else 35
        sea_lt = 42 if idx % 2 == 0 else 49
        air_lt = 7

        sku_list.append({
            "id": sid,
            "name": f"품목 {sid}",
            "forecast": forecast,
            "ship": [prev_ship, curr_ship],
            # 로트별 입고·소비기한 원장은 제공되지 않아 아래 값은 시연용 가정이다.
            "lots": [{"id": f"{sid}-L1", "qty": cur_stock, "expiry": shelf_life, "deadline": shelf_life}],
            "inbound": [{"mode": "해상", "day": 28, "qty": mean_w * 4, "expiry": shelf_life + 60, "deadline": shelf_life + 60}],
            "b2b": [],
            "planned": mean_w * 4,
            "terms": {"air": air_lt, "sea": sea_lt, "window": max(30, shelf_life - 15), "review": 14, "leadSD": 5, "cycle": 30},
            "costs": {"air": [0, 150000 if idx % 5 == 0 else 0], "sea": [310000, 320000]},
            "records": [{
                "request": -25, "accepted": -24, "depart": -15, "eta": 28,
                "status": "운송 중", "mode": "해상", "route": "해외 선적항 → 한국 · 부산항",
                "container": "20ft LCL", "load": "1.2 CBM", "cost": 320000, "vat": 32000
            }]
        })
    return sku_list

def get_initial_data():
    return load_real_sku_data()


if "sku_data" not in st.session_state:
    st.session_state.sku_data = get_initial_data()
if "events" not in st.session_state:
    st.session_state.events = []
if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()
if "read_alerts" not in st.session_state:
    st.session_state.read_alerts = set()

# ---------------------------------------------------------
# 3. 핵심 시뮬레이션 및 발주 권고 연산 엔진
# ---------------------------------------------------------
def cutoff(lot):
    return min(lot["deadline"], lot.get("expiry", lot["deadline"]))

def daily_demand(sku, day, col):
    weeks = sku["forecast"]
    wk = day // 7
    if weeks:
        row = weeks[wk] if wk < len(weeks) else weeks[-1]
        return row[col] / 7.0
    prev_s, cur_s = sku["ship"]
    return [min(prev_s, cur_s), (prev_s + cur_s) / 2.0, max(prev_s, cur_s)][col] / 7.0

def simulate(sku, col, extra=None, horizon=180):
    extra = extra or []
    cur_lots = [{**l, "deadline": cutoff(l), "day": 0, "current": True} for l in sku["lots"]]
    fut_lots = [{**l, "deadline": cutoff(l), "id": f"입고 {i+1}", "current": False} for i, l in enumerate(sku["inbound"])]
    ext_lots = [{**l, "current": False} for l in extra]
    lots = [{**l, "left": float(l["qty"]), "sold": 0.0, "lost": 0.0, "first": None, "last": None} for l in (cur_lots + fut_lots + ext_lots)]

    arrivals = {}
    for l in lots:
        arrivals.setdefault(l["day"], []).append(l)

    booked = {}
    for o in sku["b2b"]:
        booked[o["day"]] = booked.get(o["day"], 0.0) + float(o["qty"])

    active = []
    total_demand = 0.0
    total_supply = 0.0
    first_gap = None
    lines = []
    miss = []

    for day in range(horizon):
        next_active = []
        for l in active:
            if l["deadline"] > day and l["left"] > 1e-7:
                next_active.append(l)
            elif l["deadline"] <= day:
                l["lost"] = l["left"]
                total_supply -= l["left"]
                l["left"] = 0.0
        active = next_active

        for l in arrivals.get(day, []):
            if l["deadline"] <= day:
                l["lost"] = l["left"]
                l["left"] = 0.0
                continue
            active.append(l)
            total_supply += l["left"]

        active.sort(key=lambda x: (x["deadline"], x["day"]))
        need = daily_demand(sku, day, col) + booked.get(day, 0.0)
        total_demand += need

        for l in active:
            if need <= 1e-7:
                break
            take = min(need, l["left"])
            if take <= 1e-7:
                continue
            if l["first"] is None:
                l["first"] = day
            l["last"] = day
            l["sold"] += take
            l["left"] -= take
            need -= take

        if need > 1e-7 and first_gap is None:
            first_gap = day
        miss.append(max(0.0, need))
        lines.append((total_demand, total_supply))

    return {"lots": lots, "lines": lines, "miss": miss, "firstGap": first_gap}

def evaluate_all(data_list):
    t0 = time.perf_counter()
    results = []
    for sku in data_list:
        t = sku["terms"]
        air = t["air"]
        sea_days = [l["day"] for l in sku["inbound"] if l["mode"] == "해상"]
        sea_day = min(sea_days) if sea_days else t["sea"]

        max_d = max(
            [1]
            + [l["deadline"] for l in sku["lots"]]
            + [l["deadline"] for l in sku["inbound"]]
            + [t["sea"] + t["window"], sea_day + 1, air + 1]
        ) + 1

        p20 = simulate(sku, 0, horizon=max_d)
        mean = simulate(sku, 1, horizon=max_d)
        p95 = simulate(sku, 2, horizon=max_d)

        cand = {"id": "신규 해상 발주", "qty": 1e12, "day": t["sea"], "deadline": t["sea"] + t["window"]}
        cap_sim = simulate(sku, 0, extra=[cand], horizon=max_d)
        cap = max(0, int(np.floor(cap_sim["lots"][-1]["sold"] + 1e-7)))

        waste_lots = [l for l in p20["lots"] if l["current"]]
        waste = int(np.ceil(sum(l["lost"] for l in waste_lots) - 1e-7))

        first = p95["firstGap"] if (p95["firstGap"] is not None and p95["firstGap"] < sea_day) else None
        air_qty = 0 if (first is None or air >= sea_day) else int(np.ceil(sum(p95["miss"][air:sea_day]) - 1e-7))

        day_mean = sum(daily_demand(sku, d, 1) for d in range(7)) / 7.0
        booked_lt = sum(o["qty"] for o in sku["b2b"] if o["day"] < t["sea"] + t["review"])
        pos = sum(l["qty"] for l in sku["lots"] if cutoff(l) > 0) + sum(
            l["qty"] for l in sku["inbound"] if l["day"] < t["sea"] + t["review"] and cutoff(l) > l["day"]
        )
        sd_part = max(0.0, daily_demand(sku, 0, 2) - day_mean) / 1.65 * np.sqrt(t["sea"] + t["review"])
        safety = 1.65 * np.hypot(sd_part, day_mean * t["leadSD"])
        reorder = day_mean * (t["sea"] + t["review"]) + safety + booked_lt
        raw = max(0, int(np.floor(reorder + t["cycle"] * day_mean - pos))) if pos <= reorder else 0

        results.append({
            "sku": sku, "air": air, "seaDay": sea_day,
            "p20": p20, "mean": mean, "p95": p95,
            "cap": cap, "waste": waste, "wasteLots": waste_lots,
            "first": first, "airQty": air_qty, "recommend": min(raw, cap)
        })
    elapsed = time.perf_counter() - t0
    return results, elapsed

results, elapsed_sec = evaluate_all(st.session_state.sku_data)

# ---------------------------------------------------------
# 4. 상단 배너 및 SKU 선택
# ---------------------------------------------------------
st.markdown(f"""
<div class="top-banner">
    <div>
        <div class="brand-title">SELLERNOTE · 발주 프로그램</div>
        <div style="font-size:0.85rem; color:#CBE0D7; margin-top:2px;">기준일 {BASE_DATE.strftime('%Y-%m-%d')} · 관리 대상 SKU {len(st.session_state.sku_data)}개</div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
        <span class="exec-badge"> {len(results)}개 SKU 계산 완료 · {elapsed_sec:.4f}초 ({elapsed_sec*1000:.1f}ms)</span>
        <span class="demo-badge">발주 검토 대시보드</span>
    </div>
</div>
""", unsafe_allow_html=True)

default_id = "S2036649" if any(r["sku"]["id"] == "S2036649" for r in results) else results[0]["sku"]["id"]

sku_query = st.text_input(
    f"검토할 SKU ID 직접 입력 (총 {len(results)}개 SKU 로드됨)",
    value=default_id,
    placeholder="예: S2036649"
).strip()

matched_res = next(
    (r for r in results if sku_query.upper() == r["sku"]["id"].upper()),
    next((r for r in results if sku_query.upper() in r["sku"]["id"].upper()), None)
)

if matched_res is None:
    st.warning(f"입력하신 '{sku_query}'와 일치하는 SKU ID가 없습니다. 기본 SKU({default_id})를 표시합니다.")
    active_res = next(r for r in results if r["sku"]["id"] == default_id)
else:
    active_res = matched_res

selected_sku_id = active_res["sku"]["id"]
active_sku = active_res["sku"]

# ---------------------------------------------------------
# 5. 사이드바: 담당자 4단계 예외 기반 알림 센터
# ---------------------------------------------------------

with st.sidebar:
    st.header("담당자 알림 시스템")
    st.caption("경보 피로(Alert Fatigue) 방지를 위해 즉시 푸시와 일일 요약을 분리합니다.")

    with st.expander("4단계 경고 발동 기준 안내", expanded=False):
        st.markdown("""
        <div style="font-size:0.82rem; line-height:1.55;">
            <div style="margin-bottom:6px;">
                <strong style="color:#B91C1C;">🔴 긴급 (즉시 푸시)</strong><br>
                • 긴급 항공 전환 경보 (핵심 품목·고위험 부족 시 즉시 발송)
            </div>
            <div style="margin-bottom:6px;">
                <strong style="color:#1D4ED8;">🔵 변동 (즉시 푸시)</strong><br>
                • 영업팀 신규 확정 B2B 대량 주문 등록 시<br>
                • 포워더/물류 시스템 해상·항공 입항 예정일 지연 갱신 시
            </div>
            <div style="margin-bottom:6px;">
                <strong style="color:#C2410C;">🟠 주의 (일일 요약)</strong><br>
                • 판매 속도 둔화로 폐기 물량 발생 예상 시<br>
                • 품절 임박 SKU 존재 시
            </div>
            <div>
                <strong style="color:#A16207;">🟡 검토 (일일 요약)</strong><br>
                • 담당자 입력 발주량이 신규 발주 상한 초과 시
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 출고량 상위 20% 핵심 SKU(A등급) 선별
    sorted_by_vol = sorted(results, key=lambda x: x["sku"]["ship"][1], reverse=True)
    top_a_ids = {r["sku"]["id"] for r in sorted_by_vol[:max(1, len(results) // 5)]}

    all_alerts = []
    for r in results:
        sid, sname = r["sku"]["id"], r["sku"]["name"]
        is_core = sid in top_a_ids
        if r["airQty"] > 0:
            # 핵심 A등급이면서 부족량이 100개 이상인 경우만 즉시 푸시 대상 지정
            push_now = is_core and r["airQty"] >= 100
            all_alerts.append((0, "긴급", sid, sname, "긴급 항공 투입 검토", f"항공 추가 {r['airQty']:,}개 · 최초 부족 {fmt_md(r['first'])}", push_now))
        if r["waste"] > 0:
            all_alerts.append((2, "주의", sid, sname, "판매 마감 시 잔량 예상", f"현재 로트 P20 기준 {r['waste']:,}개 만료 예상", False))
        if r["first"] is not None:
            all_alerts.append((2, "주의", sid, sname, "해상 입고 전 품절 예상", f"{fmt_md(r['first'])} ({r['first']//7 + 1}주차) 품절 예상", False))
        if r["sku"]["planned"] > r["cap"]:
            all_alerts.append((3, "검토", sid, sname, "입력 발주량 상한 초과", f"{r['sku']['planned']:,}개 입력 / {r['cap']:,}개 상한", False))

    for ev in st.session_state.events:
        # 외부 확정 이벤트(변동)는 즉시 푸시 대상
        all_alerts.append((1, "변동", ev["sku"], ev["name"], ev["title"], ev["detail"], True))

    all_alerts.sort(key=lambda x: x[0])

    filter_mode = st.radio(
        "알림 표시 범위",
        ["즉시 발송 대상만 보기 (핵심 긴급 + 변동)", "전체 감지 경보 보기"],
        index=0
    )

    visible_alerts = [a for a in all_alerts if a[6]] if "즉시 발송" in filter_mode else all_alerts
    digest_cnt = sum(1 for a in all_alerts if not a[6])

    sc1, sc2 = st.columns(2)
    if sc1.button("즉시 푸시 발송", use_container_width=True):
        push_targets = [i for i, a in enumerate(all_alerts) if a[6]]
        for idx in push_targets:
            st.session_state.sent_alerts.add(idx)
        st.success(f"즉시 발송 대상 {len(push_targets)}건 전송 완료")
    if sc2.button("일일 요약 발송", use_container_width=True):
        st.info(f"주의·검토 {digest_cnt}건 일일 리포트 발송 완료")

    st.caption(f"표시 중: **{len(visible_alerts)}건** (일일 요약 대기 {digest_cnt}건 제외)")
    st.divider()

    # 최대 5건만 카드로 렌더링해 스크롤 과부하 방지
    max_show = 5
    for idx, (lvl, sev, sid, sname, title, detail, push_now) in enumerate(visible_alerts[:max_show]):
        cls_map = {0: "alert-red", 1: "alert-blue", 2: "alert-orange", 3: "alert-yellow"}
        icon_map = {0: "🔴", 1: "🔵", 2: "🟠", 3: "🟡"}
        badge_txt = "즉시푸시" if push_now else "일일요약"
        st.markdown(f"""
        <div class="alert-card {cls_map[lvl]}">
            <strong>{icon_map[lvl]} [{sev}] {title}</strong> · <span style="font-size:0.78rem;">({badge_txt})</span><br>
            <span style="font-weight:700;">{sid} ({sname})</span><br>
            {detail}
        </div>
        """, unsafe_allow_html=True)
        if st.button("추가 보기", key=f"alert_detail_{idx}_{sid}", use_container_width=True):
            st.session_state.alert_detail_sku = sid

    if len(visible_alerts) > max_show:
        st.info(f"외 **{len(visible_alerts) - max_show}건**이 더 있습니다. 알림 표시 범위를 변경해 확인하세요.")
# ---------------------------------------------------------
# 6. 메인 탭 구성
# ---------------------------------------------------------
tab1, tab4 = st.tabs([
    "종합 대시보드",
    "SKU별 발주 상태"
])

# =========================================================
# [탭 1] 종합 대시보드
# =========================================================
with tab1:
    st.subheader(f"발주 권고 · {active_sku['id']}")
    order_day = BASE_DATE if active_res["recommend"] > 0 else None
    order_c1, order_c2, order_c3 = st.columns(3)
    order_c1.metric("권고 발주량", f"{active_res['recommend']:,}개")
    order_c2.metric("담당자 입력 발주량", f"{active_sku['planned']:,}개")
    order_c3.metric("권고 발주시점", order_day.strftime("%Y.%m.%d (오늘)") if order_day else "현재 발주 불필요")
    st.caption("선택한 SKU 기준 · 입력값을 반영해 권고량을 다시 계산합니다. 담당자가 입력한 희망량은 권고량과 별도로 표시합니다. 발주시점은 재주문점 충족 시 기준일입니다.")
    st.warning("로트별 재고·소비기한과 예정 입고 데이터는 제공된 원장으로 확인되지 않았습니다. 현재 값은 시연용 가정이며, 실제 발주 전 담당자의 확인이 필요합니다.")

    st.divider()

    # 직관적인 발주 조건 및 변동 시뮬레이션 입력창
    col_in_head, col_reset = st.columns([4, 1])
    with col_in_head:
        st.subheader(f"발주 조건 및 변동 시뮬레이션 입력 — [{active_sku['id']} {active_sku['name']}]")
        st.caption("기준일(2026.11.16)로부터 며칠 뒤인지(D+일)를 입력하면 실제 달력 날짜가 옆에 자동으로 표시됩니다.")
    with col_reset:
        if st.button("초기 예시값 복원", use_container_width=True):
            st.session_state.sku_data = get_initial_data()
            st.session_state.events = []
            st.session_state.sent_alerts = set()
            st.session_state.read_alerts = set()
            st.session_state.alert_detail_sku = None
            for key in list(st.session_state):
                if key.startswith(("lqty_", "lexp_", "lend_", "iqty_", "iday_", "iend_")):
                    del st.session_state[key]
            st.rerun()

    with st.form("intuitive_order_form"):
        st.markdown('<div class="step-header">1단계 · 현재 보유 재고(로트) 및 예정 입고 현황 수정</div>', unsafe_allow_html=True)

        lot_inputs = []
        for idx, lot in enumerate(active_sku["lots"]):
            lc1, lc2, lc3, lc4 = st.columns([1.2, 1.2, 1.3, 1.3])
            lc1.markdown(f"**현재 재고 · {lot['id']}**<br><span style='font-size:0.8rem; color:#096351;'>창고 보유중</span>", unsafe_allow_html=True)
            l_qty = lc2.number_input(f"{lot['id']} 가용 수량 (개)", min_value=0, max_value=1000000, value=int(lot["qty"]), key=f"lqty_{active_sku['id']}_{idx}")
            l_exp = lc3.number_input(f"{lot['id']} 소비기한 만료 (D+일)", min_value=0, max_value=730, value=int(lot.get("expiry", lot["deadline"])), key=f"lexp_{active_sku['id']}_{idx}")
            lc3.markdown(f'<span class="date-preview">만료일: {fmt_md(l_exp)}</span>', unsafe_allow_html=True)
            l_end = lc4.number_input(f"{lot['id']} 실제 판매 마감 (D+일)", min_value=0, max_value=730, value=int(lot["deadline"]), key=f"lend_{active_sku['id']}_{idx}")
            lc4.markdown(f'<span class="date-preview">마감일: {fmt_md(l_end)}</span>', unsafe_allow_html=True)
            lot_inputs.append((idx, l_qty, l_exp, l_end))

        inb_inputs = []
        for idx, inb in enumerate(active_sku["inbound"]):
            ic1, ic2, ic3, ic4 = st.columns([1.2, 1.2, 1.3, 1.3])
            ic1.markdown(f"**예정 입고 {idx+1} ({inb['mode']})**<br><span style='font-size:0.8rem; color:#506762;'>기존 예정: {fmt_md(inb['day'])}</span>", unsafe_allow_html=True)
            i_qty = ic2.number_input(f"입고 {idx+1} 수량 (개)", min_value=0, max_value=1000000, value=int(inb["qty"]), key=f"iqty_{active_sku['id']}_{idx}")
            i_day = ic3.number_input(f"입고 {idx+1} 도착 예정 (D+일)", min_value=0, max_value=730, value=int(inb["day"]), key=f"iday_{active_sku['id']}_{idx}")
            ic3.markdown(f'<span class="date-preview">도착일: {fmt_md(i_day)}</span>', unsafe_allow_html=True)
            i_end = ic4.number_input(f"입고 {idx+1} 판매 마감 (D+일)", min_value=0, max_value=1095, value=int(inb["deadline"]), key=f"iend_{active_sku['id']}_{idx}")
            ic4.markdown(f'<span class="date-preview">마감일: {fmt_md(i_end)}</span>', unsafe_allow_html=True)
            inb_inputs.append((idx, i_qty, i_day, i_end))

        b2b_str = ", ".join([f"{fmt_md(o['day'])} ({o['qty']:,}개)" for o in active_sku["b2b"]]) or "없음"
        st.info(f"현재 등록된 확정 B2B 주문: **{b2b_str}**")

        st.markdown('<div class="step-header">2단계 · 금주 신규 발주 희망량 및 돌발 변수(확정 B2B · 입고 지연) 입력</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        in_planned = s1.number_input("금주 신규 발주 희망 수량 (개)", min_value=0, max_value=1000000, value=int(active_sku["planned"]), help="문제 2의 최대 발주 상한 초과 여부를 자동 검증합니다.")
        in_b2b_qty = s2.number_input("추가 확정 B2B 주문량 (개)", min_value=0, max_value=1000000, value=0, help="기존 주간 예측에 포함되지 않은 신규 확정 대량 주문만 입력하세요.")
        in_b2b_day = s3.number_input("확정 B2B 출고 예정일 (D+일)", min_value=0, max_value=365, value=4)
        s3.markdown(f'<span class="date-preview">출고일: {fmt_md(in_b2b_day)}</span>', unsafe_allow_html=True)
        in_delay = s4.number_input("가장 빠른 예정 입고 지연 (+일)", min_value=0, max_value=365, value=0, help="선적·통관 지연 시 가장 빠른 예정 입고일이 뒤로 밀립니다.")

        submitted = st.form_submit_button("입력값 반영 및 발주 권고 재계산", use_container_width=True, type="primary")
        if submitted:
            active_sku["planned"] = int(in_planned)
            for idx, l_qty, l_exp, l_end in lot_inputs:
                active_sku["lots"][idx]["qty"] = int(l_qty)
                active_sku["lots"][idx]["expiry"] = int(l_exp)
                active_sku["lots"][idx]["deadline"] = int(l_end)

            for idx, i_qty, i_day, i_end in inb_inputs:
                old_day = active_sku["inbound"][idx]["day"]
                active_sku["inbound"][idx]["qty"] = int(i_qty)
                active_sku["inbound"][idx]["day"] = int(i_day)
                active_sku["inbound"][idx]["deadline"] = int(i_end)
                if i_day > old_day:
                    st.session_state.events.append({
                        "sku": active_sku["id"], "name": active_sku["name"],
                        "title": f"{active_sku['inbound'][idx]['mode']} ETA 지연",
                        "detail": f"{fmt_md(old_day)} → {fmt_md(i_day)} ({i_day - old_day}일 지연)"
                    })

            if in_delay > 0 and active_sku["inbound"]:
                earliest = min(active_sku["inbound"], key=lambda x: x["day"])
                old_d = earliest["day"]
                earliest["day"] += int(in_delay)
                st.session_state.events.append({
                    "sku": active_sku["id"], "name": active_sku["name"],
                    "title": f"{earliest['mode']} 입항 지연 반영",
                    "detail": f"{fmt_md(old_d)} → {fmt_md(earliest['day'])} (+{in_delay}일 지연)"
                })

            if in_b2b_qty > 0:
                active_sku["b2b"].append({"day": int(in_b2b_day), "qty": int(in_b2b_qty)})
                st.session_state.events.append({
                    "sku": active_sku["id"], "name": active_sku["name"],
                    "title": "확정 B2B 대량 주문 등록",
                    "detail": f"{fmt_md(in_b2b_day)} 출고 · {int(in_b2b_qty):,}개 추가"
                })

            st.rerun()

    st.divider()

    # SKU별 발주 권고 결과표
    # 알림의 추가 보기에서 해당 SKU의 권고 결과를 펼친다.
    def get_rank(x):
        if x["airQty"] > 0:
            return 0
        if x["waste"] > 0 or x["first"] is not None:
            return 1
        if x["sku"]["planned"] > x["cap"]:
            return 2
        return 3

    rank_labels = {0: "🔴 긴급", 1: "🟠 주의", 2: "🟡 검토", 3: "🟢 정상"}
    sorted_res = sorted(results, key=get_rank)

    table_rows = []
    for r in sorted_res:
        rk = get_rank(r)
        table_rows.append({
            "긴급 순위": rank_labels[rk],
            "SKU ID": r["sku"]["id"],
            "품목명": r["sku"]["name"],
            "권고 발주량": f"{r['recommend']:,}개",
            "만료 예상 폐기량": f"{r['waste']:,}개",
            "신규 발주 상한": f"{r['cap']:,}개",
            "최초 품절 예상": "품절 없음" if r["first"] is None else f"{r['first']//7 + 1}주차 ({fmt_md(r['first'])})",
            "항공 긴급 투입량": f"{r['airQty']:,}개"
        })

    if st.session_state.get("alert_detail_sku"):
        detail_sku = st.session_state.alert_detail_sku
        st.subheader(f"알림 상세 · {detail_sku} 발주 권고 결과")
        st.dataframe(pd.DataFrame([row for row in table_rows if row["SKU ID"] == detail_sku]), use_container_width=True, hide_index=True)
        if st.button("결과표 닫기"):
            st.session_state.alert_detail_sku = None
            st.rerun()
    with st.expander("전체 SKU 발주 권고 결과표"):
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# SKU별 발주 상태
with tab4:
    st.subheader(f"세부 SKU별 발주·국제 물류 상태 — [{active_sku['id']} {active_sku['name']}]")
    rec_rows = []
    for o in active_sku["records"]:
        rec_rows.append({
            "의뢰일": fmt_md(o["request"]),
            "수락일": fmt_md(o["accepted"]),
            "운송 시작": fmt_md(o["depart"]),
            "도착 예정": fmt_md(o["eta"]),
            "상태 / 방식": f"{o['status']} / {o['mode']}",
            "출발지 → 도착지": o["route"],
            "컨테이너 / 적재량": f"{o['container']} / {o['load']}",
            "운송비": f"{o['cost']:,}원",
            "VAT": f"{o['vat']:,}원",
            "의뢰→도착 리드타임": f"{o['eta'] - o['request']}일"
        })
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
