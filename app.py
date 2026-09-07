import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

st.set_page_config(page_title="จัดพอร์ตหุ้นฉบับนักเรียน", page_icon="📈", layout="wide")

RF_RATE = 0.0

# ------------------------------------------------------------------
# ส่วนคำนวณ
# ------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(tickers, years_back):
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    raw = raw.ffill()
    valid = [t for t in tickers if t in raw.columns and raw[t].notna().sum() >= 30]
    return raw[valid].dropna(), valid


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(years_back):
    """ดัชนี SET Index จริง ใช้เป็นเกณฑ์เทียบภายนอก (ไม่ใช่ S&P 500 เพราะหุ้นที่วิเคราะห์เป็นหุ้นไทย)"""
    import yfinance as yf
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years_back * 365)).strftime("%Y-%m-%d")
    try:
        raw = yf.download("^SET.BK", start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        return raw.dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_caps(tickers):
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("sharesOutstanding")
        except Exception:
            caps[t] = None
    return caps


def _clean_shares(v):
    """None และ NaN ทั้งคู่ถือว่า 'ไม่มีข้อมูล' -- ป้องกัน NaN แอบหลุดเข้าไปคำนวณ (NaN เป็น truthy ใน Python
    เช็คด้วย `if v` เฉยๆ จะจับ NaN ไม่ได้)"""
    if v is None:
        return None
    try:
        return None if np.isnan(v) else v
    except TypeError:
        return v


def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap):
    mu, cov = train_ret.mean(), train_ret.cov()
    w_equal = np.array([1 / n] * n)

    def neg_sharpe(w, mu, cov):
        ret = np.sum(mu * w) * 252 - RF_RATE
        vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        return -ret / vol if vol > 1e-10 else 1e6

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    opt = minimize(neg_sharpe, w_equal, args=(mu.values, cov.values),
                   method="SLSQP", bounds=bounds, constraints=constraints)
    w_markowitz = opt.x if opt.success else w_equal

    weights = {"A: Equal-weight (จัดเงินเท่ากัน)": w_equal, "B: Markowitz (สมการคณิตศาสตร์)": w_markowitz}
    if use_marketcap:
        mcap = train_last_price * shares_arr
        weights["C: Market-cap (ตามขนาดบริษัท)"] = mcap / mcap.sum()
    return weights, opt.success


def evaluate(w, test_ret):
    port_ret = test_ret.values @ w
    cum = np.cumprod(1 + port_ret)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 1e-10 else np.nan
    running_max = np.maximum.accumulate(cum)
    max_dd = ((cum - running_max) / running_max).min()
    return cum[-1] - 1, ann_ret, ann_vol, sharpe, max_dd


def run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window):
    all_returns = data.pct_change().dropna()
    n = data.shape[1]
    records, failed = [], 0
    daily_returns = {}
    last_weights = {}
    start, round_num = 0, 0
    while start + train_window + test_window <= len(all_returns):
        round_num += 1
        train_ret = all_returns.iloc[start: start + train_window]
        test_ret = all_returns.iloc[start + train_window: start + train_window + test_window]
        # แก้บั๊ก off-by-one: ราคาสุดท้ายที่ "จริงๆ" อยู่ในช่วง train คือ index [start+train_window]
        # ไม่ใช่ [start+train_window-1] (ซึ่งจะเป็นราคาของวันก่อนหน้านั้นแทน)
        train_last_price = data.iloc[start + train_window].values
        weights, ok = get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap)
        failed += 0 if ok else 1
        last_weights = weights
        for name, w in weights.items():
            total, ar, av, sh, mdd = evaluate(w, test_ret)
            records.append({
                "round": round_num,
                "strategy": name,
                "total_return": total,
                "ann_return": ar,
                "ann_vol": av,
                "sharpe": sh,
                "max_drawdown": mdd
            })
            port_ret_series = pd.Series(test_ret.values @ w, index=test_ret.index)
            daily_returns.setdefault(name, []).append(port_ret_series)
        start += test_window
    daily_returns = {k: pd.concat(v) for k, v in daily_returns.items()}
    return pd.DataFrame(records), round_num, failed, daily_returns, last_weights


def cumulative_growth(daily_returns, capital, cost_pct, test_window):
    curves = {}
    for name, ret_series in daily_returns.items():
        equity, values = capital, []
        for i, r in enumerate(ret_series.values):
            if i > 0 and i % test_window == 0:
                equity *= (1 - cost_pct)
            equity *= (1 + r)
            values.append(equity)
        curves[name] = pd.Series(values, index=ret_series.index)
    return curves


def efficient_frontier_fig(all_returns, weights_by_name, n_portfolios=2500):
    mu, cov = all_returns.mean().values, all_returns.cov().values
    n = len(mu)
    rng = np.random.default_rng(0)
    res = np.zeros((n_portfolios, 3))
    for i in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        res[i] = [port_vol, port_ret, port_ret / port_vol if port_vol > 1e-10 else 0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sc = ax.scatter(res[:, 0] * 100, res[:, 1] * 100, c=res[:, 2], cmap="viridis", s=6, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Sharpe ratio")

    markers = list("o*D^v<>")
    for idx, (name, w) in enumerate(weights_by_name.items()):
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        ax.scatter(port_vol * 100, port_ret * 100, marker=markers[idx % len(markers)], s=200,
                   edgecolor="black", linewidth=1.3, label=name.split(" ")[0], zorder=5)

    ax.set_xlabel("Volatility / ความผันผวนต่อปี (%)")
    ax.set_ylabel("Expected Return / ผลตอบแทนคาดหวังต่อปี (%)")
    ax.set_title("Efficient Frontier")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("📈 เว็บแอปจัดพอร์ตหุ้นฉบับนักเรียน")
st.caption(
    "เปรียบเทียบ 3 วิธีแบ่งเงินลงทุนในหุ้น: แบ่งเท่ากัน (Equal-weight), "
    "คำนวณสัดส่วนที่เหมาะสมที่สุดด้วยสูตร Markowitz, และถ่วงน้ำหนักตามมูลค่าบริษัท (Market-cap) "
    "ทดสอบด้วยข้อมูลราคาหุ้นจริงย้อนหลัง แบบ walk-forward validation"
)

with st.sidebar:
    st.header("ตั้งค่า")
    capital = st.number_input("เงินลงทุนเริ่มต้น (บาท)", min_value=1000, value=5000, step=500)
    ticker_input = st.text_input(
        "พิมพ์รหัสหุ้น (คั่นด้วยจุลภาค ,)",
        value="PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK",
        help="ใส่รหัสหุ้น เช่น PTT.BK, CPALL.BK สำหรับหุ้นไทย หรือ AAPL, TSLA สำหรับหุ้นสหรัฐฯ"
    )
    selected = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))

    with st.expander("ตั้งค่าขั้นสูง (ไม่บังคับ)"):
        train_window = st.slider("ช่วง train (วันทำการ)", 126, 378, 252, step=21,
                                  help="จำนวนวันย้อนหลังที่ใช้คำนวณสัดส่วนก่อนแต่ละรอบทดสอบ")
        test_window = st.slider("ช่วง test ต่อรอบ (วันทำการ)", 21, 126, 63, step=21,
                                 help="จำนวนวันที่ใช้ทดสอบสัดส่วนที่คำนวณได้ ต่อ 1 รอบ")
        years_back = st.slider("ข้อมูลย้อนหลังกี่ปี", 3, 10, 6)
        cost_pct = st.slider("ค่าธรรมเนียมการซื้อขายต่อการปรับสมดุลพอร์ต (%)", 0.0, 1.0, 0.1, step=0.05,
                              help="หักออกจากมูลค่าพอร์ตทุกครั้งที่ปรับสัดส่วนใหม่ ในกราฟเงินโตสะสม") / 100
        stress_option = st.selectbox(
            "ทดสอบเฉพาะช่วงวิกฤต (ไม่บังคับ)",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2563)", "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)"],
            help="ประเมินด้วยสัดส่วนล่าสุดที่คำนวณได้ ว่าถ้าเจอเฉพาะช่วงนี้ผลจะเป็นอย่างไร",
        )

    run = st.button("🚀 เริ่มวิเคราะห์", type="primary", use_container_width=True)
    st.caption("ข้อมูลราคาหุ้นดึงสดจาก Yahoo Finance ทุกครั้งที่กดรัน (แคชไว้ 1 ชั่วโมง)")

if not run:
    st.info("ตั้งค่าทางซ้าย แล้วกด **เริ่มวิเคราะห์** เพื่อดูผลเปรียบเทียบ")
    st.stop()

if len(selected) < 2:
    st.error("กรุณาเลือกหุ้นอย่างน้อย 2 ตัว")
    st.stop()

with st.spinner("กำลังดึงราคาหุ้นย้อนหลัง..."):
    try:
        data, valid_tickers = load_price_data(tuple(selected), years_back)
    except Exception as e:
        st.error(f"ดึงข้อมูลราคาหุ้นไม่สำเร็จ: {e}")
        st.stop()

missing = [t for t in selected if t not in valid_tickers]
if missing:
    st.warning(f"หารหัสหุ้นนี้ไม่เจอ หรือข้อมูลไม่พอ เลยตัดออก: {', '.join(missing)}")
selected = valid_tickers

if len(selected) < 2:
    st.error("เหลือหุ้นที่ใช้ได้น้อยกว่า 2 ตัว กรุณาตรวจสอบรหัสหุ้นแล้วลองใหม่")
    st.stop()

if data.empty or len(data) < train_window + test_window * 3:
    st.error("ข้อมูลย้อนหลังไม่พอสำหรับตั้งค่านี้ ลองลดช่วง train/test หรือเพิ่มจำนวนปีย้อนหลังในตั้งค่าขั้นสูงดู")
    st.stop()

with st.spinner("กำลังตรวจสอบมูลค่าตลาด (สำหรับกลยุทธ์ Market-cap)..."):
    caps_raw = load_market_caps(tuple(selected))
    clean_caps = {t: _clean_shares(caps_raw.get(t)) for t in selected}
    use_marketcap = all(clean_caps[t] is not None for t in selected)
    if not use_marketcap:
        st.warning("ดึงข้อมูลมูลค่าตลาดของบางบริษัทไม่สำเร็จ — แสดงผลเฉพาะกลยุทธ์ Equal-weight และ Markowitz")
    shares_arr = np.array([clean_caps[t] or 0 for t in selected])

with st.spinner("กำลังรัน walk-forward validation..."):
    df, n_folds, n_failed, daily_returns, last_weights = run_walk_forward(
        data, shares_arr, use_marketcap, train_window, test_window
    )

if df.empty:
    st.error("ไม่สามารถรันได้ครบแม้แต่ 1 รอบ ลองลดค่า train/test window ในตั้งค่าขั้นสูงดู")
    st.stop()

strategies = list(df["strategy"].unique())
st.success(f"วิเคราะห์เสร็จแล้ว — ทดสอบทั้งหมด {n_folds} รอบ (walk-forward validation)")
if n_failed:
    st.caption(f"หมายเหตุ: การหาค่าเหมาะสมที่สุดไม่ลู่เข้าใน {n_failed}/{n_folds} รอบ (ใช้ equal-weight แทนในรอบนั้น)")

summary = df.groupby("strategy")[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean().reindex(strategies)
best_strategy = summary["sharpe"].idxmax()
best_return = summary.loc[best_strategy, "ann_return"]
best_vol = summary.loc[best_strategy, "ann_vol"]
best_sharpe = summary.loc[best_strategy, "sharpe"]
best_mdd = summary.loc[best_strategy, "max_drawdown"]

# ---------- เช็คนัยสำคัญทางสถิติของ "ผู้ชนะ" ก่อนใช้โทนมั่นใจ ----------
wide = df.pivot(index="round", columns="strategy", values="sharpe")
significantly_better_than_all = True
for other in strategies:
    if other == best_strategy:
        continue
    _, p_val = stats.ttest_rel(wide[best_strategy], wide[other])
    if p_val >= 0.05:
        significantly_better_than_all = False
        break

st.subheader("สรุปผล: กลยุทธ์ที่ Sharpe ดีที่สุดในการทดสอบนี้")
if significantly_better_than_all:
    st.success(
        f"**{best_strategy}** ให้ Sharpe ratio ดีที่สุด และแตกต่างจากกลยุทธ์อื่นทุกตัวอย่างมีนัยสำคัญทางสถิติ (p < 0.05) "
        f"ในการทดสอบ {n_folds} รอบนี้"
    )
else:
    st.info(
        f"**{best_strategy}** ให้ Sharpe ratio เฉลี่ยสูงสุดในการทดสอบนี้ แต่ยัง**ไม่ต่างจากกลยุทธ์อื่นอย่างมีนัยสำคัญทางสถิติ** "
        f"(ดูตาราง p-value ในส่วนวิเคราะห์เชิงลึกด้านล่าง) — ควรตีความว่า \"ยังแยกไม่ออกชัดเจนว่าวิธีไหนดีกว่าจริง\" มากกว่าฟันธงว่าตัวนี้ชนะ"
    )

# ---------- ตัวอย่างการจัดสรรเงิน ----------
st.markdown("#### ตัวอย่างการจัดสรรเงินตามกลยุทธ์นี้")
winner_weights = last_weights[best_strategy]
action_plan = []
for ticker, weight in zip(selected, winner_weights):
    action_plan.append({
        "รหัสหุ้น": ticker,
        "สัดส่วน (%)": f"{weight * 100:.2f}%",
        "จำนวนเงิน (บาท)": f"{capital * weight:,.2f}",
    })
st.dataframe(pd.DataFrame(action_plan), use_container_width=True, hide_index=True)
st.caption("ตัวเลขนี้มาจากสัดส่วนของรอบล่าสุดที่คำนวณได้ ใช้เพื่อประกอบการอธิบายวิธีการเท่านั้น ไม่ใช่คำแนะนำการลงทุนจริง")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("ผลตอบแทนเฉลี่ยต่อปี", f"{best_return * 100:.1f}%")
with col_m2:
    st.metric("ความผันผวนต่อปี", f"{best_vol * 100:.1f}%")
with col_m3:
    st.metric("Sharpe ratio", f"{best_sharpe:.2f}")
with col_m4:
    st.metric("Max Drawdown", f"{best_mdd * 100:.1f}%")

st.markdown("#### ตารางเปรียบเทียบทั้ง 3 กลยุทธ์ (ค่าเฉลี่ยตลอดทุกรอบ)")
display_summary = summary.copy()
display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
display_summary["sharpe"] = display_summary["sharpe"].round(2)
display_summary.columns = ["ผลตอบแทน/ปี", "ความผันผวน/ปี", "Sharpe ratio", "Max Drawdown"]
st.dataframe(display_summary, use_container_width=True)
st.caption("ตารางนี้ยังไม่รวมค่าธรรมเนียมการซื้อขาย (ดูผลที่รวมค่าธรรมเนียมแล้วในกราฟเงินสะสมด้านล่าง)")

st.info(
    "**A: Equal-weight** — แบ่งเงินเท่ากันทุกตัว  \n"
    "**B: Markowitz** — คำนวณสัดส่วนด้วยสูตรคณิตศาสตร์ให้ Sharpe ratio สูงสุด  \n"
    "**C: Market-cap** — ถ่วงน้ำหนักตามมูลค่าบริษัท บริษัทใหญ่กว่าได้สัดส่วนมากกว่า"
)

# ---------- เงินลงทุนสะสมจริง (ทบต้นต่อเนื่อง) + เทียบ SET Index ----------
st.subheader("เงินลงทุนสะสม ถ้าลงทุนต่อเนื่องตลอดช่วงทดสอบ (คิดทบต้น)")
curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
bench_raw = load_benchmark(years_back)

fig_cum, ax_cum = plt.subplots(figsize=(11, 5))
for name, curve in curves.items():
    ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], linewidth=1.6)

bench_note = ""
if not bench_raw.empty:
    combined_index = next(iter(curves.values())).index
    bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
    if bench_aligned.notna().sum() > 10:
        bench_ret = bench_aligned.pct_change().fillna(0)
        bench_curve = capital * (1 + bench_ret).cumprod()
        ax_cum.plot(bench_curve.index, bench_curve.values, label="Buy & Hold SET Index", linewidth=1.8,
                    linestyle="--", color="black")
    else:
        bench_note = "ข้อมูล SET Index ไม่พอสำหรับช่วงเวลานี้ แสดงเฉพาะ 3 กลยุทธ์"
else:
    bench_note = "ดึงข้อมูล SET Index (^SET.BK) ไม่สำเร็จ แสดงเฉพาะ 3 กลยุทธ์"

ax_cum.axhline(capital, color="gray", linewidth=0.7, linestyle=":")
ax_cum.set_xlabel("วันที่")
ax_cum.set_ylabel(f"มูลค่าพอร์ต (บาท) เริ่มจาก {capital:,.0f}")
ax_cum.set_title(f"Cumulative growth -- includes {cost_pct*100:.2f}% cost per rebalance")
ax_cum.legend(loc="upper left", fontsize=9)
st.pyplot(fig_cum)
if bench_note:
    st.caption(f"⚠️ {bench_note}")
st.caption(
    "เส้นนี้คือ 'เงินก้อนเดียวเดินทางต่อเนื่อง' ข้ามทุกไตรมาสจริง เส้นประดำคือ Buy & Hold SET Index "
    "ล้วนๆ ไม่ปรับพอร์ตเลย ใช้เป็นเกณฑ์เทียบจากภายนอกที่เหมาะกับหุ้นไทยกว่าดัชนีต่างประเทศ"
)

# ---------- Stress Test ----------
if stress_option != "ไม่ระบุ":
    st.subheader(f"ทดสอบเฉพาะช่วงวิกฤต: {stress_option}")
    stress_ranges = {
        "COVID-19 (ก.พ.–เม.ย. 2563)": ("2020-02-01", "2020-04-30"),
        "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)": ("2018-01-01", "2018-12-31"),
    }
    s_start, s_end = stress_ranges[stress_option]
    all_returns_full = data.pct_change().dropna()
    stress_returns = all_returns_full.loc[s_start:s_end]
    if len(stress_returns) < 5:
        st.warning("ข้อมูลย้อนหลังที่มีไม่ครอบคลุมช่วงนี้ — ลองเพิ่ม 'ข้อมูลย้อนหลังกี่ปี' ในตั้งค่าขั้นสูง")
    else:
        stress_rows = []
        for name, w in last_weights.items():
            total, ar, av, sh, mdd = evaluate(w, stress_returns)
            stress_rows.append({"กลยุทธ์": name, "ผลตอบแทนรวมช่วงนี้": f"{total:.1%}", "Max Drawdown ช่วงนี้": f"{mdd:.1%}"})
        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
        st.caption("ใช้สัดส่วนน้ำหนักจากรอบล่าสุดของแต่ละกลยุทธ์ ประเมินย้อนกลับเฉพาะช่วงวิกฤตที่เลือก (ไม่ใช่ walk-forward เต็มรูปแบบ เพราะช่วงสั้นเกินจะแบ่ง train/test ได้)")

# ---------- ภาคผนวกวิเคราะห์เชิงลึก ----------
with st.expander("🔬 ข้อมูลวิเคราะห์เชิงลึกทางสถิติ (สำหรับภาคผนวกโครงงานวิชาการ)"):
    st.markdown("#### ช่วงความเชื่อมั่น 95% และนัยสำคัญทางสถิติ (Paired t-test)")
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    means, errs = [], []
    for s in strategies:
        vals = df[df.strategy == s]["sharpe"].dropna().values
        mean, sd = vals.mean(), vals.std(ddof=1)
        se = sd / np.sqrt(len(vals))
        t_crit = stats.t.ppf(0.975, df=len(vals) - 1)
        means.append(mean)
        errs.append(t_crit * se)
    ax1.bar([s.split(" ")[0] for s in strategies], means, yerr=errs, capsize=8,
            color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(strategies)], alpha=0.85)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_ylabel(f"Sharpe ratio (mean of {n_folds} folds)")
    ax1.set_title("Sharpe ratio with 95% confidence interval")
    st.pyplot(fig1)

    rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            t_stat, p_val = stats.ttest_rel(wide[s1], wide[s2])
            rows.append({
                "เปรียบเทียบ": f"{s1.split(' ')[0]} vs {s2.split(' ')[0]}",
                "p-value": round(p_val, 3),
                "สรุป": "ต่างกันจริง (มีนัยสำคัญ, p<0.05)" if p_val < 0.05 else "ยังสรุปไม่ได้ชัดเจน (p≥0.05)",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### เส้นพรมแดนประสิทธิภาพ (Efficient Frontier) จากหุ้นที่เลือกจริง")
    all_returns_full = data.pct_change().dropna()
    fig_ef = efficient_frontier_fig(all_returns_full, last_weights)
    st.pyplot(fig_ef)
    st.caption("จุดสีคือพอร์ตสุ่ม 2,500 แบบจากหุ้นที่เลือกจริง สัญลักษณ์ขอบดำคือตำแหน่งของแต่ละกลยุทธ์ (สัดส่วนจากรอบล่าสุด)")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ดาวน์โหลดผลดิบทุกรอบเป็น CSV", csv, "walk_forward_results.csv", "text/csv")

st.divider()
st.caption(
    "⚠️ ผลลัพธ์ทั้งหมดคำนวณจากข้อมูลราคาหุ้นในอดีต (backtest) เท่านั้น ไม่ใช่การรับประกันผลตอบแทนในอนาคต "
    "ยังไม่รวมภาษี เครื่องมือนี้จัดทำเพื่อการศึกษาในโครงงานวิทยาศาสตร์ ไม่ใช่คำแนะนำการลงทุน\n\n"
    "ข้อสมมติที่ควรรู้: (1) Sharpe ratio คำนวณโดยตั้ง risk-free rate = 0 เพื่อความง่าย "
    "(2) กลยุทธ์ Market-cap ใช้จำนวนหุ้นที่ออกจำหน่ายปัจจุบัน คูณราคาย้อนหลัง เป็นค่าประมาณมูลค่าตลาดในอดีต "
    "ไม่ใช่มูลค่าตลาดจริงในวันนั้น (3) แต่ละรอบ walk-forward ใช้ข้อมูล train ที่ทับซ้อนกันบางส่วน จึงไม่เป็นอิสระจากกันทั้งหมด "
    "ผลการทดสอบนัยสำคัญทางสถิติจึงควรตีความอย่างระมัดระวัง"
)
