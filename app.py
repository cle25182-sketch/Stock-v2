import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

st.set_page_config(
    page_title="ระบบจัดพอร์ตหุ้นอัจฉริยะ (สำหรับผู้เริ่มต้น)",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    .winner-box {
        background-color: #0e2f18;
        border: 2px solid #2e7d32;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #1a1c23;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4caf50;
    }
    .badge-winner {
        background-color: #2e7d32;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

RF_RATE = 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(tickers, years_back):
    """ดึงข้อมูลราคาหุ้นย้อนหลังผ่าน yfinance"""
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
    """ดึงข้อมูล SET Index สำหรับเปรียบเทียบ"""
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
    """ดึงข้อมูล มูลค่าตลาด (Market Cap)"""
    import yfinance as yf
    caps = {}
    for t in tickers:
        try:
            caps[t] = yf.Ticker(t).info.get("sharesOutstanding")
        except Exception:
            caps[t] = None
    return caps

def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap):
    """คำนวณสัดส่วนน้ำหนักการลงทุนของแต่ละกลยุทธ์"""
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

    weights = {"A: Equal-weight (แบ่งเงินเท่ากัน)": w_equal, "B: Markowitz (คำนวณด้วยสูตรคณิตศาสตร์)": w_markowitz}
    if use_marketcap:
        mcap = train_last_price * shares_arr
        weights["C: Market-cap (จัดตามขนาดบริษัท)"] = mcap / mcap.sum()
    return weights, opt.success


def evaluate(w, test_ret):
    """ประเมินผลการลงทุน"""
    port_ret = test_ret.values @ w
    cum = np.cumprod(1 + port_ret)
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RF_RATE) / ann_vol if ann_vol > 1e-10 else np.nan
    running_max = np.maximum.accumulate(cum)
    max_dd = ((cum - running_max) / running_max).min()
    return cum[-1] - 1, ann_ret, ann_vol, sharpe, max_dd

def run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window):
    """จำลองการทดสอบพอร์ตย้อนหลังแบบ Walk-Forward Validation"""
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
        train_last_price = data.iloc[start + train_window - 1].values
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
    """คำนวณมูลค่าพอร์ตสะสมคิดทบต้นพร้อมหักค่าธรรมเนียม"""
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

def efficient_frontier_fig(all_returns, weights_by_name, n_portfolios=3000):
    """สร้างกราฟ Efficient Frontier"""
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
    sc = ax.scatter(res[:, 0] * 100, res[:, 1] * 100, c=res[:, 2], cmap="viridis", s=6, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Sharpe ratio (คะแนนความคุ้มค่า)")
    markers = list("o*D^v<>")
    for idx, (name, w) in enumerate(weights_by_name.items()):
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        ax.scatter(port_vol * 100, port_ret * 100, marker=markers[idx % len(markers)], s=220,
                   edgecolor="black", linewidth=1.3, label=name.split(" ")[0], zorder=5)
    ax.set_xlabel("Volatility / ความผันผวนต่อปี (%)")
    ax.set_ylabel("Expected Return / ผลตอบแทนคาดหวังต่อปี (%)")
    ax.set_title("Efficient Frontier (ความสัมพันธ์ระหว่างความเสี่ยงและผลตอบแทน)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig

st.title("🎯 ระบบวิเคราะห์และจัดพอร์ตหุ้นอัจฉริยะ")
st.markdown(
    "เครื่องมือที่จะช่วยคุณ **ตัดสินใจแบ่งเงินซื้อหุ้นแต่ละตัวอย่างชัดเจน** โดยใช้ระบบจำลองย้อนหลังตามข้อมูลจริง "
    "เพื่อหากลยุทธ์ที่ทำกำไรได้ดีที่สุดและเสี่ยงน้อยที่สุดสำหรับคุณ"
)

with st.sidebar:
    st.header("⚙️ ตั้งค่าเงินลงทุนและหุ้น")
    capital = st.number_input("💵 เงินลงทุนเริ่มต้น (บาท)", min_value=1000, value=100000, step=5000,
                              help="ระบุจำนวนเงินจริงที่คุณต้องการนำมาจัดพอร์ต")
    ticker_input = st.text_input(
        "📌 พิมพ์รหัสหุ้น (คั่นด้วย comma ,)",
        value="PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK",
        help="ใส่รหัสหุ้น เช่น PTT.BK, CPALL.BK สำหรับหุ้นไทย หรือ AAPL, TSLA สำหรับหุ้นสหรัฐฯ"
    )
    selected = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))
    
    with st.expander("🛠️ ตั้งค่าเพิ่มเติม (สำหรับผู้ใช้งานระดับสูง)"):
        train_window = st.slider("ช่วงเรียนรู้ของโมเดล Train (วัน)", 126, 378, 252, step=21)
        test_window = st.slider("ช่วงทดสอบจริงต่อรอบ Test (วัน)", 21, 126, 63, step=21)
        years_back = st.slider("ดึงข้อมูลย้อนหลัง (ปี)", 3, 10, 6)
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขายต่อรอบ (%)", 0.0, 1.0, 0.15, step=0.05) / 100
        stress_option = st.selectbox(
            "ทดสอบความแข็งแกร่งในช่วงวิกฤต",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2563)", "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)"]
        )
        
    run = st.button("🚀 เริ่มวิเคราะห์พอร์ต", type="primary", use_container_width=True)

if not run:
    st.info("👈 กรอกเงินลงทุนและหุ้นทางด้านซ้าย แล้วกด **'เริ่มวิเคราะห์พอร์ต'** เพื่อดูคำแนะนำ")
    st.stop()

if len(selected) < 2:
    st.error("⚠️ กรุณาเลือกหุ้นอย่างน้อย 2 ตัวขึ้นไป เพื่อทดสอบการกระจายความเสี่ยง")
    st.stop()

with st.spinner("🔍 กำลังดึงข้อมูลราคาหุ้นและประมวลผลย้อนหลัง..."):
    try:
        data, valid_tickers = load_price_data(tuple(selected), years_back)
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลหุ้นได้: {e}")
        st.stop()

missing = [t for t in selected if t not in valid_tickers]
if missing:
    st.warning(f"⚠️ ไม่พบข้อมูลหุ้นบางตัว จึงถูกตัดออก: {', '.join(missing)}")
selected = valid_tickers

if len(selected) < 2:
    st.error("เหลือหุ้นที่ใช้งานได้น้อยกว่า 2 ตัว กรุณาเปลี่ยนรหัสหุ้นใหม่")
    st.stop()

caps = load_market_caps(tuple(selected))
use_marketcap = all(caps.get(t) for t in selected)
shares_arr = np.array([caps.get(t) or 0 for t in selected])

df, n_folds, n_failed, daily_returns, last_weights = run_walk_forward(
    data, shares_arr, use_marketcap, train_window, test_window
)

if df.empty:
    st.error("ข้อมูลไม่เพียงพอในการทดสอบ กรุณาลดช่วงวัน Train/Test หรือเพิ่มจำนวนปีย้อนหลัง")
    st.stop()

strategies = list(df["strategy"].unique())
summary = df.groupby("strategy")[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean().reindex(strategies)

# Identify Best Strategy
best_strategy = summary["sharpe"].idxmax()
best_return = summary.loc[best_strategy, "ann_return"]
best_vol = summary.loc[best_strategy, "ann_vol"]
best_sharpe = summary.loc[best_strategy, "sharpe"]
best_mdd = summary.loc[best_strategy, "max_drawdown"]

st.subheader("💡 สรุปผลการวิเคราะห์ & คำแนะนำการจัดพอร์ต")

# WINNER BANNER
st.markdown(f"""
<div class="winner-box">
    <span class="badge-winner">🏆 กลยุทธ์ที่แนะนำที่สุดสำหรับคุณ</span>
    <h2 style="color: #4caf50; margin-top: 10px; margin-bottom: 5px;">{best_strategy}</h2>
    <p style="font-size: 1.05rem; color: #e0e0e0;">
        จากข้อมูลย้อนหลัง <b>{years_back} ปี</b> (ทดสอบทั้งหมด {n_folds} รอบ) กลยุทธ์นี้ให้ <b>ความคุ้มค่าเทียบกับความเสี่ยงสูงที่สุด</b> 
        ช่วยให้พอร์ตของคุณเติบโตได้ดีในขณะที่มีความผันผวนและการดอยต่ำที่สุด
    </p>
</div>
""", unsafe_allow_html=True)

# ACTION PLAN TABLE (ALLOCATION IN BAHT)
st.markdown("### 💰 แผนการจัดสรรเงินซื้อหุ้นจริง (Action Plan)")
st.caption(f"หากคุณนำเงินทุนเริ่มต้น **{capital:,.0f} บาท** มาจัดพอร์ตตามกลยุทธ์ผู้ชนะ จะแบ่งซื้อหุ้นดังนี้:")

winner_weights = last_weights[best_strategy]
action_plan = []
for ticker, weight in zip(selected, winner_weights):
    allocated_amount = capital * weight
    action_plan.append({
        "รหัสหุ้น (Ticker)": ticker,
        "สัดส่วนน้ำหนัก (%)": f"{weight * 100:.2f}%",
        "จำนวนเงินที่ต้องซื้อ (บาท)": f"{allocated_amount:,.2f} บาท"
    })

action_df = pd.DataFrame(action_plan)
st.dataframe(action_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📊 ตัวเลขคาดการณ์สำคัญของกลยุทธ์นี้")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric(
        label="ผลตอบแทนเฉลี่ยต่อปี", 
        value=f"{best_return * 100:.1f}%",
        help="คาดการณ์เปอร์เซ็นต์กำไรทบต้นที่พอร์ตทำได้ในแต่ละปี"
    )
    st.caption("📈 กำไรคาดหวังต่อปี")

with col_m2:
    st.metric(
        label="ระดับความเหวี่ยง (Volatility)", 
        value=f"{best_vol * 100:.1f}%",
        help="ถ้าน้อยแปลว่าพอร์ตวิ่งนิ่งๆ ถ้ามากแปลว่าราคาพอร์ตเหวี่ยงขึ้นลงหวือหวา"
    )
    st.caption("กระเพื่อมของราคาต่อปี")

with col_m3:
    st.metric(
        label="คะแนนความคุ้มค่า (Sharpe)", 
        value=f"{best_sharpe:.2f}",
        help="ยิ่งสูงยิ่งดี! แสดงว่ากำไรที่ได้ คุ้มค่ากับความเหวี่ยงที่คุณต้องเจอ"
    )
    st.caption("⭐ ยิ่งสูง ยิ่งคุ้มเสี่ยง")

with col_m4:
    st.metric(
        label="ช่วงดอยหนักสุด (Max Drawdown)", 
        value=f"{best_mdd * 100:.1f}%",
        help="เปอร์เซ็นต์การขาดทุนชั่วคราวสูงสุดที่เคยเกิดขึ้นในอดีต (ต้องเตรียมใจรับให้ได้)"
    )
    st.caption("🔻 ขาดทุนชั่วคราวลึกสุด")

st.markdown("---")
st.subheader("⚖️ เปรียบเทียบผลลัพธ์ทั้ง 3 กลยุทธ์แบบเข้าใจง่าย")

display_summary = summary.copy()
display_summary["ann_return"] = (display_summary["ann_return"] * 100).round(1).astype(str) + "%"
display_summary["ann_vol"] = (display_summary["ann_vol"] * 100).round(1).astype(str) + "%"
display_summary["max_drawdown"] = (display_summary["max_drawdown"] * 100).round(1).astype(str) + "%"
display_summary["sharpe"] = display_summary["sharpe"].round(2)

display_summary.columns = [
    "ผลตอบแทนต่อปี (%)", 
    "ความเหวี่ยงต่อปี (%)", 
    "คะแนนความคุ้มค่า (Sharpe)", 
    "ดอยหนักสุดในอดีต (%)"
]

st.dataframe(display_summary, use_container_width=True)

st.info("""
💡 **คำอธิบายรูปแบบการจัดพอร์ต:**
* **A: Equal-weight:** กระจายเงินลงทุนในหุ้นทุกตัวเท่าๆ กัน (เช่น มี 5 ตัว ซื้อตัวละ 20%) เป็นวิธีที่เข้าใจง่ายแต่ได้ผลดีเยี่ยม
* **B: Markowitz:** ใช้สูตรคณิตศาสตร์คำนวณหาสัดส่วนที่เสี่ยงน้อยที่สุดและทำกำไรได้ดีที่สุดจากสถิติในอดีต
* **C: Market-cap:** ลงเงินตามขนาดบริษัท บริษัทที่ใหญ่อันดับต้นๆ จะถูกแบ่งเงินไปลงมากที่สุด
""")

st.subheader("📈 การเติบโตของเงินทุนสะสม (เมื่อลงทุนต่อเนื่อง)")
curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
bench_raw = load_benchmark(years_back)

fig_cum, ax_cum = plt.subplots(figsize=(10, 4.5))
for name, curve in curves.items():
    ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], linewidth=1.8)

if not bench_raw.empty:
    combined_index = next(iter(curves.values())).index
    bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
    if bench_aligned.notna().sum() > 10:
        bench_ret = bench_aligned.pct_change().fillna(0)
        bench_curve = capital * (1 + bench_ret).cumprod()
        ax_cum.plot(bench_curve.index, bench_curve.values, label="ซื้อแล้วถือยาว SET Index", linewidth=1.5,
                    linestyle="--", color="gray")

ax_cum.axhline(capital, color="red", linewidth=0.8, linestyle=":", label="เงินทุนเริ่มต้น")
ax_cum.set_xlabel("ปี/เดือน")
ax_cum.set_ylabel("มูลค่าพอร์ต (บาท)")
ax_cum.set_title(f"เส้นทางมูลค่าพอร์ตจริงจากเงินทุนเริ่มต้น {capital:,.0f} บาท (หักค่าธรรมเนียมแล้ว)")
ax_cum.legend(loc="upper left", fontsize=8)
st.pyplot(fig_cum)
st.caption("📌 **คำบรรยายภาพ:** กราฟนี้แสดงการเติบโตของเงินทุนจริงเมื่อเวลาผ่านไป ยิ่งเส้นอยู่สูง แปลว่าสร้างเงินทบต้นได้มากเท่านั้น")

# STRESS TEST SECTION
if stress_option != "ไม่ระบุ":
    st.markdown("---")
    st.subheader(f"🛡️ ผลทดสอบความอึดของพอร์ตช่วงวิกฤต: {stress_option}")
    stress_ranges = {
        "COVID-19 (ก.พ.–เม.ย. 2563)": ("2020-02-01", "2020-04-30"),
        "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)": ("2018-01-01", "2018-12-31"),
    }
    s_start, s_end = stress_ranges[stress_option]
    all_returns_full = data.pct_change().dropna()
    stress_returns = all_returns_full.loc[s_start:s_end]
    
    if len(stress_returns) >= 5:
        stress_rows = []
        for name, w in last_weights.items():
            total, ar, av, sh, mdd = evaluate(w, stress_returns)
            stress_rows.append({
                "กลยุทธ์": name, 
                "ผลตอบแทนรวมช่วงวิกฤต": f"{total * 100:.1f}%", 
                "ดอยหนักสุดช่วงนี้ (Max Drawdown)": f"{mdd * 100:.1f}%"
            })
        st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
        st.caption("📌 แสดงให้เห็นว่าหากเกิดวิกฤตเศรษฐกิจขึ้น พอร์ตแต่ละแบบจะได้รับผลกระทบหนักแค่ไหน")

st.markdown("---")
with st.expander("🔬 ข้อมูลวิเคราะห์เชิงลึกทางสถิติ (สำหรับโครงงานวิชาการ / ผู้ใช้ขั้นสูง)"):
    st.markdown("#### 1. ความน่าเชื่อถือทางสถิติ (Paired t-test)")
    wide = df.pivot(index="round", columns="strategy", values="sharpe")
    rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            s1, s2 = strategies[i], strategies[j]
            t_stat, p_val = stats.ttest_rel(wide[s1], wide[s2])
            rows.append({
                "คู่เปรียบเทียบ": f"{s1.split(' ')[0]} vs {s2.split(' ')[0]}",
                "p-value": round(p_val, 4),
                "สรุปความต่างทางสถิติ": "แตกต่างกันอย่างมีนัยสำคัญ (p < 0.05)" if p_val < 0.05 else "ยังสรุปไม่ได้ว่าต่างกันชัดเจน (p ≥ 0.05)"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### 2. กราฟพรมแดนประสิทธิภาพ (Efficient Frontier)")
    all_returns_full = data.pct_change().dropna()
    fig_ef = efficient_frontier_fig(all_returns_full, last_weights)
    st.pyplot(fig_ef)
    st.caption("จุดสีสุ่มคือสัดส่วนการลงทุนแบบต่างๆ 3,000 รูปแบบ กลยุทธ์ที่ดีควรอยู่ชิดขอบบนซ้ายของกลุ่มจุด")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ดาวน์โหลดข้อมูลการทดสอบดิบทั้งหมดเป็นไฟล์ CSV", csv, "portfolio_backtest_results.csv", "text/csv")

st.divider()
st.caption(
    "⚠️ **ข้อควรระวัง:** ผลลัพธ์ทั้งหมดคำนวณจากข้อมูลราคาหุ้นย้อนหลังในอดีต (Backtest) เพื่อการศึกษาในโครงงานเท่านั้น "
    "ไม่ถือเป็นการการันตีผลตอบแทนในอนาคต หรือเป็นคำแนะนำทางการเงิน"
    )
