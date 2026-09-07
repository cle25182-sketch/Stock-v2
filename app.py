import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from datetime import datetime, timedelta
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="THE FINANCIAL GAZETTE & STOCK CHRONICLE (1602)",
    page_icon="📜",
    layout="wide"
)

# Custom Vintage Newspaper CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;900&family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=Sarabun:wght@300;400;600&display=swap');

    /* Background and Global Font */
    .stApp {
        background-color: #f4ebd9 !important;
        color: #1a1612 !important;
        font-family: 'Playfair Display', 'Georgia', 'Sarabun', serif !important;
    }
    
    /* Sidebar Vintage Style */
    [data-testid="stSidebar"] {
        background-color: #eaedd5 !important;
        background-color: #e8dcc4 !important;
        border-right: 3px double #3d2f1d !important;
    }
    
    /* Newspaper Masthead Header */
    .masthead {
        text-align: center;
        border-top: 4px double #2b2013;
        border-bottom: 4px double #2b2013;
        padding: 15px 0px;
        margin-bottom: 25px;
    }
    .masthead-sub {
        font-family: 'Cinzel', serif;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-bottom: 1px solid #2b2013;
        border-top: 1px solid #2b2013;
        padding: 4px 0;
        margin: 10px 0;
    }
    .masthead-title {
        font-family: 'Cinzel', serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #1a1612;
        letter-spacing: 2px;
        text-transform: uppercase;
        line-height: 1.1;
    }
    
    /* Vintage Breaking News Winner Box */
    .winner-box {
        background-color: #ebdcc1;
        border: 3px double #3d2f1d;
        padding: 22px;
        margin-bottom: 25px;
        box-shadow: 4px 4px 0px #3d2f1d;
    }
    .badge-winner {
        background-color: #5c1d1d;
        color: #f4ebd9;
        padding: 4px 12px;
        font-family: 'Cinzel', serif;
        font-size: 0.85rem;
        font-weight: bold;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-family: 'Cinzel', 'Georgia', serif !important;
        color: #5c1d1d !important;
        font-weight: bold !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3d2f1d !important;
        color: #f4ebd9 !important;
        border: 2px solid #1a1612 !important;
        font-family: 'Cinzel', serif !font-weight: bold;
        border-radius: 0px !important;
        box-shadow: 3px 3px 0px #1a1612;
    }
    .stButton > button:hover {
        background-color: #5c1d1d !important;
        color: #ffffff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #e6d5b8 !important;
        border: 1px solid #3d2f1d !important;
        font-family: 'Cinzel', serif !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Cinzel', serif !important;
        color: #2b2013 !important;
        font-weight: 700 !important;
    }
    
    /* Divider line */
    hr {
        border-top: 2px solid #3d2f1d !important;
    }
</style>
""", unsafe_allow_html=True)

RF_RATE = 0.0

# ------------------------------------------------------------------
# Functions (Backtest & Data Engine)
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


def style_fig_vintage(fig, ax):
    """ฟังก์ชันปรับแต่งกราฟ Matplotlib ให้อยู่ในธีมหนังสือพิมพ์โบราณ"""
    fig.patch.set_facecolor('#f4ebd9')
    ax.set_facecolor('#fdfaf3')
    ax.spines['top'].set_color('#3d2f1d')
    ax.spines['bottom'].set_color('#3d2f1d')
    ax.spines['left'].set_color('#3d2f1d')
    ax.spines['right'].set_color('#3d2f1d')
    ax.spines['top'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.xaxis.label.set_color('#2b2013')
    ax.yaxis.label.set_color('#2b2013')
    ax.title.set_color('#2b2013')
    ax.tick_params(colors='#2b2013', labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')


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
    sc = ax.scatter(res[:, 0] * 100, res[:, 1] * 100, c=res[:, 2], cmap="copper", s=6, alpha=0.6)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Sharpe ratio (คะแนนความคุ้มค่า)", color="#2b2013", family='serif')
    cb.ax.yaxis.set_tick_params(color='#2b2013')
    plt.setp(plt.getp(cb.ax, 'yticklabels'), color='#2b2013', family='serif')

    markers = list("o*D^v<>")
    for idx, (name, w) in enumerate(weights_by_name.items()):
        port_ret = w @ mu * 252
        port_vol = np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(252)
        ax.scatter(port_vol * 100, port_ret * 100, marker=markers[idx % len(markers)], s=200,
                   edgecolor="#1a1612", color="#5c1d1d", linewidth=1.5, label=name.split(" ")[0], zorder=5)
    
    ax.set_xlabel("Volatility / ความผันผวนต่อปี (%)", family='serif')
    ax.set_ylabel("Expected Return / ผลตอบแทนคาดหวังต่อปี (%)", family='serif')
    ax.set_title("Efficient Frontier (ความสัมพันธ์ระหว่างความเสี่ยงและผลตอบแทน)", family='serif', weight='bold')
    ax.legend(loc="lower right", fontsize=8, facecolor='#f4ebd9', edgecolor='#3d2f1d')
    style_fig_vintage(fig, ax)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Vintage Newspaper UI Structure
# ------------------------------------------------------------------

# Header Banner
st.markdown("""
<div class="masthead">
    <div class="masthead-title">THE FINANCIAL GAZETTE</div>
    <div class="masthead-sub">
        <span>VOL. CXXIV NO. 1602</span> &nbsp;•&nbsp; 
        <span>หนังสือพิมพ์ข่าวสารการเงินและจัดพอร์ตหุ้น</span> &nbsp;•&nbsp; 
        <span>EST. ค.ศ. 1602</span>
    </div>
    <p style="font-style: italic; font-size: 0.95rem; color: #4a3823; margin-bottom: 0;">
        "การกระจายความเสี่ยงอย่างชาญฉลาด คือหนทางสู่ความมั่งคั่งอันยั่งยืนแห่งยุคสมัย"
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📜 ประกาศจากหอการค้า")
    st.caption("กรอกรายละเอียดเงินทุนและหุ้นเพื่อคำนวณ")
    
    capital = st.number_input("💵 เงินลงทุนเริ่มต้น (บาท)", min_value=1000, value=100000, step=5000)
    ticker_input = st.text_input(
        "📌 พิมพ์รหัสหุ้น (คั่นด้วย comma ,)",
        value="PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK",
        help="ใส่รหัสหุ้น เช่น PTT.BK, CPALL.BK สำหรับหุ้นไทย หรือ AAPL, TSLA สำหรับหุ้นสหรัฐฯ"
    )
    selected = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))
    
    with st.expander("⚙️ ตั้งค่าคณิตศาสตร์เชิงลึก"):
        train_window = st.slider("ช่วงเรียนรู้ Train (วัน)", 126, 378, 252, step=21)
        test_window = st.slider("ช่วงทดสอบจริง Test (วัน)", 21, 126, 63, step=21)
        years_back = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 10, 6)
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขาย (%)", 0.0, 1.0, 0.15, step=0.05) / 100
        stress_option = st.selectbox(
            "ทดสอบความแข็งแกร่งยุควิกฤต",
            ["ไม่ระบุ", "COVID-19 (ก.พ.–เม.ย. 2563)", "สงครามการค้าจีน-สหรัฐฯ (ม.ค.–ธ.ค. 2561)"]
        )
        
    run = st.button("📰 ตีพิมพ์รายงานวิเคราะห์", type="primary", use_container_width=True)

if not run:
    st.info("📜 **คำแนะนำ:** กรอกเงินทุนและหุ้นทางแถบซ้ายมือ จากนั้นกด **'ตีพิมพ์รายงานวิเคราะห์'** เพื่ออ่านฉบับเต็ม")
    st.stop()

if len(selected) < 2:
    st.error("⚠️ กรุณาเลือกหุ้นอย่างน้อย 2 ตัวขึ้นไป เพื่อทำการกระจายความเสี่ยง")
    st.stop()

with st.spinner("📜 กำลังค้นหารายงานราคาหุ้นย้อนหลังตามบันทึก..."):
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

# Identify Winner Strategy
best_strategy = summary["sharpe"].idxmax()
best_return = summary.loc[best_strategy, "ann_return"]
best_vol = summary.loc[best_strategy, "ann_vol"]
best_sharpe = summary.loc[best_strategy, "sharpe"]
best_mdd = summary.loc[best_strategy, "max_drawdown"]

# FRONT PAGE EXTRA! EXTRA!
st.markdown("## 📢 ข่าวกรองการลงทุนประจำวัน: ฟันธงกลยุทธ์ผู้ชนะ")

st.markdown(f"""
<div class="winner-box">
    <span class="badge-winner">🏆 EXTRA! กลยุทธ์ที่แนะนำที่สุดสำหรับคุณ</span>
    <h2 style="color: #5c1d1d; margin-top: 12px; margin-bottom: 5px; font-family: 'Cinzel', serif;">{best_strategy}</h2>
    <p style="font-size: 1.05rem; color: #2b2013; line-height: 1.6;">
        จากการทดสอบย้อนหลัง <b>{years_back} ปี</b> (ทดสอบทั้งหมด {n_folds} ไตรมาส) กลยุทธ์นี้พิสูจน์แล้วว่าให้ 
        <b>ความคุ้มค่าเทียบกับความเสี่ยงสูงที่สุด</b> ช่วยปกป้องเงินทุนของคุณจากสภาวะตลาดผันผวนได้อย่างทรงประสิทธิภาพที่สุด
    </p>
</div>
""", unsafe_allow_html=True)

# ACTION PLAN (ALLOCATION IN BAHT)
st.markdown("### 💰 ใบสั่งซื้อและจัดสรรเงินทุนจริง (Action Plan)")
st.caption(f"หากนำเงินทุนเริ่มต้น **{capital:,.0f} บาท** มากระจายตามกลยุทธ์ผู้ชนะ จะได้แผนแบ่งซื้อดังนี้:")

winner_weights = last_weights[best_strategy]
action_plan = []
for ticker, weight in zip(selected, winner_weights):
    allocated_amount = capital * weight
    action_plan.append({
        "รหัสหุ้น (Ticker)": ticker,
        "สัดส่วนการลงทุน (%)": f"{weight * 100:.2f}%",
        "จำนวนเงินที่ต้องจัดซื้อ (บาท)": f"{allocated_amount:,.2f} บาท"
    })

action_df = pd.DataFrame(action_plan)
st.dataframe(action_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 📊 บันทึกตัวเลขคาดการณ์สำคัญของพอร์ตผู้ชนะ")

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
        help="เปอร์เซ็นต์การขาดทุนชั่วคราวสูงสุดที่เคยเกิดขึ้นในอดีต"
    )
    st.caption("🔻 ขาดทุนชั่วคราวลึกสุด")

st.markdown("---")
st.subheader("⚖️ ตารางเปรียบเทียบผลลัพธ์ทั้ง 3 กลยุทธ์")

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
* **A: Equal-weight:** กระจายเงินในหุ้นทุกตัวเท่ากัน เป็นวิธีที่เรียบง่ายแต่ทรงพลัง
* **B: Markowitz:** คำนวณหาสัดส่วนที่เสี่ยงน้อยที่สุดและทำกำไรดีที่สุดด้วยคณิตศาสตร์
* **C: Market-cap:** ลงเงินตามขนาดบริษัท บริษัทใหญ่สุดจะได้รับสัดส่วนเงินมากที่สุด
""")

st.markdown("---")
st.subheader("📈 การเติบโตของเงินทุนสะสม (เมื่อลงทุนต่อเนื่อง)")

curves = cumulative_growth(daily_returns, capital, cost_pct, test_window)
bench_raw = load_benchmark(years_back)

fig_cum, ax_cum = plt.subplots(figsize=(10, 4.5))

palette = ["#5c1d1d", "#2d4a3e", "#2b4353"]
for idx, (name, curve) in enumerate(curves.items()):
    ax_cum.plot(curve.index, curve.values, label=name.split(" ")[0], linewidth=2.0, color=palette[idx % len(palette)])

if not bench_raw.empty:
    combined_index = next(iter(curves.values())).index
    bench_aligned = bench_raw.reindex(bench_raw.index.union(combined_index)).ffill().reindex(combined_index)
    if bench_aligned.notna().sum() > 10:
        bench_ret = bench_aligned.pct_change().fillna(0)
        bench_curve = capital * (1 + bench_ret).cumprod()
        ax_cum.plot(bench_curve.index, bench_curve.values, label="ซื้อแล้วถือยาว SET Index", linewidth=1.5,
                    linestyle="--", color="#7a6244")

ax_cum.axhline(capital, color="#8c2323", linewidth=1.0, linestyle=":", label="เงินทุนเริ่มต้น")
ax_cum.set_xlabel("ปี / เดือน", family='serif')
ax_cum.set_ylabel("มูลค่าพอร์ต (บาท)", family='serif')
ax_cum.set_title(f"เส้นทางมูลค่าพอร์ตจริงจากเงินทุนเริ่มต้น {capital:,.0f} บาท (หักค่าธรรมเนียมแล้ว)", family='serif', weight='bold')
ax_cum.legend(loc="upper left", fontsize=8, facecolor='#f4ebd9', edgecolor='#3d2f1d')
style_fig_vintage(fig_cum, ax_cum)

st.pyplot(fig_cum)
st.caption("📌 **คำบรรยายภาพ:** กราฟลายเส้นแสดงการเติบโตของเงินทุนจริงทบต้น ยิ่งเส้นอยู่สูง แปลว่าสร้างความมั่งคั่งได้มากเท่านั้น")

# STRESS TEST SECTION
if stress_option != "ไม่ระบุ":
    st.markdown("---")
    st.subheader(f"🛡️ รายงานการทดสอบพอร์ตช่วงวิกฤต: {stress_option}")
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
        st.caption("📌 แสดงให้เห็นว่าหากเกิดวิกฤตเศรษฐกิจ พอร์ตแต่ละแบบจะได้รับผลกระทบหนักแค่ไหน")

st.markdown("---")
with st.expander("🔬 ข้อมูลวิเคราะห์เชิงลึกทางสถิติ (สำหรับภาคผนวกโครงงานวิชาการ)"):
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

    st.markdown("#### 2. เส้นพรมแดนประสิทธิภาพ (Efficient Frontier)")
    all_returns_full = data.pct_change().dropna()
    fig_ef = efficient_frontier_fig(all_returns_full, last_weights)
    st.pyplot(fig_ef)
    st.caption("จุดสีสุ่มคือสัดส่วนการลงทุน 2,500 รูปแบบ กลยุทธ์ที่ดีควรอยู่ชิดขอบบนซ้ายของกลุ่มจุด")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ ดาวน์โหลดบันทึกข้อมูลดิบการทดสอบเป็นไฟล์ CSV", csv, "vintage_portfolio_backtest.csv", "text/csv")

st.divider()
st.caption(
    "📜 **ข้อควรระวัง:** ผลลัพธ์ทั้งหมดคำนวณจากบันทึกราคาหุ้นย้อนหลังในอดีต (Backtest) เพื่อการศึกษาในโครงงานเท่านั้น "
    "มิใช่คำแนะนำทางการเงินหรือการรับประกันผลตอบแทนในอนาคต"
    )
