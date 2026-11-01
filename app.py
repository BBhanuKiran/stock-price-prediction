"""
TrendIQ - Intelligent Stock Market Trend Predictor using ML
All-in-one learning + analysis platform. EDUCATIONAL USE ONLY - NOT FINANCIAL ADVICE.
Run:  streamlit run app.py
"""
import os, json, joblib
import numpy as np, pandas as pd, yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from features import add_features, time_split, FEATURES, SEQ_LEN

st.set_page_config(page_title="TrendIQ | Learn & Analyze Markets", page_icon="📈", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0b1f3a,#16457a);padding:26px 30px;border-radius:16px;color:#fff;margin-bottom:14px}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:4px 0 0;color:#cfe0f5}
.card{background:#f6f9fd;border:1px solid #dbe5f1;border-radius:12px;padding:14px 18px;margin-bottom:10px;color:#12263f}
.warn{background:#fff6e5;border-left:5px solid #f0a020;padding:10px 14px;border-radius:8px;color:#5a3b00;margin:8px 0}
.up{color:#0a8f4d;font-weight:700}.down{color:#d02b3a;font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:4px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{background:#eef3f9;border-radius:8px 8px 0 0;padding:8px 14px;color:#12263f}
.stTabs [aria-selected="true"]{background:#16457a;color:#fff}
</style>""", unsafe_allow_html=True)

DISCLAIMER = ("⚠️ **Disclaimer:** Educational project only. Predictions are statistical estimates, "
              "NOT financial advice. No model can predict markets with certainty. Never invest money you cannot afford to lose.")

# ------------------------------------------------------------------ ASSET CATALOG
CATALOG = {
 "🇮🇳 Indian Stocks (NSE)": {"TCS":"TCS.NS","Reliance":"RELIANCE.NS","Infosys":"INFY.NS","HDFC Bank":"HDFCBANK.NS",
   "ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS","ITC":"ITC.NS","L&T":"LT.NS","Bharti Airtel":"BHARTIARTL.NS",
   "Wipro":"WIPRO.NS","HCL Tech":"HCLTECH.NS","Kotak Bank":"KOTAKBANK.NS","Axis Bank":"AXISBANK.NS",
   "Maruti Suzuki":"MARUTI.NS","Tata Motors":"TATAMOTORS.NS","Tata Steel":"TATASTEEL.NS","Sun Pharma":"SUNPHARMA.NS",
   "Asian Paints":"ASIANPAINT.NS","Bajaj Finance":"BAJFINANCE.NS","HUL":"HINDUNILVR.NS","Adani Enterprises":"ADANIENT.NS",
   "ONGC":"ONGC.NS","Titan":"TITAN.NS","NTPC":"NTPC.NS","Zomato":"ZOMATO.NS"},
 "📊 Indices": {"Nifty 50":"^NSEI","Sensex":"^BSESN","Bank Nifty":"^NSEBANK","S&P 500":"^GSPC","Nasdaq":"^IXIC",
   "Dow Jones":"^DJI","FTSE 100":"^FTSE","Nikkei 225":"^N225"},
 "🌎 US Stocks": {"Apple":"AAPL","Microsoft":"MSFT","Google":"GOOGL","Amazon":"AMZN","Tesla":"TSLA","Nvidia":"NVDA","Meta":"META"},
 "₿ Crypto": {"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","Solana":"SOL-USD","XRP":"XRP-USD","BNB":"BNB-USD",
   "Dogecoin":"DOGE-USD","Cardano":"ADA-USD"},
 "🪙 Commodities": {"Gold":"GC=F","Silver":"SI=F","Crude Oil":"CL=F","Natural Gas":"NG=F","Copper":"HG=F"},
 "💱 Forex": {"USD/INR":"USDINR=X","EUR/INR":"EURINR=X","GBP/INR":"GBPINR=X","EUR/USD":"EURUSD=X","USD/JPY":"JPY=X"},
 "🧺 ETFs": {"Nifty BeES":"NIFTYBEES.NS","Gold BeES":"GOLDBEES.NS","Bank BeES":"BANKBEES.NS","SPY":"SPY","QQQ":"QQQ"},
}
DURATIONS = {"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730,"5 Years":1825,"10 Years":3650,"Max (All data)":None,"Custom range":-1}

# ------------------------------------------------------------------ DATA
@st.cache_data(ttl=900, show_spinner=False)
def load_full(ticker):
    """Downloads ALL available history (more data = better learning). Handles missing data."""
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all").ffill()           # fill missing values with previous day
    return df

@st.cache_data(ttl=600, show_spinner=False)
def snapshot(tickers):
    d = yf.download(list(tickers), period="7d", auto_adjust=True, progress=False)["Close"]
    if isinstance(d, pd.Series): d = d.to_frame(tickers[0])
    d = d.ffill().dropna(how="all")
    return d

def pct(a, b): return (a / b - 1) * 100

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.markdown("## 📈 TrendIQ")
cat = st.sidebar.selectbox("1️⃣ What do you want to explore?", list(CATALOG.keys()))
q = st.sidebar.text_input("🔍 Search name or symbol", placeholder="e.g. tata, gold, btc")
opts = {k: v for k, v in CATALOG[cat].items() if q.lower() in k.lower() or q.lower() in v.lower()}
if not opts: st.sidebar.warning("No match in this category. Try a custom ticker below.")
name = st.sidebar.selectbox("2️⃣ Select asset", list(opts.keys())) if opts else None
custom = st.sidebar.text_input("Or type any Yahoo Finance ticker", placeholder="e.g. PAYTM.NS")
ticker = custom.strip().upper() if custom.strip() else (opts[name] if name else "TCS.NS")
label = custom.strip().upper() if custom.strip() else (name or ticker)
dur = st.sidebar.selectbox("3️⃣ Duration", list(DURATIONS.keys()), index=3)
if DURATIONS[dur] == -1:
    c1, c2 = st.sidebar.columns(2)
    d_start = c1.date_input("From", pd.Timestamp.today() - pd.Timedelta(days=365))
    d_end = c2.date_input("To", pd.Timestamp.today())
st.sidebar.markdown("---"); st.sidebar.caption(DISCLAIMER)

with st.spinner(f"Loading {ticker} ..."):
    full = load_full(ticker)
if full.empty or len(full) < 30:
    st.error(f"No data found for '{ticker}'. Check the symbol (NSE stocks end with .NS)."); st.stop()
feat = add_features(full)                       # indicators on FULL history (correct warm-up)
if DURATIONS[dur] is None: view = feat
elif DURATIONS[dur] == -1: view = feat.loc[str(d_start):str(d_end)]
else: view = feat[feat.index >= feat.index[-1] - pd.Timedelta(days=DURATIONS[dur])]
if len(view) < 5: st.warning("Very few rows in this duration - choose a longer one."); st.stop()
last, prev = float(full["Close"].iloc[-1]), float(full["Close"].iloc[-2])

st.markdown(f"""<div class="hero"><h1>📈 TrendIQ - Intelligent Stock Market Trend Predictor</h1>
<p>Learn markets from scratch • Analyze stocks, crypto, commodities & forex • ML-based UP/DOWN trend • Practice with virtual money</p></div>""", unsafe_allow_html=True)

tabs = st.tabs(["🏠 Home","📡 Live Data","📉 Technical Charts","🤖 Predict Trend","🏆 Model Performance",
                "🎮 Practice (No Risk)","📚 Learn Trading","🏦 Apps & Brokers","🛡️ Safety","👤 Team Details"])

# ------------------------------------------------------------------ HOME
with tabs[0]:
    st.markdown(DISCLAIMER)
    st.subheader("Market Pulse")
    pulse = {"Nifty 50":"^NSEI","Sensex":"^BSESN","Bitcoin":"BTC-USD","Gold":"GC=F","USD/INR":"USDINR=X","S&P 500":"^GSPC"}
    try:
        s = snapshot(tuple(pulse.values())); cols = st.columns(len(pulse))
        for col, (n, t) in zip(cols, pulse.items()):
            if t in s and s[t].dropna().shape[0] >= 2:
                a, b = s[t].dropna().iloc[-1], s[t].dropna().iloc[-2]
                col.metric(n, f"{a:,.2f}", f"{pct(a, b):+.2f}%")
    except Exception: st.info("Market pulse unavailable right now (network/API).")
    st.markdown("""<div class="card"><b>How to use this site</b><br>
1️⃣ Pick a category in the sidebar → 2️⃣ search & select an asset → 3️⃣ choose duration.<br>
Then explore <b>Live Data</b>, <b>Technical Charts</b>, <b>Predict Trend</b>, practice with <b>virtual money</b>, and study the <b>Learn Trading</b> manual.</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="warn"><b>Honest note:</b> Markets are noisy. A realistic accuracy for next-day UP/DOWN prediction is
roughly 50-56%. Anyone promising "100% accurate" prediction is wrong or scamming. Use this tool to <i>learn</i>, not to bet your savings.</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ LIVE DATA
with tabs[1]:
    st.subheader(f"{label}  ({ticker})")
    chg = pct(last, prev)
    m = st.columns(5)
    m[0].metric("Last Price", f"{last:,.2f}", f"{chg:+.2f}%")
    m[1].metric("Period High", f"{view['High'].max():,.2f}")
    m[2].metric("Period Low", f"{view['Low'].min():,.2f}")
    m[3].metric("Period Return", f"{pct(view['Close'].iloc[-1], view['Close'].iloc[0]):+.2f}%")
    m[4].metric("Avg Volume", f"{view['Volume'].mean():,.0f}")
    fig = go.Figure(go.Scatter(x=view.index, y=view["Close"], fill="tozeroy", line=dict(color="#16457a")))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Category snapshot - {cat}** (last day change)")
    try:
        tk = tuple(CATALOG[cat].values()); s = snapshot(tk); rows = []
        for n, t in CATALOG[cat].items():
            if t in s and s[t].dropna().shape[0] >= 2:
                a, b = s[t].dropna().iloc[-1], s[t].dropna().iloc[-2]; rows.append([n, t, round(a, 2), round(pct(a, b), 2)])
        snap = pd.DataFrame(rows, columns=["Name", "Symbol", "Price", "Change %"]).sort_values("Change %", ascending=False)
        st.dataframe(snap, use_container_width=True, hide_index=True)
    except Exception: st.info("Snapshot unavailable right now.")
    with st.expander("📄 Raw data table & download"):
        st.dataframe(view[["Open", "High", "Low", "Close", "Volume"]].tail(200))
        st.download_button("Download CSV", view[["Open", "High", "Low", "Close", "Volume"]].to_csv(), f"{ticker}.csv")

# ------------------------------------------------------------------ TECHNICAL CHARTS
with tabs[2]:
    o = st.columns(4)
    kind = o[0].radio("Chart", ["Candlestick", "Line"], horizontal=True)
    sma = o[1].checkbox("MA20 / MA50", True); bbs = o[2].checkbox("Bollinger Bands", False)
    ind = o[3].multiselect("Indicators", ["Volume", "RSI", "MACD"], ["Volume", "RSI", "MACD"])
    rows = 1 + len(ind); heights = [0.55] + [0.45 / len(ind)] * len(ind) if ind else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    if kind == "Candlestick":
        fig.add_trace(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low, close=view.Close, name="Price"), 1, 1)
    else: fig.add_trace(go.Scatter(x=view.index, y=view.Close, name="Close", line=dict(color="#16457a")), 1, 1)
    if sma:
        fig.add_trace(go.Scatter(x=view.index, y=view.MA20, name="MA20", line=dict(color="orange", width=1.3)), 1, 1)
        fig.add_trace(go.Scatter(x=view.index, y=view.MA50, name="MA50", line=dict(color="purple", width=1.3)), 1, 1)
    if bbs:
        for c_, n_ in [("BB_high", "BB Upper"), ("BB_low", "BB Lower")]:
            fig.add_trace(go.Scatter(x=view.index, y=view[c_], name=n_, line=dict(color="gray", dash="dot", width=1)), 1, 1)
    r = 2
    for i in ind:
        if i == "Volume": fig.add_trace(go.Bar(x=view.index, y=view.Volume, name="Volume", marker_color="#8fb3d9"), r, 1)
        if i == "RSI":
            fig.add_trace(go.Scatter(x=view.index, y=view.RSI, name="RSI(14)", line=dict(color="#16457a")), r, 1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=r, col=1); fig.add_hline(y=30, line_dash="dash", line_color="green", row=r, col=1)
        if i == "MACD":
            fig.add_trace(go.Scatter(x=view.index, y=view.MACD, name="MACD"), r, 1)
            fig.add_trace(go.Scatter(x=view.index, y=view.MACD_signal, name="Signal"), r, 1)
            fig.add_trace(go.Bar(x=view.index, y=view.MACD - view.MACD_signal, name="Histogram", marker_color="#9ab"), r, 1)
        r += 1
    fig.update_layout(height=480 + 150 * len(ind), xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    rs, mac = float(view.RSI.iloc[-1]), float(view.MACD.iloc[-1] - view.MACD_signal.iloc[-1])
    st.markdown(f"""<div class="card"><b>Indicator reading (learning aid)</b><br>
RSI = {rs:.1f} → {"Overbought zone (>70): price rose fast, pullback risk" if rs > 70 else "Oversold zone (<30): price fell fast, bounce possible" if rs < 30 else "Neutral zone"}<br>
MACD vs Signal → {"Bullish momentum (MACD above signal)" if mac > 0 else "Bearish momentum (MACD below signal)"}<br>
Price vs MA50 → {"Above long-term average (uptrend bias)" if last > view.MA50.iloc[-1] else "Below long-term average (downtrend bias)"}</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ MODELS
@st.cache_resource
def load_pooled():
    """Loads models trained by train_models.py (if present)."""
    out = {}
    try:
        out["rf"], out["scaler"] = joblib.load("rf_model.pkl"), joblib.load("scaler.pkl")
        from tensorflow.keras.models import load_model
        out["lstm"] = load_model("lstm_model.h5", compile=False)
    except Exception: pass
    return out

@st.cache_data(show_spinner=False)
def asset_model(ticker, _df):
    """Trains RandomForest on THIS asset's complete history with time split (no leakage)."""
    d = add_features(_df); p_row = d.iloc[[-1]]
    d = d.dropna(subset=FEATURES + ["Target"])
    tr, te = time_split(d, 0.8)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    rf.fit(tr[FEATURES], tr["Target"]); pred = rf.predict(te[FEATURES])
    y = te["Target"].values
    return dict(acc=accuracy_score(y, pred), prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                base=max(y.mean(), 1 - y.mean()), cm=confusion_matrix(y, pred).tolist(),
                p_up=float(rf.predict_proba(p_row[FEATURES].fillna(0))[0][1]),
                dates=te.index, actual=y, pred=pred, n_train=len(tr), n_test=len(te),
                imp=dict(zip(FEATURES, rf.feature_importances_)))

with tabs[3]:
    st.subheader("🤖 Next-Day Trend Prediction")
    st.markdown(DISCLAIMER)
    if len(full) < 300: st.warning("Not enough history to train a reliable model (need 300+ days)."); 
    else:
        with st.spinner("Training on all past data ..."):
            R = asset_model(ticker, full)
        P = load_pooled(); probs = {"RandomForest (this asset)": R["p_up"]}
        if "rf" in P:
            probs["RandomForest (multi-asset)"] = float(P["rf"].predict_proba(feat.iloc[[-1]][FEATURES].fillna(0))[0][1])
        if "lstm" in P and len(feat.dropna(subset=FEATURES)) >= SEQ_LEN:
            seq = P["scaler"].transform(feat.dropna(subset=FEATURES)[FEATURES].iloc[-SEQ_LEN:])
            probs["LSTM (60-day sequence)"] = float(P["lstm"].predict(seq[None].astype("float32"), verbose=0)[0][0])
        final = float(np.mean(list(probs.values())))            # ensemble = average probability
        up = final >= 0.5; conf = max(final, 1 - final) * 100
        st.markdown(f"## {ticker} - Tomorrow: <span class='{'up' if up else 'down'}'>{'UP' if up else 'DOWN'} Trend</span> (Confidence {conf:.0f}%)", unsafe_allow_html=True)
        st.progress(final, text=f"Probability of UP: {final*100:.1f}%")
        if conf < 56: st.markdown('<div class="warn">Confidence is low - the model is nearly undecided. Treat this as "no clear signal".</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Model": probs.keys(), "P(UP) %": [round(v * 100, 1) for v in probs.values()]}), hide_index=True)
        c = st.columns(5)
        for col, (k, v) in zip(c, [("Accuracy", R["acc"]), ("Precision", R["prec"]), ("Recall", R["rec"]), ("F1", R["f1"]), ("Naive baseline", R["base"])]):
            col.metric(k, f"{v*100:.1f}%")
        st.caption(f"Backtest on unseen last 20% ({R['n_test']} days) after training on {R['n_train']} earlier days. "
                   "'Naive baseline' = accuracy of always guessing the majority direction; a useful model must beat it.")
        n = min(120, len(R["dates"])); ch = go.Figure()
        ch.add_trace(go.Scatter(x=R["dates"][-n:], y=R["actual"][-n:], name="Actual", mode="lines", line=dict(shape="hv", color="#16457a")))
        ch.add_trace(go.Scatter(x=R["dates"][-n:], y=R["pred"][-n:] * 0.9 + 0.05, name="Predicted", mode="lines", line=dict(shape="hv", color="orange", dash="dot")))
        ch.update_layout(height=300, template="plotly_white", title="Actual vs Predicted trend (1 = UP, 0 = DOWN)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(ch, use_container_width=True)
        imp = pd.Series(R["imp"]).sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#16457a")).update_layout(
            height=340, template="plotly_white", title="Why? Feature importance (explainable AI)", margin=dict(l=0, r=0, t=40, b=0)), use_container_width=True)
        st.markdown('<div class="warn"><b>Suggestion with caution:</b> Use this only as one input among many. Combine with chart reading, news and your risk limit. Always use a stop-loss. The final decision and the risk are yours.</div>', unsafe_allow_html=True)

with tabs[4]:
    st.subheader("🏆 Model Performance: RandomForest vs LSTM")
    if os.path.exists("metrics.json"):
        M = json.load(open("metrics.json"))
        st.caption(f"Trained on {len(M['assets'])} assets, {M['train_rows']} train rows, {M['test_rows']} unseen test rows. Always-UP baseline: {M['baseline_always_up']*100:.1f}%")
        names = ["accuracy", "precision", "recall", "f1"]
        bar = go.Figure([go.Bar(name=m, x=names, y=[M[m][k] * 100 for k in names]) for m in ["RandomForest", "LSTM"]])
        bar.update_layout(barmode="group", template="plotly_white", yaxis_title="%", height=360, title="Model comparison")
        st.plotly_chart(bar, use_container_width=True)
        cc = st.columns(2)
        for col, m in zip(cc, ["RandomForest", "LSTM"]):
            z = np.array(M[m]["confusion_matrix"])
            col.plotly_chart(go.Figure(go.Heatmap(z=z, x=["Pred DOWN", "Pred UP"], y=["Actual DOWN", "Actual UP"], text=z, texttemplate="%{text}", colorscale="Blues")).update_layout(
                title=f"{m} confusion matrix", height=320), use_container_width=True)
    else:
        st.info("Run `python train_models.py` once to create metrics.json, rf_model.pkl and lstm_model.h5, then refresh.")
    with st.expander("🎓 Viva: why these models? what is data leakage?"):
        st.markdown("""
**RandomForest = baseline.** Simple, fast, explainable (feature importance), resists overfitting. A complex model must beat it to be justified.  
**LSTM = for time-series.** It has memory: *forget gate* (what old info to drop), *input gate* (what new info to store), *output gate* (what to pass on), plus a *cell state* carrying long-term memory - so it reads the 60-day sequence in order.  
**Data leakage prevention:** (1) time split 80-20 with *no shuffle*; (2) scaler fitted on train only; (3) all features use only today/past data; the future shift is used only for the label; (4) validation = last part of train, in time order.  
**Why accuracy ≈ 50-56%?** Markets are close to random (Efficient Market Hypothesis). Beating the naive baseline consistently is already a good result.""")

# ------------------------------------------------------------------ PAPER TRADING
with tabs[5]:
    st.subheader("🎮 Practice Trading with Virtual Money (zero risk)")
    ss = st.session_state
    ss.setdefault("cash", 100000.0); ss.setdefault("hold", {}); ss.setdefault("log", [])
    st.caption("You start with ₹1,00,000 virtual cash. Prices are real market prices of the selected asset. Resets when you refresh.")
    qty = st.number_input("Quantity", 1, 100000, 1)
    b, s_ = st.columns(2)
    if b.button(f"Buy {qty} @ {last:,.2f}"):
        cost = qty * last
        if cost <= ss.cash: ss.cash -= cost; ss.hold[ticker] = ss.hold.get(ticker, [0, 0]); h = ss.hold[ticker]; ss.hold[ticker] = [h[0] + qty, h[1] + cost]; ss.log.append(f"BUY {qty} {ticker} @ {last:,.2f}")
        else: st.error("Not enough virtual cash.")
    if s_.button(f"Sell {qty} @ {last:,.2f}"):
        h = ss.hold.get(ticker, [0, 0])
        if h[0] >= qty: avg = h[1] / h[0]; ss.cash += qty * last; ss.hold[ticker] = [h[0] - qty, avg * (h[0] - qty)]; ss.log.append(f"SELL {qty} {ticker} @ {last:,.2f} (P/L {(last-avg)*qty:+,.2f})")
        else: st.error("You don't hold that many.")
    st.metric("Virtual cash", f"₹{ss.cash:,.2f}")
    if any(v[0] for v in ss.hold.values()):
        st.dataframe(pd.DataFrame([[k, v[0], round(v[1] / v[0], 2)] for k, v in ss.hold.items() if v[0]], columns=["Asset", "Qty", "Avg Buy"]), hide_index=True)
    st.write(ss.log[-10:] if ss.log else "No trades yet.")

# ------------------------------------------------------------------ LEARN
with tabs[6]:
    st.subheader("📚 Complete Trading Manual - From Zero")
    L = {
     "Level 1 - Basics": """**Stock** = a small ownership share of a company. **Exchange** = marketplace (NSE, BSE). **Index** = basket showing market mood (Nifty 50 = top 50 NSE companies, Sensex = 30 BSE).
**Demat account** holds your shares electronically; **trading account** places orders; both link to your bank. **Bull market** = rising, **bear market** = falling.
**IPO** = company's first sale to public. **Dividend** = profit share. **Market cap** = price × total shares (large / mid / small cap).
**Trading hours (India):** 9:15 AM - 3:30 PM, Mon-Fri, excluding holidays.""",
     "Level 2 - Orders & Terms": """**Market order** (buy at current price) vs **Limit order** (buy only at your price). **Stop-loss** = automatic exit to cap your loss.
**Delivery** = hold for days/years. **Intraday** = buy & sell same day (highest risk for beginners). **Bid/Ask** = best buyer / seller price. **Volume** = shares traded.
**Mutual fund / SIP** = pooled money invested regularly (best beginner route). **ETF** = fund traded like a stock. **F&O (futures & options)** = leveraged derivatives; most retail traders lose money - avoid until experienced.""",
     "Level 3 - Fundamental Analysis (is it a good company?)": """Check: **P/E ratio** (price vs earnings), **EPS growth**, **ROE/ROCE**, **Debt-to-Equity** (lower is safer), **profit & sales growth** for 5 years, promoter holding, management quality, industry outlook.
Rule: buy *businesses*, not tickers. Read annual reports and quarterly results.""",
     "Level 4 - Technical Analysis (when to buy/sell?)": """**Candlestick** shows open/high/low/close. **Support** = price floor, **Resistance** = ceiling. **MA20/MA50** = average of last 20/50 closes; price above them = uptrend bias.
**RSI(14)** >70 overbought, <30 oversold. **MACD** crossing above its signal = bullish momentum. **Bollinger Bands** = volatility envelope. **Volume** confirms moves.
Use *several* indicators together; none is perfect. Practice in the Practice tab first.""",
     "Level 5 - Crypto, Gold, Forex": """**Crypto** (Bitcoin, Ethereum) trades 24×7, extremely volatile; in India profits are taxed heavily and TDS applies - verify current rules. **Gold** is a classic safe-haven; consider Gold ETFs / sovereign gold bonds.
**Forex** = currency pairs (USD/INR); leverage makes it dangerous; trade only via regulated Indian brokers in permitted pairs.""",
     "Level 6 - Risk Management (most important)": """1) Risk only **1-2% of capital per trade**. 2) Always set a **stop-loss**. 3) **Diversify** across sectors/assets. 4) Never use borrowed money. 5) Keep 6 months emergency fund *outside* markets.
6) Don't revenge-trade after losses. 7) Keep a trade journal. 8) Ignore tips from Telegram/WhatsApp/YouTube "gurus".""",
     "Level 7 - Taxes (India, verify latest rules)": """Short-term capital gains (equity held <1 year) and long-term gains (>1 year) are taxed at different rates, with an annual LTCG exemption threshold. Intraday/F&O counts as business income. Crypto gains have special flat tax + TDS. Rules change every budget - check incometax.gov.in or a CA.""",
     "Learning Roadmap": """Month 1: basics + open demat, paper trade. Month 2: read financial statements, start a small SIP. Month 3-6: technical analysis + journaling with small real amounts. After 1 year: review journal honestly before increasing capital."""}
    for k, v in L.items():
        with st.expander(k, expanded=k.startswith("Level 1")): st.markdown(v)

# ------------------------------------------------------------------ BROKERS
with tabs[7]:
    st.subheader("🏦 Trading Apps & How to Start")
    st.dataframe(pd.DataFrame([
        ["Groww", "Stocks, MF, ETF, F&O", "Very simple UI; popular with first-timers"],
        ["Zerodha (Kite + Coin)", "Stocks, MF, F&O, commodities", "Powerful charts; Varsity free learning"],
        ["Angel One", "Stocks, MF, F&O, commodities", "Full-service features, research & tips"],
        ["Upstox", "Stocks, MF, F&O", "Fast app, good charting"],
        ["Dhan / 5paisa / Paytm Money", "Stocks, MF, F&O", "Low-cost alternatives"],
        ["ICICI Direct / HDFC Securities / Kotak", "Full-service", "Bank-linked, higher charges, good support"]],
        columns=["App / Broker", "What you can do", "Good for"]), hide_index=True, use_container_width=True)
    st.markdown("""**Steps to start:** ① Keep PAN, Aadhaar (linked to mobile), bank account, e-signature ready → ② Choose a **SEBI-registered** broker → ③ Complete e-KYC → ④ Account opens in ~1-3 days → ⑤ Add small amount → ⑥ Place first *delivery* order of a large-cap stock or index ETF.
**Compare:** account-opening/AMC fees, brokerage per order, hidden charges, app stability, customer support. Fees change - check the broker's official page.
**Verify any broker/advisor** on SEBI's website (sebi.gov.in) before sending money.
Free official learning: Zerodha Varsity, NSE Academy, SEBI Investor Education (investor.sebi.gov.in).""")

# ------------------------------------------------------------------ SAFETY
with tabs[8]:
    st.subheader("🛡️ Precautions - How Not to Lose Your Money")
    st.markdown("""<div class="warn"><b>Reality check:</b> Most retail traders (especially in intraday & F&O) lose money. There is no risk-free profit.</div>""", unsafe_allow_html=True)
    st.markdown("""- ❌ Don't invest rent, loans, emergency or education money.
- ❌ Beware of **guaranteed returns**, "100% accurate" signals, paid tips, unregistered advisors, and fake trading apps.
- ❌ Never share OTP/PIN/password; use 2-factor authentication. Check app/website URL before login.
- ❌ Avoid leverage/F&O/penny stocks until you have 1+ year of consistent paper trading.
- ✅ Start with index funds / large caps, invest regularly (SIP), diversify.
- ✅ Use stop-loss on every trade; never average down on a losing speculative bet.
- ✅ Think long term (3-5+ years) for equity; keep emotions out.
- ✅ Report fraud: cybercrime.gov.in / helpline 1930.""")
    st.markdown("#### 🧮 Position-size calculator")
    cc = st.columns(4)
    cap = cc[0].number_input("Capital (₹)", 1000.0, 1e9, 100000.0); rk = cc[1].slider("Risk per trade %", 0.5, 5.0, 1.0, 0.5)
    ent = cc[2].number_input("Entry price", 0.01, 1e7, round(last, 2)); sl = cc[3].number_input("Stop-loss price", 0.01, 1e7, round(last * 0.95, 2))
    if ent > sl:
        qty_ = int((cap * rk / 100) / (ent - sl)); st.success(f"Max quantity: **{qty_}** | Max loss if stop hits: ₹{qty_*(ent-sl):,.0f} | Capital used: ₹{qty_*ent:,.0f}")
    else: st.warning("Stop-loss must be below entry for a buy trade.")

# ------------------------------------------------------------------ TEAM
with tabs[9]:
    st.subheader("👤 Project Details")
    st.markdown("""**Project:** Intelligent Stock Market Trend Predictor using ML  
**Developer:** Bhanu Kiran Bonam  
**Program:** B.Tech Computer Science Engineering (2022-2026)  
**College:** DNR College of Engineering and Technology, Bhimavaram  
**Tech:** Python • Streamlit • yfinance • pandas • ta • scikit-learn • TensorFlow/Keras • Plotly  
**Future scope:** news sentiment analysis, portfolio optimisation, more models (XGBoost, GRU, Transformers).""")
    st.markdown(DISCLAIMER)
