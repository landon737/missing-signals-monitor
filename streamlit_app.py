"""
Missing Signals Monitor – Live Dashboard Prototype (Streamlit)

This app simulates a "live" version of the dashboard we designed together.
Right now it uses mock data, but the structure is ready for you to plug in
real APIs (prices, liquidity, on-chain, etc.).

How to run:

1. Install dependencies (in a terminal):
   pip install streamlit pandas numpy plotly

2. Save this file as: missing_signals_live_dashboard.py

3. Run the app:
   streamlit run missing_signals_live_dashboard.py

Then open the URL it prints (usually http://localhost:8501) in your browser.
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import time
from requests.exceptions import HTTPError

# -------------------------
# Config & page layout
# -------------------------

st.set_page_config(
    page_title="Missing Signals Monitor – Live Prototype",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------
# Mock data generators
# (replace these with real data sources later)
# -------------------------

def generate_mock_times(hours: int = 12, step_minutes: int = 10) -> pd.DatetimeIndex:
    now = dt.datetime.utcnow().replace(second=0, microsecond=0)
    periods = int(hours * 60 / step_minutes)
    times = pd.date_range(end=now, periods=periods, freq=f"{step_minutes}min")
    return times


def generate_mock_streams(hours: int = 12, step_minutes: int = 10) -> pd.DataFrame:
    """Generate simulated BTC price, liquidity, peg and risk score streams.

    This is just for visual demo. Later, swap these out with:
    - Exchange price feeds
    - Liquidity metrics (TGA, RRP, CB balance sheets)
    - Stablecoin peg prices
    - Your own risk model
    """
    times = generate_mock_times(hours, step_minutes)
    n = len(times)

    # BTC price – random walk + one shock event
    base_price = 42000
    rng = np.random.default_rng(seed=int(dt.datetime.utcnow().timestamp()) // 60)  # change every minute
    noise = rng.normal(0, 40, n)
    btc = base_price + np.cumsum(noise)

    # Inject an 8% move around 3/4 of the series
    event_idx = int(n * 0.75)
    direction = rng.choice([-1, 1])
    btc[event_idx:] += direction * base_price * 0.08

    # Liquidity index – 0–100, drifts a bit
    liq = 60 + np.sin(np.linspace(0, 5, n)) * 10 + rng.normal(0, 2, n)
    if direction < 0:
        liq[event_idx:] -= 10  # tightening if price drops
    else:
        liq[event_idx:] += 10  # easing if price spikes

    # Stablecoin peg (e.g., USDC)
    peg = 1 + rng.normal(0, 0.0005, n)
    peg[event_idx - 2:event_idx + 3] += -0.004 * direction  # wobble around event

    # Composite risk score 0–100
    risk = 30 + rng.normal(0, 5, n)
    risk[event_idx - 4:] += 40
    risk = np.clip(risk, 0, 100)

    df = pd.DataFrame(
        {
            "time": times,
            "btc": btc,
            "liquidity_index": liq,
            "peg": peg,
            "risk_score": risk,
            "event_flag": 0,
        }
    )
    df.loc[event_idx, "event_flag"] = 1
    return df

# -------------------------
# Live BTC price (CoinGecko)
# -------------------------

def get_live_btc_price_usd() -> float | None:
    """Fetch live BTC price in USD from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        return float(data["bitcoin"]["usd"])
    except Exception as e:
        # Show a warning in the app, but don't crash
        st.warning(f"Live BTC price fetch failed: {e}")
        return None

# -------------------------
# Live USDC "peg" via CoinGecko
# -------------------------

def get_live_usdc_peg_coingecko() -> float | None:
    """
    Approximate USDC/USDT by taking USDC/USD and USDT/USD from CoinGecko
    and computing their ratio.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "usd-coin,tether",
            "vs_currencies": "usd",
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        usdc_usd = float(data["usd-coin"]["usd"])
        usdt_usd = float(data["tether"]["usd"])
        if usdt_usd == 0:            raise ValueError("USDT/USD is zero")
        return usdc_usd / usdt_usd
    except Exception as e:
        st.warning(f"Live USDC peg fetch (CoinGecko) failed: {e}")
        return None

# -------------------------
# Crypto Liquidity Proxy (CoinGecko Global)
# -------------------------

def get_crypto_liquidity_proxy() -> dict | None:
    """
    Fetch basic global crypto market data from CoinGecko:
    - Total market cap (USD)
    - 24h volume (USD)
    - BTC dominance (%)
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()["data"]

        total_mcap = float(data["total_market_cap"]["usd"])
        total_volume = float(data["total_volume"]["usd"])
        btc_dominance = float(data["market_cap_percentage"]["btc"])

        return {
            "total_mcap": total_mcap,
            "total_volume": total_volume,
            "btc_dominance": btc_dominance,
        }
    except Exception as e:
        st.warning(f"Crypto liquidity proxy fetch failed: {e}")
        return None

# -------------------------
# Exchange API health / latency
# -------------------------

def check_exchange_health(name: str, url: str, timeout: float = 3.0) -> dict:
    """
    Ping a public API endpoint and measure latency (ms).
    Returns a dict: {name, latency_ms, ok, error}
    """
    start = time.monotonic()
    try:
        r = requests.get(url, timeout=timeout)
        latency_ms = (time.monotonic() - start) * 1000
        r.raise_for_status()
        return {
            "name": name,
            "latency_ms": latency_ms,
            "ok": True,
            "error": None,
        }
    except HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        return {
            "name": name,
            "latency_ms": None,
            "ok": False,
            "error": f"HTTP {status}" if status is not None else "HTTPError",
        }
    except Exception as e:
        return {
            "name": name,
            "latency_ms": None,
            "ok": False,
            "error": type(e).__name__,
        }

# -------------------------
# Narrative pulse (NewsAPI)
# -------------------------

def get_narrative_pulse() -> dict | None:
    """
    Use NewsAPI to get a simple 'narrative pulse' for crypto:
    - total articles (last 24h)
    - counts for key themes: ETF, hack, regulation, SEC, macro (CPI/Fed)
    """
    api_key = st.secrets.get("newsapi_key")
    if not api_key:
        st.info("No NewsAPI key configured; skipping narrative pulse.")
        return None

    try:
        url = "https://newsapi.org/v2/everything"
        from_date = (dt.datetime.utcnow() - dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "q": "crypto OR bitcoin OR ethereum",
            "language": "en",
            "sortBy": "publishedAt",
            "from": from_date,
            "pageSize": 50,
            "apiKey": api_key,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()

        articles = data.get("articles", [])
        total = len(articles)

        themes = {
            "ETF": 0,
            "hack": 0,
            "regulation": 0,
            "SEC": 0,
            "CPI/Fed": 0,
        }

        for a in articles:
            text = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
            if "etf" in text:
                themes["ETF"] += 1
            if "hack" in text or "exploit" in text:
                themes["hack"] += 1
            if "regulat" in text or "ban" in text:
                themes["regulation"] += 1
            if "sec" in text:
                themes["SEC"] += 1
            if "cpi" in text or "fed " in text or "fomc" in text:
                themes["CPI/Fed"] += 1

        return {"total": total, "themes": themes}

    except Exception as e:
        st.warning(f"Narrative pulse fetch failed: {e}")
        return None

# -------------------------
# Risk interpretation
# -------------------------

def interpret_risk(score: float) -> tuple[str, str]:
    """Return (label, description) for a given risk score."""
    if score >= 70:
        return (
            "HIGH RISK – ±8% MOVE LIKELY",
            "Multiple stress signals are elevated. Consider tighter risk, staggered entries, or waiting for volatility to settle.",
        )
    elif score >= 40:
        return (
            "MODERATE RISK – VOLATILITY ELEVATED",
            "Conditions are choppy. Position sizing and time horizons matter more than usual.",
        )
    else:
        return (
            "LOW RISK – NORMAL CONDITIONS",
            "No major stress signals. Market behaviour is closer to its typical regime.",
        )


# -------------------------
# Sidebar controls
# -------------------------

st.sidebar.title("Controls")

hours = st.sidebar.slider("History window (hours)", min_value=3, max_value=48, value=12, step=3)
step_minutes = st.sidebar.selectbox("Time step", options=[5, 10, 15, 30], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Thresholds (for explanation only)**")
high_thr = st.sidebar.slider("High risk threshold", 60, 90, 70)
med_thr = st.sidebar.slider("Moderate risk threshold", 30, 70, 40)

st.sidebar.markdown("---")
st.sidebar.caption(
    "This prototype uses simulated data so you can see how the dashboard *behaves*. "
    "Later, replace the generator with live market / on-chain feeds."
)


# -------------------------
# Data
# -------------------------

df = generate_mock_streams(hours=hours, step_minutes=step_minutes)
event_time = df.loc[df["event_flag"] == 1, "time"].iloc[0]
latest = df.iloc[-1]


# -------------------------
# Layout: title
# -------------------------

st.markdown(
    "<h2 style='margin-bottom:0.2rem;'>Missing Signals Monitor – Live Prototype</h2>"
    "<span style='color:gray;'>Visualising BTC, liquidity, peg stability and composite risk in one view.</span>",
    unsafe_allow_html=True,
)

st.markdown(
    "<small style='color:#888;'>Build: <b>v1.1 – live BTC metric enabled</b></small>",
    unsafe_allow_html=True,
)


st.markdown(
    f"<small>Last update (UTC): <b>{latest['time']:%Y-%m-%d %H:%M}</b></small>",
    unsafe_allow_html=True,
)

st.markdown("---")


# -------------------------
# Top: BTC price panel
# -------------------------

btc_col = st.container()
with btc_col:
    st.markdown("#### BTC Price & Event Reaction")
    fig_btc = px.line(df, x="time", y="btc", template="plotly_dark")
    fig_btc.add_vline(x=event_time, line_dash="dash", line_color="orange")
    fig_btc.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="BTC Price (USD)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_btc, use_container_width=True)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)

    # Column 1: live BTC from CoinGecko (with fallback)
    with c1:
        live_price = get_live_btc_price_usd()
        if live_price is not None:
            st.metric("Live BTC (CoinGecko)", f"{live_price:,.0f} USD")
        else:
            st.metric("Simulated BTC (fallback)", f"{latest['btc']:,.0f} USD")

    # Column 2: move since event
    with c2:
        pct_from_event = (latest["btc"] - df.loc[df["event_flag"] == 1, "btc"].iloc[0]) / df.loc[df["event_flag"] == 1, "btc"].iloc[0] * 100
        st.metric("Move since event", f"{pct_from_event:+.2f}%")

    # Column 3: move in window
    with c3:
        intraday_ret = (latest["btc"] - df["btc"].iloc[0]) / df["btc"].iloc[0] * 100
        st.metric("Move in window", f"{intraday_ret:+.2f}%")

    # Column 4: event time
    with c4:
        st.metric("Event time", event_time.strftime("%Y-%m-%d %H:%M"))



# -------------------------
# Bottom: 2-column layout
# -------------------------

left_col, right_col = st.columns([2, 2])

# LEFT COLUMN – Liquidity + Peg
with left_col:
    st.markdown("#### Global Liquidity & Stablecoin Peg (v1.3)")

    sub1, sub2 = st.columns(2)

    # Liquidity subpanel
    with sub1:
        st.caption("Liquidity Index & Crypto Market")

        fig_liq = px.line(df, x="time", y="liquidity_index", template="plotly_dark")
        fig_liq.add_vline(x=event_time, line_dash="dash", line_color="orange")
        fig_liq.update_layout(
            xaxis_title="Time",
            yaxis_title="Index",
            height=260,
            margin=dict(l=30, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_liq, use_container_width=True)

        liq_trend = "tightening" if latest["liquidity_index"] < df["liquidity_index"].iloc[0] else "easing or stable"

        lines = [
            f"- **Simulated Liquidity Index:** {latest['liquidity_index']:.1f}",
            f"- **Trend:** {liq_trend.capitalize()}",
        ]

        proxy = get_crypto_liquidity_proxy()
        if proxy is not None:
            total_mcap_trillions = proxy["total_mcap"] / 1e12
            total_vol_billions = proxy["total_volume"] / 1e9
            lines.append(f"- **Total crypto mkt cap:** ${total_mcap_trillions:.2f}T")
            lines.append(f"- **24h volume:** ${total_vol_billions:.1f}B")
            lines.append(f"- **BTC dominance:** {proxy['btc_dominance']:.1f}%")
        else:
            lines.append("- **Crypto market data:** unavailable (using index only)")

        st.markdown("  \n".join(lines))


    # USDC peg subpanel
    with sub2:
        st.caption("USDC Peg vs 1.0000")

        # Plot the simulated peg history for visual context
        fig_peg = px.line(df, x="time", y="peg", template="plotly_dark")
        fig_peg.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig_peg.add_vline(x=event_time, line_dash="dash", line_color="orange")
        fig_peg.update_layout(
            xaxis_title="Time",
            yaxis_title="Price",
            height=260,
            margin=dict(l=30, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_peg, use_container_width=True)

        # Try live peg from CoinGecko, otherwise fall back to simulated data
        live_peg = get_live_usdc_peg_coingecko()
        used_live = live_peg is not None

        if used_live:
            peg_value = live_peg
            source_label = "CoinGecko (USDC/USDT via USD)"
        else:
            peg_value = float(latest["peg"])
            source_label = "Simulated peg (fallback)"


        peg_dev = abs(peg_value - 1.0)
        peg_status = "Normal" if peg_dev <= 0.002 else "Under Pressure"

        st.markdown(
            f"- **Last price:** {peg_value:.4f}  \n"
            f"- **Deviation:** {peg_dev:.4f}  \n"
            f"- **Peg status:** **{peg_status}**  \n"
            f"- **Source:** {source_label}"
        )

# RIGHT COLUMN – Composite Risk
with right_col:
    st.markdown("#### Composite Risk, Alerts & Exchange Health (v1.0 composite)")

    # --- Compute components for composite risk score ---
    intraday_ret = (latest["btc"] - df["btc"].iloc[0]) / df["btc"].iloc[0] * 100
    liq_index = float(latest["liquidity_index"])
    liq_trend = "tightening" if liq_index < df["liquidity_index"].iloc[0] else "easing or stable"

    # Live peg (or fallback)
    live_peg = get_live_usdc_peg_coingecko()
    if live_peg is None:
        live_peg = float(latest["peg"])
    peg_dev = abs(live_peg - 1.0)

    # Exchange health (we'll reuse this later in the health panel)
    exchanges_to_check = [
        ("Binance", "https://api.binance.com/api/v3/time"),
        ("Coinbase", "https://api.coinbase.com/v2/time"),
        ("Kraken", "https://api.kraken.com/0/public/Time"),
    ]
    health_results = [check_exchange_health(name, url) for name, url in exchanges_to_check]

    # Composite risk score
    composite_score, risk_details = compute_composite_risk(
        intraday_ret_pct=intraday_ret,
        liq_index=liq_index,
        liq_trend=liq_trend,
        peg_dev=peg_dev,
        health_results=health_results,
    )

    risk_label, risk_desc = interpret_risk(composite_score)
    if composite_score >= high_thr:
        risk_color = "#e74c3c"  # red
    elif composite_score >= med_thr:
        risk_color = "#f1c40f"  # yellow
    else:
        risk_color = "#2ecc71"  # green

    # Big risk badge
    st.markdown(
        f"""
        <div style="border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.7rem;
                    background-color:{risk_color}20;border:1px solid {risk_color};">
            <span style="font-weight:700;color:{risk_color};">⚠ {risk_label}</span><br>
            <span style="font-size:0.9rem;color:#ddd;">{risk_desc}</span><br>
            <span style="font-size:0.8rem;color:#aaa;">Composite risk score: {composite_score:.1f} / 100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    if latest["risk_score"] >= high_thr:
        risk_color = "#e74c3c"  # red
    elif latest["risk_score"] >= med_thr:
        risk_color = "#f1c40f"  # yellow
    else:
        risk_color = "#2ecc71"  # green



    # Risk history chart
    fig_risk = px.line(df, x="time", y="risk_score", template="plotly_dark")
    fig_risk.add_vline(x=event_time, line_dash="dash", line_color="orange")
    fig_risk.update_layout(
        xaxis_title="Time",
        yaxis_title="Risk Score (0–100)",
        height=260,
        margin=dict(l=30, r=10, t=30, b=30),
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    # Breakdown bullets
    st.markdown("**Signals feeding into this risk score:**")
    st.markdown(
        f"""
        - Price: **{risk_details['price_comment']}**
        - Liquidity: **{risk_details['liq_comment']}**
        - Stablecoin peg: **{risk_details['peg_comment']}**
        - Exchange health: **{risk_details['exch_comment']}**
        """
    )




    # Exchange health section
    st.markdown("---")
    st.markdown("**Exchange API health (latency, lower is better):**")

    lines = []
    for h in health_results:
        if h["ok"] and h["latency_ms"] is not None:
            lines.append(f"- {h['name']}: **{h['latency_ms']:.0f} ms** (OK)")
        else:
            err = h["error"] or "unknown error"
            if isinstance(err, str) and err.startswith("HTTP 451"):
                lines.append(f"- {h['name']}: **blocked from this region (HTTP 451)**")
            else:
                lines.append(f"- {h['name']}: **unreachable** ({err})")


    st.markdown("  \n".join(lines))

    st.caption(
        "If one or more major exchanges become slow or unreachable, this can be an early sign of "
        "stress, outages, or unusual load before price fully reflects it."
    )

st.markdown("---")
st.markdown("#### Narrative Pulse (News) – last 24h")

pulse = get_narrative_pulse()

if pulse is None:
    st.markdown(
        "- Narrative data unavailable. Configure `newsapi_key` in Streamlit secrets to enable this panel."
    )
else:
    total = pulse["total"]
    themes = pulse["themes"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total articles", total)
    c2.metric("ETF mentions", themes["ETF"])
    c3.metric("Hack mentions", themes["hack"])
    c4.metric("Regulation", themes["regulation"])
    c5.metric("CPI/Fed", themes["CPI/Fed"])

    st.caption(
        "This is a simple 'narrative heat' view. Spikes in one or more themes, "
        "combined with liquidity and exchange stress, often precede large moves."
    )

# -------------------------
# Composite risk score (0–100)
# -------------------------

def compute_composite_risk(
    intraday_ret_pct: float,
    liq_index: float,
    liq_trend: str,
    peg_dev: float,
    health_results: list[dict],
) -> tuple[float, dict]:
    """
    Very simple first-pass composite risk score.

    Returns (score_0_100, details_dict).
    """
    score = 0.0

    # 1) Price move in window: big moves → higher risk
    score += min(abs(intraday_ret_pct) * 2.0, 35.0)

    # 2) Liquidity index level & trend
    if liq_index < 50:
        score += 15.0
    elif liq_index < 55:
        score += 8.0

    if liq_trend == "tightening":
        score += 7.0

    # 3) Stablecoin peg deviation
    if peg_dev > 0.010:      # > 1%
        score += 30.0
    elif peg_dev > 0.005:    # 0.5–1%
        score += 15.0
    elif peg_dev > 0.002:    # 0.2–0.5%
        score += 5.0

    # 4) Exchange health
    bad_exchanges = [h for h in health_results if not h["ok"]]
    high_latency = [
        h for h in health_results
        if h["ok"] and h["latency_ms"] is not None and h["latency_ms"] > 800
    ]

    if bad_exchanges:
        score += 10.0
    if high_latency:
        score += 5.0

    score = max(0.0, min(100.0, score))

    details = {
        "price_comment": f"Intraday move: {intraday_ret_pct:+.2f}%",
        "liq_comment": f"Liquidity index: {liq_index:.1f} ({liq_trend})",
        "peg_comment": f"Peg deviation: {peg_dev:.4f}",
        "exch_comment": f"Unhealthy exchanges: {len(bad_exchanges)}, high-latency: {len(high_latency)}",
    }
    return score, details

