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
        if usdt_usd == 0:
            raise ValueError("USDT/USD is zero")
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
    st.markdown("#### Composite Risk & Alerts")

    risk_label, risk_desc = interpret_risk(latest["risk_score"])
    if latest["risk_score"] >= high_thr:
        risk_color = "#e74c3c"  # red
    elif latest["risk_score"] >= med_thr:
        risk_color = "#f1c40f"  # yellow
    else:
        risk_color = "#2ecc71"  # green

    # Big risk badge
    st.markdown(
        f"""
        <div style="border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.7rem;
                    background-color:{risk_color}20;border:1px solid {risk_color};">
            <span style="font-weight:700;color:{risk_color};">⚠ {risk_label}</span><br>
            <span style="font-size:0.9rem;color:#ddd;">{risk_desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown("**Signals feeding into risk (in this prototype):**")
    st.markdown(
        f"""
        - Liquidity: **{'tightening' if latest['liquidity_index'] < 55 else 'neutral/easing'}**
        - BTC: **event shock detected** around {event_time:%H:%M} UTC
        - Stablecoin peg: **{peg_status}**
        - Recent risk trend: {'rising' if latest['risk_score'] > df['risk_score'].iloc[0] else 'falling or flat'}
        """
    )

    st.caption(
        "In a real implementation, this panel would combine many more inputs: "
        "on-chain flows, exchange fragility, macro surprise indices, and narrative velocity."
    )
