"""
BTC Leading Indicator - XRP Trading Dashboard
==============================================

Streamlit Cloud dashboard for monitoring your XRP trading bot.
Uses public Binance API for market data (works from any location).
Account balances use authenticated API (may be geo-restricted).
"""

import streamlit as st
import requests
import hmac
import hashlib
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode

# --- Page Config ---
st.set_page_config(
    page_title="XRP Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Binance API Helpers (no python-binance dependency) ---
BINANCE_BASE = "https://api.binance.com"
BINANCE_US_BASE = "https://api.binance.us"


def _public_get(endpoint, params=None):
    """Call a public Binance API endpoint. Falls back to Binance US if blocked."""
    for base in [BINANCE_BASE, BINANCE_US_BASE]:
        try:
            r = requests.get(f"{base}{endpoint}", params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


def _signed_get(endpoint, params=None):
    """Call a signed (authenticated) Binance API endpoint."""
    try:
        api_key = st.secrets["BINANCE_API_KEY"]
        api_secret = st.secrets["BINANCE_API_SECRET"]
    except Exception:
        return None

    if params is None:
        params = {}
    params["timestamp"] = int(time.time() * 1000)
    query_string = urlencode(params)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = signature
    headers = {"X-MBX-APIKEY": api_key}

    for base in [BINANCE_BASE, BINANCE_US_BASE]:
        try:
            r = requests.get(
                f"{base}{endpoint}", params=params, headers=headers, timeout=10
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


# --- Data Functions ---
def get_balances():
    """Get XRP and USDT balances from authenticated API."""
    data = _signed_get("/api/v3/account")
    if data and "balances" in data:
        xrp = 0.0
        usdt = 0.0
        for b in data["balances"]:
            if b["asset"] == "XRP":
                xrp = float(b["free"]) + float(b["locked"])
            elif b["asset"] == "USDT":
                usdt = float(b["free"]) + float(b["locked"])
        return xrp, usdt, True
    # Geo-blocked — return last known values
    return 181.62, 0.0, False


def get_price(symbol="XRPUSDT"):
    """Get current price (public endpoint, works everywhere)."""
    data = _public_get("/api/v3/ticker/price", {"symbol": symbol})
    if data:
        return float(data["price"])
    return 0.0


def get_klines(symbol, interval="1h", limit=200):
    """Get historical klines (public endpoint, works everywhere)."""
    data = _public_get(
        "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit}
    )
    if not data:
        return [], [], [], []
    closes = [float(k[4]) for k in data]
    highs = [float(k[2]) for k in data]
    lows = [float(k[3]) for k in data]
    timestamps = [datetime.fromtimestamp(k[0] / 1000) for k in data]
    return timestamps, closes, highs, lows


# --- Indicator Functions ---
def calculate_roc(closes, period=6):
    """Calculate Rate of Change as percentage."""
    if len(closes) < period + 1:
        return 0.0, 0.0
    current = ((closes[-1] - closes[-1 - period]) / closes[-1 - period]) * 100
    previous = ((closes[-2] - closes[-2 - period]) / closes[-2 - period]) * 100
    return current, previous


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return 0.0
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_atr(highs, lows, closes, period=10):
    """Calculate Average True Range."""
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period


def generate_signal(
    btc_closes, roc_threshold=0.3, ema_fast_period=8, ema_slow_period=21, roc_period=6
):
    """Generate the trading signal from BTC data."""
    btc_roc, btc_roc_prev = calculate_roc(btc_closes, roc_period)
    btc_ema_fast = calculate_ema(btc_closes, ema_fast_period)
    btc_ema_slow = calculate_ema(btc_closes, ema_slow_period)

    btc_trend_up = btc_ema_fast > btc_ema_slow
    btc_trend_down = btc_ema_fast < btc_ema_slow

    long_signal = btc_roc > roc_threshold and btc_roc > btc_roc_prev and btc_trend_up
    short_signal = (
        btc_roc < -roc_threshold and btc_roc < btc_roc_prev and btc_trend_down
    )

    if long_signal:
        signal = "LONG"
    elif short_signal:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return signal, btc_roc, btc_ema_fast, btc_ema_slow


# --- Strategy Parameters ---
STRATEGY_PARAMS = {
    "roc_length": 6,
    "roc_threshold": 0.3,
    "ema_fast": 8,
    "ema_slow": 21,
    "atr_length": 10,
    "atr_mult_sl": 2.0,
    "atr_mult_tp": 3.0,
}


# --- Main App ---
def main():
    # Header
    st.markdown(
        '<h1 style="color: #58a6ff;">📊 BTC Leading Indicator — XRP Trading Bot</h1>',
        unsafe_allow_html=True,
    )

    # Fetch all data
    xrp_bal, usdt_bal, balances_live = get_balances()
    xrp_price = get_price("XRPUSDT")
    btc_price = get_price("BTCUSDT")
    xrp_value = xrp_bal * xrp_price
    total_value = xrp_value + usdt_bal

    btc_ts, btc_closes, btc_highs, btc_lows = get_klines("BTCUSDT", "1h", 200)
    xrp_ts, xrp_closes, xrp_highs, xrp_lows = get_klines("XRPUSDT", "1h", 200)

    signal, btc_roc, btc_ema_fast, btc_ema_slow = generate_signal(
        btc_closes,
        roc_threshold=STRATEGY_PARAMS["roc_threshold"],
        ema_fast_period=STRATEGY_PARAMS["ema_fast"],
        ema_slow_period=STRATEGY_PARAMS["ema_slow"],
        roc_period=STRATEGY_PARAMS["roc_length"],
    )

    # --- Row 1: Signal (the most important thing) ---
    if signal == "LONG":
        st.success(f"📈 **BULLISH SIGNAL** — BTC ROC: {btc_roc:+.2f}% — Bot is buying/holding XRP")
    elif signal == "SHORT":
        st.error(f"📉 **BEARISH SIGNAL** — BTC ROC: {btc_roc:+.2f}% — Bot is selling XRP")
    else:
        st.info(f"➡️ **NEUTRAL** — BTC ROC: {btc_roc:+.2f}% — No action")

    # --- Row 2: Prices ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("XRP Price", f"${xrp_price:.4f}")
    with col2:
        st.metric("BTC Price", f"${btc_price:,.0f}")
    with col3:
        st.metric("BTC Momentum", f"{btc_roc:+.2f}%",
                  delta=f"Threshold: ±{STRATEGY_PARAMS['roc_threshold']}%")

    # --- Row 3: Account balances (if available) ---
    if balances_live:
        position = "IN XRP" if xrp_value > usdt_bal else "IN USDT"
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("XRP Balance", f"{xrp_bal:.2f} XRP", f"${xrp_value:.2f}")
        with col2:
            st.metric("USDT Balance", f"${usdt_bal:.2f}")
        with col3:
            st.metric("Total Portfolio", f"${total_value:.2f}")
    else:
        st.caption(
            "💡 Live account balances unavailable (Binance API is geo-restricted from this server). "
            "Check your local dashboard at localhost:8080 for balances. "
            "All market data and signals above are live."
        )

    st.divider()

    # --- XRP Price Chart ---
    st.subheader("XRP Price — Last 7 Days")
    if xrp_ts and xrp_closes:
        chart_df = pd.DataFrame(
            {"Time": xrp_ts[-168:], "Price (USD)": xrp_closes[-168:]}
        )
        chart_df = chart_df.set_index("Time")
        st.line_chart(chart_df, color="#58a6ff")

    # --- BTC Momentum Chart ---
    st.subheader("BTC Momentum (ROC) — Last 7 Days")
    if btc_closes and len(btc_closes) > STRATEGY_PARAMS["roc_length"] + 1:
        roc_values = []
        roc_times = []
        period = STRATEGY_PARAMS["roc_length"]
        for i in range(period, len(btc_closes)):
            roc = (
                (btc_closes[i] - btc_closes[i - period]) / btc_closes[i - period]
            ) * 100
            roc_values.append(roc)
            if i < len(btc_ts):
                roc_times.append(btc_ts[i])

        roc_df = pd.DataFrame(
            {"Time": roc_times[-168:], "BTC ROC (%)": roc_values[-168:]}
        )
        roc_df = roc_df.set_index("Time")
        st.line_chart(roc_df, color="#f0883e")

        st.caption(
            f"Threshold: ±{STRATEGY_PARAMS['roc_threshold']}% · "
            f"Above = Bullish · Below = Bearish · Between = Neutral"
        )

    st.divider()

    # --- Strategy Parameters ---
    st.subheader("Active Strategy Parameters")
    pcol1, pcol2, pcol3, pcol4, pcol5, pcol6, pcol7 = st.columns(7)
    pcol1.metric("ROC Length", STRATEGY_PARAMS["roc_length"])
    pcol2.metric("ROC Threshold", f"{STRATEGY_PARAMS['roc_threshold']}%")
    pcol3.metric("EMA Fast", STRATEGY_PARAMS["ema_fast"])
    pcol4.metric("EMA Slow", STRATEGY_PARAMS["ema_slow"])
    pcol5.metric("ATR Length", STRATEGY_PARAMS["atr_length"])
    pcol6.metric("ATR SL", f"{STRATEGY_PARAMS['atr_mult_sl']}x")
    pcol7.metric("ATR TP", f"{STRATEGY_PARAMS['atr_mult_tp']}x")

    st.caption(
        "Backtest results: 46.5% win rate · +370.5% return · 398 trades over 2 years · "
        "Parameters are optimized continuously by the local optimizer"
    )

    st.divider()

    # --- Technical Details ---
    with st.expander("Technical Details"):
        st.markdown(f"""
**BTC Indicators:**
- EMA Fast ({STRATEGY_PARAMS['ema_fast']}): ${btc_ema_fast:,.2f}
- EMA Slow ({STRATEGY_PARAMS['ema_slow']}): ${btc_ema_slow:,.2f}
- Trend: {"Uptrend ↑" if btc_ema_fast > btc_ema_slow else "Downtrend ↓"}
- ROC ({STRATEGY_PARAMS['roc_length']}-period): {btc_roc:+.2f}%

**XRP Indicators:**
- ATR ({STRATEGY_PARAMS['atr_length']}): ${calculate_atr(xrp_highs, xrp_lows, xrp_closes, STRATEGY_PARAMS['atr_length']):.4f}
- Current Price: ${xrp_price:.4f}

**Signal Logic:**
- LONG: BTC ROC > {STRATEGY_PARAMS['roc_threshold']}% AND accelerating AND BTC uptrend
- SHORT: BTC ROC < -{STRATEGY_PARAMS['roc_threshold']}% AND accelerating AND BTC downtrend
""")

    # --- Footer ---
    st.divider()
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        st.caption(
            f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            "Trading bot runs locally via launchd"
        )
    with col_f2:
        st.button("🔄 Refresh", use_container_width=True)


if __name__ == "__main__":
    main()
