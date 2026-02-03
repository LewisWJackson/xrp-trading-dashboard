"""
BTC Leading Indicator - XRP Trading Dashboard
==============================================

Streamlit Cloud dashboard for monitoring your XRP trading bot.
Pulls all data directly from the Binance API - no local files needed.

Your trading bot runs locally on your Mac (via launchd).
This dashboard is a read-only monitoring view accessible from any device.
"""

import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client

# --- Page Config ---
st.set_page_config(
    page_title="XRP Trading Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Secrets ---
@st.cache_resource
def get_client():
    """Create a cached Binance client from Streamlit secrets."""
    return Client(
        st.secrets["6JJK9yNHQMKNArrtEJaZ2e1lGcwdhSKF6SSM1ql0acLSB9cuieDMvLXCgRo7s67d"],
        st.secrets["CGn7hEGeAR5BuVHill9vsGCD5te0AE6vnp0fcJE1tmTYJ3gCU8r1qDeEEAZHLZvB"],
    )


# --- Data Functions ---
def get_balances(client):
    """Get XRP and USDT balances."""
    try:
        account = client.get_account()
        xrp = 0.0
        usdt = 0.0
        for b in account["balances"]:
            if b["asset"] == "XRP":
                xrp = float(b["free"]) + float(b["locked"])
            elif b["asset"] == "USDT":
                usdt = float(b["free"]) + float(b["locked"])
        return xrp, usdt
    except Exception as e:
        st.error(f"Balance fetch error: {e}")
        return 0.0, 0.0


def get_price(client, symbol="XRPUSDT"):
    """Get current price for a symbol."""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception:
        return 0.0


def get_klines(client, symbol, interval="1h", limit=200):
    """Get historical kline data."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        timestamps = [datetime.fromtimestamp(k[0] / 1000) for k in klines]
        return timestamps, closes, highs, lows
    except Exception as e:
        st.error(f"Klines error: {e}")
        return [], [], [], []


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


def generate_signal(btc_closes, roc_threshold=0.3, ema_fast_period=8, ema_slow_period=21, roc_period=6):
    """Generate the trading signal from BTC data."""
    btc_roc, btc_roc_prev = calculate_roc(btc_closes, roc_period)
    btc_ema_fast = calculate_ema(btc_closes, ema_fast_period)
    btc_ema_slow = calculate_ema(btc_closes, ema_slow_period)

    btc_trend_up = btc_ema_fast > btc_ema_slow
    btc_trend_down = btc_ema_fast < btc_ema_slow

    long_signal = btc_roc > roc_threshold and btc_roc > btc_roc_prev and btc_trend_up
    short_signal = btc_roc < -roc_threshold and btc_roc < btc_roc_prev and btc_trend_down

    if long_signal:
        signal = "LONG"
    elif short_signal:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return signal, btc_roc, btc_ema_fast, btc_ema_slow


def get_xrp_price_history(client, days=7):
    """Get XRP price history for charting."""
    try:
        interval = Client.KLINE_INTERVAL_1HOUR
        klines = client.get_klines(
            symbol="XRPUSDT",
            interval=interval,
            limit=min(days * 24, 1000),
        )
        timestamps = [datetime.fromtimestamp(k[0] / 1000) for k in klines]
        closes = [float(k[4]) for k in klines]
        return timestamps, closes
    except Exception:
        return [], []


# --- Strategy Parameters (baseline, updated by optimizer) ---
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
    client = get_client()

    # Header
    st.markdown(
        '<h1 style="color: #58a6ff;">📊 BTC Leading Indicator — XRP Trading Bot</h1>',
        unsafe_allow_html=True,
    )

    # Fetch all data
    xrp_bal, usdt_bal = get_balances(client)
    xrp_price = get_price(client, "XRPUSDT")
    btc_price = get_price(client, "BTCUSDT")
    xrp_value = xrp_bal * xrp_price
    total_value = xrp_value + usdt_bal

    btc_ts, btc_closes, btc_highs, btc_lows = get_klines(client, "BTCUSDT", "1h", 200)
    xrp_ts, xrp_closes, xrp_highs, xrp_lows = get_klines(client, "XRPUSDT", "1h", 200)

    signal, btc_roc, btc_ema_fast, btc_ema_slow = generate_signal(
        btc_closes,
        roc_threshold=STRATEGY_PARAMS["roc_threshold"],
        ema_fast_period=STRATEGY_PARAMS["ema_fast"],
        ema_slow_period=STRATEGY_PARAMS["ema_slow"],
        roc_period=STRATEGY_PARAMS["roc_length"],
    )

    # Position detection
    position = "IN XRP" if xrp_value > usdt_bal else "IN USDT"

    # --- Row 1: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("XRP Balance", f"{xrp_bal:.2f} XRP", f"${xrp_value:.2f}")

    with col2:
        st.metric("USDT Balance", f"${usdt_bal:.2f}")

    with col3:
        st.metric("XRP Price", f"${xrp_price:.4f}")

    with col4:
        st.metric("BTC Price", f"${btc_price:,.0f}")

    # --- Row 2: Signal + Position + Portfolio ---
    col1, col2, col3 = st.columns(3)

    with col1:
        if signal == "LONG":
            st.success(f"📈 **BULLISH** — BTC ROC: {btc_roc:+.2f}%")
            st.caption("Signal: Hold / Buy XRP")
        elif signal == "SHORT":
            st.error(f"📉 **BEARISH** — BTC ROC: {btc_roc:+.2f}%")
            st.caption("Signal: Sell XRP")
        else:
            st.info(f"➡️ **NEUTRAL** — BTC ROC: {btc_roc:+.2f}%")
            st.caption("Signal: No action")

    with col2:
        if position == "IN XRP":
            st.success(f"🟢 **{position}**")
            st.caption("Waiting for sell signal")
        else:
            st.warning(f"🟡 **{position}**")
            st.caption("Waiting for buy signal")

    with col3:
        st.metric("Total Portfolio", f"${total_value:.2f}")

    st.divider()

    # --- XRP Price Chart ---
    st.subheader("XRP Price — Last 7 Days")
    if xrp_ts and xrp_closes:
        import pandas as pd

        chart_df = pd.DataFrame({"Time": xrp_ts[-168:], "Price (USD)": xrp_closes[-168:]})
        chart_df = chart_df.set_index("Time")
        st.line_chart(chart_df, color="#58a6ff")

    # --- BTC Momentum Chart ---
    st.subheader("BTC Momentum (ROC) — Last 7 Days")
    if btc_closes and len(btc_closes) > STRATEGY_PARAMS["roc_length"] + 1:
        import pandas as pd

        roc_values = []
        roc_times = []
        period = STRATEGY_PARAMS["roc_length"]
        for i in range(period, len(btc_closes)):
            roc = ((btc_closes[i] - btc_closes[i - period]) / btc_closes[i - period]) * 100
            roc_values.append(roc)
            if i < len(btc_ts):
                roc_times.append(btc_ts[i])

        # Show last 168 hours (7 days)
        roc_df = pd.DataFrame({"Time": roc_times[-168:], "BTC ROC (%)": roc_values[-168:]})
        roc_df = roc_df.set_index("Time")
        st.line_chart(roc_df, color="#f0883e")

        # Threshold lines info
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
