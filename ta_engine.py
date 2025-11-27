# ===============================================
#   SWING MODE ENGINE v2 — UNIVERSAL STOCK ANALYZER
#   Supports any NSE/BSE stock (e.g., RELIANCE.NS, TCS.NS, INFY.NS)
#   Computes EMA, RSI, MACD, Support/Resistance & Breakout Signals
#   Generates professional candlestick chart
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import yfinance as yf
import io
import base64


# --- Core Technical Indicator Functions ---

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff().to_numpy()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.squeeze(gain)
    loss = np.squeeze(loss)
    gain_avg = pd.Series(gain).rolling(period).mean()
    loss_avg = pd.Series(loss).rolling(period).mean()
    rsi = 100 - (100 / (1 + (gain_avg / loss_avg)))
    return rsi.fillna(method="bfill")

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# --- Support & Resistance Estimation ---

def detect_support_resistance(df, window=10):
    supports, resistances = [], []
    for i in range(window, len(df) - window):
        low_range = df["Low"].iloc[i - window:i + window]
        high_range = df["High"].iloc[i - window:i + window]
        if df["Low"].iloc[i] == low_range.min():
            supports.append(df["Low"].iloc[i])
        if df["High"].iloc[i] == high_range.max():
            resistances.append(df["High"].iloc[i])
    supports = sorted(list(set([round(x, 2) for x in supports])))[-5:]
    resistances = sorted(list(set([round(x, 2) for x in resistances])))[:5]
    return supports, resistances


# --- Breakout Detection Logic ---

def detect_breakout(df):
    ema20, ema50, ema200 = df["EMA20"].iloc[-1], df["EMA50"].iloc[-1], df["EMA200"].iloc[-1]
    rsi, macd, macd_signal = df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["Signal"].iloc[-1]
    close = df["Close"].iloc[-1]
    breakout = False
    signal = None

    # EMA Alignment Check
    ema_aligned = ema20 > ema50 > ema200

    # MACD Crossover
    macd_cross = macd > macd_signal

    # RSI Confirmation
    rsi_ok = 50 <= rsi <= 65

    if ema_aligned and macd_cross and rsi_ok and close > ema20:
        breakout = True
        signal = "Bullish breakout confirmed"
    elif close < ema200:
        signal = "Bearish structure"
    else:
        signal = "No strong signal yet"

    return breakout, signal


# --- Main Analysis Function ---

def analyze_stock(symbol="RELIANCE.NS", lookback_days=365):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        raise ValueError(f"No data available for {symbol}")

    df["EMA20"] = compute_ema(df["Close"], 20)
    df["EMA50"] = compute_ema(df["Close"], 50)
    df["EMA200"] = compute_ema(df["Close"], 200)
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"], df["Signal"], df["Hist"] = compute_macd(df["Close"])

    # --- Detect Support/Resistance & Breakout ---
    supports, resistances = detect_support_resistance(df)
    breakout, signal = detect_breakout(df)

    # --- Candlestick Chart Generation ---
    try:
        apds = [
            mpf.make_addplot(df["EMA20"], color="orange", width=0.8),
            mpf.make_addplot(df["EMA50"], color="blue", width=0.8),
            mpf.make_addplot(df["EMA200"], color="green", width=0.8),
        ]

        fig, _ = mpf.plot(
            df.tail(150),
            type="candle",
            style="yahoo",
            title=f"{symbol} — Swing Chart ({datetime.now():%d-%b-%Y})",
            addplot=apds,
            volume=True,
            returnfig=True,
            figsize=(10, 7)
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        chart_b64 = None

    # --- Latest Indicator Snapshot ---
    latest = df.iloc[-1]
    result = {
        "symbol": symbol,
        "date": str(latest.name.date()),
        "latest_close": round(latest["Close"], 2),
        "ema20": round(latest["EMA20"], 2),
        "ema50": round(latest["EMA50"], 2),
        "ema200": round(latest["EMA200"], 2),
        "rsi": round(latest["RSI"], 2),
        "macd": round(latest["MACD"], 2),
        "macd_signal": round(latest["Signal"], 2),
        "macd_hist": round(latest["Hist"], 2),
        "supports": supports,
        "resistances": resistances,
        "breakout_signal": signal,
        "breakout": breakout,
        "chart_base64": chart_b64
    }

    return result
