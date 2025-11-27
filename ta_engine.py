# ===============================================
#   SWING MODE ENGINE v2.2 — UNIVERSAL STOCK ANALYZER
#   Supports any NSE/BSE stock (e.g., RELIANCE.NS, TCS.NS, INFY.NS)
#   Computes EMA, RSI, MACD, Support/Resistance, Trend & Breakout Signals
#   Generates professional candlestick chart
# ===============================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    """Safe RSI calculation (handles NaN gracefully)."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Replace NaN with neutral 50.0
    rsi = rsi.fillna(50.0)
    return rsi

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
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    ema200 = float(df["EMA200"].iloc[-1])
    rsi = float(df["RSI"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    macd_signal = float(df["Signal"].iloc[-1])
    close = float(df["Close"].iloc[-1])

    ema_aligned = (ema20 > ema50) and (ema50 > ema200)
    macd_cross = macd > macd_signal
    rsi_ok = 50 <= rsi <= 65

    if ema_aligned and macd_cross and rsi_ok and close > ema20:
        return True, "✅ Bullish breakout confirmed"
    elif close < ema200:
        return False, "⚠️ Bearish structure"
    else:
        return False, "⏸ No strong signal yet"

# --- Main Analysis Function ---

def analyze_stock(symbol="RELIANCE.NS", lookback_days=365):
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    df = yf.download(symbol, start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data available for {symbol}")

    # ✅ Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [str(c).title().strip() for c in df.columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from Yahoo data: {missing}")

    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_cols)

    # --- Compute Indicators ---
    df["EMA20"] = compute_ema(df["Close"], 20)
    df["EMA50"] = compute_ema(df["Close"], 50)
    df["EMA200"] = compute_ema(df["Close"], 200)
    df["RSI"] = compute_rsi(df["Close"], 14)
    macd_line, signal_line, hist = compute_macd(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = macd_line, signal_line, hist

    # ✅ Drop all rows with any NaN after indicators
    df = df.dropna()

    # --- Detect Support/Resistance & Breakout ---
    supports, resistances = detect_support_resistance(df)
    breakout, signal = detect_breakout(df)

    # --- Chart Generation (safe) ---
    try:
        df_plot = df.tail(150).copy()
        apds = [
            mpf.make_addplot(df_plot["EMA20"], color="orange", width=0.8),
            mpf.make_addplot(df_plot["EMA50"], color="blue", width=0.8),
            mpf.make_addplot(df_plot["EMA200"], color="green", width=0.8),
        ]
        fig, _ = mpf.plot(
            df_plot,
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

    # --- Final Snapshot ---
    latest = df.iloc[-1]

    # ✅ Convert NaN → None for JSON safety
    def safe_float(val):
        if pd.isna(val) or np.isinf(val):
            return None
        return round(float(val), 2)

    # --- Trend Classification ---
    if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
        trend = "Bullish"
    elif latest["EMA20"] < latest["EMA50"] < latest["EMA200"]:
        trend = "Bearish"
    else:
        trend = "Sideways"

    # --- Final JSON Result ---
    result = {
        "symbol": symbol,
        "date": str(latest.name.date()),
        "latest_close": safe_float(latest["Close"]),
        "ema20": safe_float(latest["EMA20"]),
        "ema50": safe_float(latest["EMA50"]),
        "ema200": safe_float(latest["EMA200"]),
        "rsi": safe_float(latest["RSI"]),
        "macd": safe_float(latest["MACD"]),
        "macd_signal": safe_float(latest["Signal"]),
        "macd_hist": safe_float(latest["Hist"]),
        "supports": [safe_float(x) for x in supports],
        "resistances": [safe_float(x) for x in resistances],
        "breakout_signal": signal,
        "breakout": bool(breakout),
        "trend": trend,  # 👈 Added Trend Classifier
        "chart_base64": chart_b64,
    }

    return result

# =============================
#   FASTAPI APP SETUP
# =============================

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class StockInput(BaseModel):
    symbol: str

app = FastAPI(title="Swing Mode Engine v2.2")

@app.post("/analyze")
def analyze_endpoint(payload: StockInput):
    try:
        result = analyze_stock(payload.symbol)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {
        "status": "✅ Swing Mode Engine v2.2 is live",
        "usage": "POST /analyze with JSON { 'symbol': 'TCS.NS' }",
        "example": {
            "url": "https://ta-engine-v927.onrender.com/analyze",
            "body": {"symbol": "RELIANCE.NS"}
        }
    }
