# ===============================================
#   SWING MODE ENGINE v3.1-F — UNIVERSAL STOCK ANALYZER (STABLE)
#   Adds robust Yahoo fallback, retry logic, and safe JSON returns
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
import time

# ------------------------------------------------
#   CORE TECHNICAL INDICATOR FUNCTIONS
# ------------------------------------------------

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ------------------------------------------------
#   SUPPORT & RESISTANCE DETECTION
# ------------------------------------------------

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

# ------------------------------------------------
#   BREAKOUT DETECTION
# ------------------------------------------------

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

# ------------------------------------------------
#   ROBUST YAHOO DATA FETCHER
# ------------------------------------------------

def fetch_yahoo_data(symbol, start, end):
    """Fetches Yahoo Finance data with retries and auto-adjust fallback."""
    for attempt in range(3):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1} failed for {symbol}: {e}")
        print(f"[WARN] Empty data on attempt {attempt+1} for {symbol}. Retrying...")
        time.sleep(2)
    return pd.DataFrame()

# ------------------------------------------------
#   MAIN ANALYSIS FUNCTION
# ------------------------------------------------

def analyze_stock(symbol="RELIANCE.NS", lookback_days=365):
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        # Robust data fetch with fallback
        df = fetch_yahoo_data(symbol, start, end)

        if df.empty:
            print(f"[WARN] Primary fetch failed. Extending lookback to 5 years for {symbol}.")
            start = end - timedelta(days=1825)
            df = fetch_yahoo_data(symbol, start, end)

        if df.empty:
            print(f"[ERROR] No valid Yahoo Finance data for {symbol}.")
            return {
                "symbol": symbol,
                "status": "warning",
                "message": f"No valid Yahoo Finance data returned for {symbol}.",
                "latest_close": None,
                "ema20": None,
                "ema50": None,
                "ema200": None,
                "rsi": None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None,
                "supports": [],
                "resistances": [],
                "breakout_signal": "⚠️ No valid data",
                "breakout": False,
                "trend": "Unknown",
                "chart_base64": None
            }

        # Clean dataframe
        df.columns = [str(c).title().strip() for c in df.columns]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[df["Close"] > 0]

        if len(df) < 100:
            print(f"[WARN] Only {len(df)} rows after cleaning — limited data accuracy.")

        # Compute indicators
        df["EMA20"] = compute_ema(df["Close"], min(20, max(5, len(df)//5)))
        df["EMA50"] = compute_ema(df["Close"], min(50, max(10, len(df)//3)))
        df["EMA200"] = compute_ema(df["Close"], min(200, max(20, len(df)//2)))
        df["RSI"] = compute_rsi(df["Close"], 14)
        macd_line, signal_line, hist = compute_macd(df["Close"])
        df["MACD"], df["Signal"], df["Hist"] = macd_line, signal_line, hist

        df = df.dropna()
        if df.empty or len(df) < 30:
            print(f"[WARN] Sparse data ({len(df)} rows) — returning fallback response.")
            return {
                "symbol": symbol,
                "status": "warning",
                "message": "Data too sparse after cleaning — fallback mode.",
                "latest_close": None,
                "ema20": None,
                "ema50": None,
                "ema200": None,
                "rsi": None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None,
                "supports": [],
                "resistances": [],
                "breakout_signal": "⚠️ No valid data",
                "breakout": False,
                "trend": "Unknown",
                "chart_base64": None
            }

        supports, resistances = detect_support_resistance(df)
        breakout, signal = detect_breakout(df)

        # Chart generation
        chart_b64 = None
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
            print(f"[WARN] Chart generation failed: {e}")

        def safe_float(val):
            if pd.isna(val) or np.isinf(val):
                return None
            return round(float(val), 2)

        latest = df.iloc[-1]
        if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
            trend = "Bullish"
        elif latest["EMA20"] < latest["EMA50"] < latest["EMA200"]:
            trend = "Bearish"
        else:
            trend = "Sideways"

        return {
            "symbol": symbol,
            "status": "ok",
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
            "trend": trend,
            "chart_base64": chart_b64
        }

    except Exception as e:
        return {
            "status": "warning",
            "error": str(e),
            "message": "Unexpected runtime issue — fallback mode active."
        }

# ------------------------------------------------
#   FASTAPI APP SETUP
# ------------------------------------------------

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

class StockInput(BaseModel):
    symbol: str
    lookback_days: int = 365

app = FastAPI(title="Swing Mode Engine v3.1-F")

@app.post("/analyze")
def analyze_endpoint(payload: StockInput):
    result = analyze_stock(payload.symbol, payload.lookback_days)
    return JSONResponse(content=result, status_code=200)

@app.get("/")
def root():
    return {
        "status": "✅ Swing Mode Engine v3.1-F is live",
        "usage": "POST /analyze with JSON { 'symbol': 'RELIANCE.NS', 'lookback_days': 365 }",
        "example": {
            "url": "https://<your-render-url>/analyze",
            "body": {"symbol": "TCS.NS", "lookback_days": 365}
        }
    }
