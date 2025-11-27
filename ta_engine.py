# ===============================================
#   SWING MODE ENGINE v3.1 — UNIVERSAL STOCK ANALYZER (STABLE)
#   Auto-recovers missing data, dynamic lookback, safe indicators
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

# ------------------------------------------------
#   CORE TECHNICAL INDICATOR FUNCTIONS
# ------------------------------------------------

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
#   MAIN ANALYSIS FUNCTION
# ------------------------------------------------

def analyze_stock(symbol="RELIANCE.NS", lookback_days=365):
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)

        df = yf.download(symbol, start=start, end=end, auto_adjust=False)
        if df.empty:
            raise ValueError("No data received from Yahoo Finance.")

        # Auto-extend lookback if insufficient data
        if len(df) < 250:
            extra_days = 365 if lookback_days < 730 else 0
            new_start = start - timedelta(days=extra_days)
            print(f"[INFO] Extending lookback automatically → {extra_days} days")
            df = yf.download(symbol, start=new_start, end=end, auto_adjust=False)

        # Flatten multi-index if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = [str(c).title().strip() for c in df.columns]

        # Fallback for Adj Close
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required_cols = ["Open", "High", "Low", "Close"]
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=required_cols)
        df = df[df["Close"] > 0]

        if len(df) < 100:
            raise ValueError("Insufficient clean OHLC data even after fallback fetch.")

        # Compute indicators safely
        df["EMA20"] = compute_ema(df["Close"], min(20, max(5, len(df)//5)))
        df["EMA50"] = compute_ema(df["Close"], min(50, max(10, len(df)//3)))
        df["EMA200"] = compute_ema(df["Close"], min(200, max(20, len(df)//2)))
        df["RSI"] = compute_rsi(df["Close"], 14)
        macd_line, signal_line, hist = compute_macd(df["Close"])
        df["MACD"], df["Signal"], df["Hist"] = macd_line, signal_line, hist

        df = df.dropna()
        if df.empty or len(df) < 30:
            raise ValueError(f"Data too sparse after cleaning. Received only {len(df)} valid rows.")

        # Compute supports, resistances, breakout
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
            print(f"Chart generation failed: {e}")

        # Safe float conversion
        def safe_float(val):
            if pd.isna(val) or np.isinf(val):
                return None
            return round(float(val), 2)

        latest = df.iloc[-1]

        # Trend classification
        if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
            trend = "Bullish"
        elif latest["EMA20"] < latest["EMA50"] < latest["EMA200"]:
            trend = "Bearish"
        else:
            trend = "Sideways"

        # Final JSON result
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
            "trend": trend,
            "chart_base64": chart_b64,
        }

        return result

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "suggestion": "Try increasing lookback_days to 730 or 1095, or ensure symbol is valid (e.g., RELIANCE.NS)."
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

app = FastAPI(title="Swing Mode Engine v3.1")

@app.post("/analyze")
def analyze_endpoint(payload: StockInput):
    result = analyze_stock(payload.symbol, payload.lookback_days)
    status = 200 if "error" not in result else 400
    return JSONResponse(content=result, status_code=status)

@app.get("/")
def root():
    return {
        "status": "✅ Swing Mode Engine v3.1 is live",
        "usage": "POST /analyze with JSON { 'symbol': 'RELIANCE.NS', 'lookback_days': 365 }",
        "example": {
            "url": "https://ta-engine-v927.onrender.com/analyze",
            "body": {"symbol": "TCS.NS", "lookback_days": 365}
        }
    }
