import io
import base64
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# -----------------------------
# Request Model
# -----------------------------
class AnalyzeRequest(BaseModel):
    symbol: str
    interval: str = "1d"
    lookback: int = 365
    capital: float = 100000
    risk_percent: float = 2
    signal_mode: str = "swing"  # "swing" or "intraday"


# -----------------------------
# Chart Generator
# -----------------------------
def generate_chart(df: pd.DataFrame) -> str:
    df.columns = [c.capitalize() for c in df.columns]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    buf = io.BytesIO()
    mpf.plot(
        df,
        type="candle",
        mav=(20, 50, 200),
        volume=True,
        style="charles",
        savefig=dict(fname=buf, dpi=100, pad_inches=0.2),
    )
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img


# -----------------------------
# Support / Resistance Detection
# -----------------------------
def detect_support_resistance(df, window=10):
    df["Min"] = df["Low"].rolling(window=window, center=True).min()
    df["Max"] = df["High"].rolling(window=window, center=True).max()
    return (
        sorted(list(set(df["Min"].dropna().round(2).tail(5)))),
        sorted(list(set(df["Max"].dropna().round(2).tail(5)))),
    )


# -----------------------------
# Trend Classifier
# -----------------------------
def classify_trend(ema20, ema50, ema200, rsi, macd_hist):
    if ema20 > ema50 > ema200 and rsi > 55 and macd_hist > 0:
        return "Bullish Breakout"
    elif ema20 > ema50 > ema200 and 45 <= rsi <= 55:
        return "Sideways / Pullback"
    elif ema20 < ema50 < ema200 and rsi < 45 and macd_hist < 0:
        return "Bearish Trend"
    else:
        return "Weak / Indecisive"


# -----------------------------
# Trade Levels with Mode Sensitivity
# -----------------------------
def calculate_trade_levels(latest_close, support_levels, resistance_levels, trend_signal, mode="swing"):
    if not support_levels or not resistance_levels:
        return (None,) * 7

    last_support = support_levels[-1]
    last_resistance = resistance_levels[-1]

    # Mode-based volatility multiplier
    if mode == "intraday":
        stop_mult, tgt1_mult, tgt2_mult = 0.005, 0.008, 0.015
    else:  # swing
        stop_mult, tgt1_mult, tgt2_mult = 0.02, 0.03, 0.06

    if "Bullish" in trend_signal:
        breakout_entry = last_resistance * 1.01
        retest_entry = last_resistance * 0.99
        stop_loss = breakout_entry * (1 - stop_mult)
        target1 = breakout_entry * (1 + tgt1_mult)
        target2 = breakout_entry * (1 + tgt2_mult)
    elif "Bearish" in trend_signal:
        breakout_entry = last_support * 0.99
        retest_entry = last_support * 1.01
        stop_loss = breakout_entry * (1 + stop_mult)
        target1 = breakout_entry * (1 - tgt1_mult)
        target2 = breakout_entry * (1 - tgt2_mult)
    else:
        breakout_entry = latest_close
        retest_entry = latest_close * 0.99
        stop_loss = latest_close * (1 - stop_mult)
        target1 = latest_close * (1 + tgt1_mult)
        target2 = latest_close * (1 + tgt2_mult)

    # R:R Calculation
    risk = abs(latest_close - stop_loss)
    reward = abs(target1 - latest_close)
    rr_ratio = round(reward / risk, 2) if risk else None
    risk_level = (
        "Low Risk (R:R ≥ 2)"
        if rr_ratio and rr_ratio >= 2
        else "Medium Risk"
        if rr_ratio and rr_ratio >= 1.2
        else "High Risk"
    )

    return (
        round(breakout_entry, 2),
        round(retest_entry, 2),
        round(stop_loss, 2),
        round(target1, 2),
        round(target2, 2),
        rr_ratio,
        risk_level,
    )


# -----------------------------
# Position Size Engine
# -----------------------------
def calculate_position_size(capital, risk_percent, entry_price, stop_loss, target1, target2):
    max_risk_amount = capital * (risk_percent / 100)
    per_share_risk = abs(entry_price - stop_loss)
    if per_share_risk == 0:
        return (None,) * 5

    qty = int(max_risk_amount // per_share_risk)
    alloc = round(qty * entry_price, 2)
    max_loss = round(qty * per_share_risk, 2)
    profit_t1 = round(qty * abs(target1 - entry_price), 2)
    profit_t2 = round(qty * abs(target2 - entry_price), 2)
    return qty, alloc, max_loss, profit_t1, profit_t2


# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    print(f"Analyzing {req.symbol} [{req.signal_mode.upper()} Mode]")

    df = yf.download(req.symbol, period=f"{req.lookback}d", interval=req.interval, auto_adjust=False)
    if df.empty:
        raise ValueError("No data retrieved")

    df = df.dropna().astype(float)
    try:
        chart = generate_chart(df)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        chart = None

    # Indicators
    latest_close = df["Close"].iloc[-1]
    ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["Close"].ewm(span=200).mean().iloc[-1]
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rsi = 100 - (100 / (1 + pd.Series(gain).rolling(14).mean() / pd.Series(loss).rolling(14).mean()))
    macd_line = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line

    support, resistance = detect_support_resistance(df)
    trend = classify_trend(ema20, ema50, ema200, rsi.iloc[-1], macd_hist.iloc[-1])
    (
        breakout,
        retest,
        sl,
        t1,
        t2,
        rr,
        risk_lvl,
    ) = calculate_trade_levels(latest_close, support, resistance, trend, req.signal_mode)

    qty, alloc, max_loss, p1, p2 = calculate_position_size(req.capital, req.risk_percent, breakout, sl, t1, t2)

    return {
        "symbol": req.symbol,
        "mode": req.signal_mode,
        "latest_close": round(latest_close, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "ema200": round(ema200, 2),
        "rsi": round(rsi.iloc[-1], 2),
        "macd": round(macd_line.iloc[-1], 3),
        "macd_signal": round(signal_line.iloc[-1], 3),
        "macd_hist": round(macd_hist.iloc[-1], 3),
        "trend_signal": trend,
        "support_levels": support[-3:],
        "resistance_levels": resistance[-3:],
        "breakout_entry": breakout,
        "retest_entry": retest,
        "stop_loss": sl,
        "target1": t1,
        "target2": t2,
        "rr_ratio": rr,
        "risk_level": risk_lvl,
        "recommended_qty": qty,
        "capital_allocated": alloc,
        "max_loss_if_sl_hit": max_loss,
        "profit_target1": p1,
        "profit_target2": p2,
        "chart_image_base64": chart,
    }
