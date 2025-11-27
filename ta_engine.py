from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import base64
from io import BytesIO

app = FastAPI(title="TA Engine with Charts")

# ----------- Request Model ----------- #
class TARequest(BaseModel):
    symbol: str
    interval: str = "1d"
    lookback: int = 365


# ----------- Indicator Functions ----------- #
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


# ----------- Chart Generator ----------- #
def generate_chart(df):
    df = df.copy()
    df.set_index("Date", inplace=True)

    # Add overlays
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    # Create additional panels
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

    # Plot style
    style = mpf.make_mpf_style(base_mpf_style="yahoo")

    # Buffer for image
    buf = BytesIO()

    mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        mav=(20, 50, 200),
        addplot=[
            mpf.make_addplot(df["RSI"], panel=1, color="purple", ylabel="RSI"),
            mpf.make_addplot(df["MACD"], panel=2, color="blue"),
            mpf.make_addplot(df["MACD_signal"], panel=2, color="red"),
            mpf.make_addplot(df["MACD_hist"], type="bar", panel=2, color="dimgray"),
        ],
        figsize=(14, 10),
        tight_layout=True,
        savefig=dict(fname=buf, dpi=100, pad_inches=0.2),
    )

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


# ----------- FastAPI Endpoint ----------- #
@app.post("/analyze")
def analyze(req: TARequest):
    try:
        df = yf.download(req.symbol, period=f"{req.lookback}d", interval=req.interval)
        df = df.reset_index()
        df = df.rename(columns={
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        })
        if df.empty:
            raise HTTPException(400, "No data for symbol.")
    except Exception as e:
        raise HTTPException(500, str(e))

    # Generate the chart image
    chart_image = generate_chart(df)

    # Latest indicators
    ema20 = ema(df["Close"], 20).iloc[-1]
    ema50 = ema(df["Close"], 50).iloc[-1]
    ema200 = ema(df["Close"], 200).iloc[-1]
    rsi_val = rsi(df["Close"]).iloc[-1]
    macd_line, macd_signal, macd_hist = macd(df["Close"])

    return {
        "symbol": req.symbol,
        "latest_close": float(df["Close"].iloc[-1]),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "ema200": float(ema200),
        "rsi": float(rsi_val),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_signal.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "chart_image_base64": chart_image
    }


@app.get("/")
def home():
    return {"message": "TA Engine with chart generation is running"}
