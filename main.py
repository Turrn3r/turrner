import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_ENV = os.getenv("APP_ENV", "prod")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")  # e.g. https://yourdomain.com
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "")
TWELVEDATA_BASE = "https://api.twelvedata.com"

app = FastAPI()

# If you use Option A (Fly direct), you MUST enable CORS for your Plesk domain.
# If you use Option B (Plesk proxy), you can set ALLOWED_ORIGIN="" and skip CORS.
if ALLOWED_ORIGIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ALLOWED_ORIGIN],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


# ---------- Models ----------
class TradeRequest(BaseModel):
    symbol: str
    side: str  # "BUY" or "SELL"
    collateral: float
    leverage: float
    tp: float | None = None
    sl: float | None = None


# ---------- Helpers ----------
def map_symbol_to_twelvedata(symbol: str) -> str:
    # Twelve Data uses symbols like XAU/USD, BTC/USD, etc.
    if symbol.upper() == "XAUUSD":
        return "XAU/USD"
    if symbol.upper() == "BTCUSD":
        return "BTC/USD"
    return symbol

def map_tf(tf: str) -> tuple[str, str]:
    # returns (interval, outputsize-ish handling)
    # Twelve Data intervals are like "1min", "5min", "1h", "4h", "1day" depending on plan/endpoints.
    tf = tf.lower()
    if tf == "1m": return ("1min", "minutes")
    if tf == "5m": return ("5min", "minutes")
    if tf == "1h": return ("1h", "hours")
    if tf == "4h": return ("4h", "hours")
    if tf == "1d": return ("1day", "days")
    return ("1h", "hours")


# ---------- Routes ----------
@app.get("/api/health")
def health():
    # Replace with real network + wallet when you wire Ostium SDK
    return {"ok": True, "network": "arbitrum", "address": "0x0000000000000000000000000000000000000000"}

@app.get("/api/balance")
def balance():
    # Replace with Ostium SDK balance call
    return {"usdc": 0}

@app.get("/api/orders")
def orders():
    # Replace with Ostium SDK open orders call
    return {"orders": []}

@app.get("/api/positions")
def positions():
    # Replace with Ostium SDK open positions call
    return {"positions": []}

@app.post("/api/trade")
def trade(req: TradeRequest):
    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")

    # ---- WHERE YOU PLUG OSTIUM SDK IN ----
    # 1) Validate inputs
    if req.collateral <= 0 or req.leverage <= 0:
        raise HTTPException(status_code=400, detail="collateral and leverage must be > 0")

    # 2) Place trade using Ostium SDK (server-side)
    # tx_hash = ostium.place_trade(...)
    # For now, return a fake tx so your UI works:
    fake_tx = hex(int(time.time()))  # just a placeholder
    return {"ok": True, "tx_hash": fake_tx}

@app.get("/api/candles")
def candles(symbol: str, tf: str = "1h", limit: int = 500):
    if not TWELVEDATA_KEY:
        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_KEY on server")

    interval, _ = map_tf(tf)
    td_symbol = map_symbol_to_twelvedata(symbol)

    # Twelve Data time_series endpoint (common pattern)
    params = {
        "symbol": td_symbol,
        "interval": interval,
        "outputsize": min(max(limit, 10), 5000),
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }
    r = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=20)
    data = r.json()

    if r.status_code != 200 or "values" not in data:
        raise HTTPException(status_code=502, detail=f"TwelveData error: {data}")

    # Twelve Data usually returns newest-first; your chart expects oldest->newest.
    values = list(reversed(data["values"]))

    out = []
    for v in values:
        # datetime format varies; easiest: parse with dateutil in real code.
        # Here we assume ISO-like and let Python parse in a minimal way:
        # Better production parsing is recommended.
        dt = v.get("datetime")
        if not dt:
            continue

        # naive conversion: use requests to parse is too much here; keep it simple:
        # Expect "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
        try:
            # If you want robust parsing: pip install python-dateutil and parse
            import datetime as _dt
            if " " in dt:
                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
            else:
                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d").timestamp() * 1000)
        except Exception:
            continue

        out.append({
            "time": ts,
            "open": v.get("open"),
            "high": v.get("high"),
            "low": v.get("low"),
            "close": v.get("close"),
        })

    return {"candles": out}
