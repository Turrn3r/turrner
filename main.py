diff --git a/main.py b/main.py
index 4e78b5c44da91cd711fe16005bf5baf17c5650cf..e4dd128776c993cdf21b981d97dd22b4c63f4454 100644
--- a/main.py
+++ b/main.py
@@ -1,150 +1,563 @@
-import os
-import time
-import requests
-from fastapi import FastAPI, HTTPException
-from fastapi.middleware.cors import CORSMiddleware
-from pydantic import BaseModel
-
-APP_ENV = os.getenv("APP_ENV", "prod")
-ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")  # e.g. https://yourdomain.com
-TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "")
-TWELVEDATA_BASE = "https://api.twelvedata.com"
-
-app = FastAPI()
-
-# If you use Option A (Fly direct), you MUST enable CORS for your Plesk domain.
-# If you use Option B (Plesk proxy), you can set ALLOWED_ORIGIN="" and skip CORS.
-if ALLOWED_ORIGIN:
-    app.add_middleware(
-        CORSMiddleware,
-        allow_origins=[ALLOWED_ORIGIN],
-        allow_credentials=False,
-        allow_methods=["GET", "POST", "OPTIONS"],
-        allow_headers=["*"],
-    )
-
-
-# ---------- Models ----------
-class TradeRequest(BaseModel):
-    symbol: str
-    side: str  # "BUY" or "SELL"
-    collateral: float
-    leverage: float
-    tp: float | None = None
-    sl: float | None = None
-
-
-# ---------- Helpers ----------
-def map_symbol_to_twelvedata(symbol: str) -> str:
-    # Twelve Data uses symbols like XAU/USD, BTC/USD, etc.
-    if symbol.upper() == "XAUUSD":
-        return "XAU/USD"
-    if symbol.upper() == "BTCUSD":
-        return "BTC/USD"
-    return symbol
-
-def map_tf(tf: str) -> tuple[str, str]:
-    # returns (interval, outputsize-ish handling)
-    # Twelve Data intervals are like "1min", "5min", "1h", "4h", "1day" depending on plan/endpoints.
-    tf = tf.lower()
-    if tf == "1m": return ("1min", "minutes")
-    if tf == "5m": return ("5min", "minutes")
-    if tf == "1h": return ("1h", "hours")
-    if tf == "4h": return ("4h", "hours")
-    if tf == "1d": return ("1day", "days")
-    return ("1h", "hours")
-
-
-# ---------- Routes ----------
-@app.get("/api/health")
-def health():
-    # Replace with real network + wallet when you wire Ostium SDK
-    return {"ok": True, "network": "arbitrum", "address": "0x0000000000000000000000000000000000000000"}
-
-@app.get("/api/balance")
-def balance():
-    # Replace with Ostium SDK balance call
-    return {"usdc": 0}
-
-@app.get("/api/orders")
-def orders():
-    # Replace with Ostium SDK open orders call
-    return {"orders": []}
-
-@app.get("/api/positions")
-def positions():
-    # Replace with Ostium SDK open positions call
-    return {"positions": []}
-
-@app.post("/api/trade")
-def trade(req: TradeRequest):
-    side = req.side.upper()
-    if side not in ("BUY", "SELL"):
-        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
-
-    # ---- WHERE YOU PLUG OSTIUM SDK IN ----
-    # 1) Validate inputs
-    if req.collateral <= 0 or req.leverage <= 0:
-        raise HTTPException(status_code=400, detail="collateral and leverage must be > 0")
-
-    # 2) Place trade using Ostium SDK (server-side)
-    # tx_hash = ostium.place_trade(...)
-    # For now, return a fake tx so your UI works:
-    fake_tx = hex(int(time.time()))  # just a placeholder
-    return {"ok": True, "tx_hash": fake_tx}
-
-@app.get("/api/candles")
-def candles(symbol: str, tf: str = "1h", limit: int = 500):
-    if not TWELVEDATA_KEY:
-        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_KEY on server")
-
-    interval, _ = map_tf(tf)
-    td_symbol = map_symbol_to_twelvedata(symbol)
-
-    # Twelve Data time_series endpoint (common pattern)
-    params = {
-        "symbol": td_symbol,
-        "interval": interval,
-        "outputsize": min(max(limit, 10), 5000),
-        "apikey": TWELVEDATA_KEY,
-        "format": "JSON",
-    }
-    r = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=20)
-    data = r.json()
-
-    if r.status_code != 200 or "values" not in data:
-        raise HTTPException(status_code=502, detail=f"TwelveData error: {data}")
-
-    # Twelve Data usually returns newest-first; your chart expects oldest->newest.
-    values = list(reversed(data["values"]))
-
-    out = []
-    for v in values:
-        # datetime format varies; easiest: parse with dateutil in real code.
-        # Here we assume ISO-like and let Python parse in a minimal way:
-        # Better production parsing is recommended.
-        dt = v.get("datetime")
-        if not dt:
-            continue
-
-        # naive conversion: use requests to parse is too much here; keep it simple:
-        # Expect "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
-        try:
-            # If you want robust parsing: pip install python-dateutil and parse
-            import datetime as _dt
-            if " " in dt:
-                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
-            else:
-                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d").timestamp() * 1000)
-        except Exception:
-            continue
-
-        out.append({
-            "time": ts,
-            "open": v.get("open"),
-            "high": v.get("high"),
-            "low": v.get("low"),
-            "close": v.get("close"),
-        })
-
-    return {"candles": out}
+import os
+import time
+import math
+import hashlib
+from pathlib import Path
+from datetime import datetime, timedelta, timezone
+from email.utils import parsedate_to_datetime
+from xml.etree import ElementTree as ET
+
+import requests
+from fastapi import FastAPI, HTTPException
+from fastapi.middleware.cors import CORSMiddleware
+from fastapi.responses import FileResponse
+from pydantic import BaseModel
+
+APP_ENV = os.getenv("APP_ENV", "prod")
+ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")  # backward-compatible single origin
+ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")  # comma-separated origins
+TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY", "")
+TWELVEDATA_BASE = "https://api.twelvedata.com"
+REQUEST_TIMEOUT = 20
+BASE_DIR = Path(__file__).resolve().parent
+INDEX_HTML = BASE_DIR / "index.html"
+
+GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
+INTEL_CACHE_TTL_SECONDS = int(os.getenv("INTEL_CACHE_TTL_SECONDS", "90"))
+
+
+PAIR_INTEL_CONFIG = {
+    "XRP": {
+        "label": "XRP",
+        "queries": [
+            '(xrp OR ripple OR "ripple labs") (sec OR regulator OR regulation OR lawsuit OR court OR ruling OR settlement)',
+            '(xrp OR ripple) (etf OR listing OR delisting OR custody OR derivatives OR open interest)',
+            '(xrp OR ripple OR crypto) (hack OR exploit OR breach OR liquidation OR outage OR sanctions)',
+        ],
+        "direction_up": ["settlement", "approved", "approval", "dismissed", "listing", "easing"],
+        "direction_down": ["appeal", "enforcement", "ban", "delisting", "hack", "lawsuit"],
+    },
+    "XAU": {
+        "label": "Gold (XAU)",
+        "queries": [
+            '(gold OR xau OR bullion OR "spot gold") (fed OR rates OR inflation OR cpi OR yields OR dollar OR dxy)',
+            '(gold OR bullion OR "safe haven") (war OR conflict OR sanctions OR missile OR ceasefire)',
+            '(gold OR bullion) ("central bank" OR reserves OR "gold buying")',
+        ],
+        "direction_up": ["rate cut", "ceasefire", "weaker dollar", "safe haven", "reserve buying"],
+        "direction_down": ["rate hike", "strong dollar", "higher yields", "risk-on"],
+    },
+    "SILVER": {
+        "label": "Silver",
+        "queries": [
+            '(silver OR xag OR "spot silver") (fed OR rates OR inflation OR yields OR dollar OR recession)',
+            '(silver OR xag) (solar OR photovoltaic OR industrial OR manufacturing OR shortage OR supply OR mine)',
+            '(silver OR xag) (tariff OR sanctions OR export ban OR disruption OR strike)',
+        ],
+        "direction_up": ["solar demand", "industrial demand", "shortage", "supply disruption"],
+        "direction_down": ["slowdown", "demand slump", "rate hike", "strong dollar"],
+    },
+    "OIL": {
+        "label": "Crude Oil",
+        "queries": [
+            '(oil OR brent OR wti OR "crude oil") (opec OR opec+ OR quota OR "production cut" OR "output cut")',
+            '(oil OR crude OR brent OR wti) (red sea OR "strait of hormuz" OR shipping OR attack OR drone OR blockade)',
+            '(oil OR crude OR brent OR wti) (sanctions OR embargo OR export ban OR price cap OR inventory OR eia OR api)',
+        ],
+        "direction_up": ["production cut", "outage", "disruption", "inventory draw", "sanctions"],
+        "direction_down": ["production rise", "inventory build", "demand slump", "ceasefire"],
+    },
+}
+
+
+GOV_RSS_FEEDS = [
+    "https://www.federalreserve.gov/feeds/press_all.xml",
+    "https://www.ecb.europa.eu/rss/press.html",
+    "https://www.imf.org/en/News/rss",
+    "https://www.worldbank.org/en/news/all?output=rss",
+    "https://www.opec.org/opec_web/en/press_room/rss.xml",
+    "https://www.iaea.org/newscenter/rss.xml",
+]
+
+
+TRUSTED_SOURCE_BOOST = {
+    "reuters.com": 16,
+    "bloomberg.com": 16,
+    "ft.com": 12,
+    "wsj.com": 12,
+    "federalreserve.gov": 18,
+    "ecb.europa.eu": 18,
+    "imf.org": 17,
+    "worldbank.org": 16,
+    "opec.org": 18,
+    "iaea.org": 17,
+}
+
+_intel_cache: dict[str, tuple[float, dict]] = {}
+
+app = FastAPI()
+
+origins = []
+if ALLOWED_ORIGIN:
+    origins.append(ALLOWED_ORIGIN.strip())
+if ALLOWED_ORIGINS:
+    origins.extend([origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()])
+
+if origins:
+    app.add_middleware(
+        CORSMiddleware,
+        allow_origins=sorted(set(origins)),
+        allow_credentials=False,
+        allow_methods=["GET", "POST", "OPTIONS"],
+        allow_headers=["*"],
+    )
+
+
+# ---------- Models ----------
+class TradeRequest(BaseModel):
+    symbol: str
+    side: str  # "BUY" or "SELL"
+    collateral: float
+    leverage: float
+    tp: float | None = None
+    sl: float | None = None
+
+
+class IntelEvent(BaseModel):
+    title: str
+    source: str
+    url: str
+    published_at: str
+    pair: str
+    impact_score: int
+    impact_direction: str
+    confidence: int
+    tags: list[str]
+
+
+class IntelSummary(BaseModel):
+    pair: str
+    direction: str
+    confidence: int
+    total_events: int
+
+
+class IntelResponse(BaseModel):
+    summary: IntelSummary
+    events: list[IntelEvent]
+
+
+# ---------- Helpers ----------
+def map_symbol_to_twelvedata(symbol: str) -> str:
+    # Twelve Data uses symbols like XAU/USD, BTC/USD, etc.
+    if symbol.upper() == "XAUUSD":
+        return "XAU/USD"
+    if symbol.upper() == "BTCUSD":
+        return "BTC/USD"
+    return symbol
+
+def map_tf(tf: str) -> tuple[str, str]:
+    # returns (interval, outputsize-ish handling)
+    # Twelve Data intervals are like "1min", "5min", "1h", "4h", "1day" depending on plan/endpoints.
+    tf = tf.lower()
+    if tf == "1m": return ("1min", "minutes")
+    if tf == "5m": return ("5min", "minutes")
+    if tf == "1h": return ("1h", "hours")
+    if tf == "4h": return ("4h", "hours")
+    if tf == "1d": return ("1day", "days")
+    return ("1h", "hours")
+
+
+def now_utc() -> datetime:
+    return datetime.now(timezone.utc)
+
+
+def parse_datetime(value: str | None) -> datetime | None:
+    if not value:
+        return None
+    text = value.strip()
+    if not text:
+        return None
+
+    candidates = [
+        "%Y-%m-%dT%H:%M:%S%z",
+        "%Y-%m-%dT%H:%M:%SZ",
+        "%Y-%m-%d %H:%M:%S",
+        "%Y-%m-%d",
+    ]
+    for fmt in candidates:
+        try:
+            dt = datetime.strptime(text, fmt)
+            if dt.tzinfo is None:
+                dt = dt.replace(tzinfo=timezone.utc)
+            return dt.astimezone(timezone.utc)
+        except ValueError:
+            pass
+
+    try:
+        dt = parsedate_to_datetime(text)
+        if dt.tzinfo is None:
+            dt = dt.replace(tzinfo=timezone.utc)
+        return dt.astimezone(timezone.utc)
+    except Exception:
+        return None
+
+
+def recency_score(published_at: datetime | None) -> int:
+    if not published_at:
+        return 0
+    age_hours = max(0.0, (now_utc() - published_at).total_seconds() / 3600)
+    return max(0, int(round(18 * math.exp(-age_hours / 8))))
+
+
+def direction_signal(pair: str, text: str) -> int:
+    cfg = PAIR_INTEL_CONFIG.get(pair, {})
+    lowered = text.lower()
+    score = 0
+    for token in cfg.get("direction_up", []):
+        if token in lowered:
+            score += 1
+    for token in cfg.get("direction_down", []):
+        if token in lowered:
+            score -= 1
+    return score
+
+
+def normalize_domain(url: str) -> str:
+    try:
+        from urllib.parse import urlparse
+
+        return urlparse(url).netloc.replace("www.", "").lower()
+    except Exception:
+        return ""
+
+
+def classify_tags(title: str) -> list[str]:
+    t = title.lower()
+    tags = []
+    taxonomy = {
+        "government": ["government", "ministry", "regulator", "sec", "court", "sanction", "policy"],
+        "macro": ["inflation", "cpi", "rates", "yield", "dollar", "fed", "ecb"],
+        "geopolitics": ["war", "conflict", "sanctions", "attack", "missile", "ceasefire"],
+        "supply": ["opec", "production", "output", "inventory", "shortage", "mine"],
+        "risk": ["hack", "exploit", "breach", "outage", "default", "liquidation"],
+    }
+    for tag, words in taxonomy.items():
+        if any(w in t for w in words):
+            tags.append(tag)
+    return tags or ["macro"]
+
+
+def score_event(pair: str, title: str, source: str, published_at: datetime | None) -> tuple[int, str, int, list[str]]:
+    source_domain = normalize_domain(source)
+    source_boost = TRUSTED_SOURCE_BOOST.get(source_domain, 4)
+    tags = classify_tags(title)
+    tag_boost = 6 * len(tags)
+    recency = recency_score(published_at)
+    dir_raw = direction_signal(pair, title)
+    direction = "neutral"
+    if dir_raw >= 2:
+        direction = "buy"
+    elif dir_raw <= -2:
+        direction = "sell"
+
+    score = max(0, min(100, int(18 + source_boost + tag_boost + recency + abs(dir_raw) * 6)))
+    confidence = max(20, min(99, int(30 + source_boost + recency + abs(dir_raw) * 5)))
+    return score, direction, confidence, tags
+
+
+def fetch_gdelt_events(pair: str, hours: int, per_query: int) -> list[dict]:
+    cfg = PAIR_INTEL_CONFIG[pair]
+    end_time = now_utc()
+    start_time = end_time - timedelta(hours=hours)
+    items: list[dict] = []
+
+    for query in cfg["queries"]:
+        params = {
+            "query": query,
+            "mode": "ArtList",
+            "format": "json",
+            "sort": "DateDesc",
+            "maxrecords": str(per_query),
+            "startdatetime": start_time.strftime("%Y%m%d%H%M%S"),
+            "enddatetime": end_time.strftime("%Y%m%d%H%M%S"),
+        }
+        response = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT)
+        if response.status_code != 200:
+            continue
+        payload = response.json()
+        for article in payload.get("articles", []):
+            title = article.get("title") or article.get("name") or ""
+            url = article.get("url") or ""
+            seen = article.get("seendate") or article.get("date")
+            items.append(
+                {
+                    "title": title,
+                    "url": url,
+                    "source": url,
+                    "published_at": parse_datetime(seen),
+                    "pair": pair,
+                }
+            )
+    return items
+
+
+def fetch_rss_events(pair: str, limit_per_feed: int) -> list[dict]:
+    query_terms = [pair.lower(), PAIR_INTEL_CONFIG[pair]["label"].lower(), "crypto", "gold", "silver", "oil"]
+    results = []
+    for feed in GOV_RSS_FEEDS:
+        try:
+            response = requests.get(feed, timeout=REQUEST_TIMEOUT)
+            if response.status_code != 200:
+                continue
+            root = ET.fromstring(response.content)
+            entries = root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry")
+            count = 0
+            for entry in entries:
+                if count >= limit_per_feed:
+                    break
+                title = (entry.findtext("title") or entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
+                link = (entry.findtext("link") or entry.findtext("{http://www.w3.org/2005/Atom}link") or "").strip()
+                if not link:
+                    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
+                    if link_el is not None:
+                        link = link_el.attrib.get("href", "")
+                pub_raw = (
+                    entry.findtext("pubDate")
+                    or entry.findtext("published")
+                    or entry.findtext("updated")
+                    or entry.findtext("{http://www.w3.org/2005/Atom}updated")
+                )
+                combined = f"{title} {link}".lower()
+                if not any(term in combined for term in query_terms):
+                    continue
+                results.append(
+                    {
+                        "title": title,
+                        "url": link,
+                        "source": feed,
+                        "published_at": parse_datetime(pub_raw),
+                        "pair": pair,
+                    }
+                )
+                count += 1
+        except Exception:
+            continue
+    return results
+
+
+def dedupe_events(events: list[dict]) -> list[dict]:
+    uniq = {}
+    for e in events:
+        fingerprint = hashlib.sha1(f"{e.get('title','')}|{e.get('url','')}".encode("utf-8")).hexdigest()
+        if fingerprint not in uniq:
+            uniq[fingerprint] = e
+    return list(uniq.values())
+
+
+def compute_summary(pair: str, events: list[IntelEvent]) -> IntelSummary:
+    if not events:
+        return IntelSummary(pair=pair, direction="neutral", confidence=0, total_events=0)
+
+    net = 0
+    conf_sum = 0
+    for event in events:
+        weight = max(1, event.impact_score // 20)
+        if event.impact_direction == "buy":
+            net += weight
+        elif event.impact_direction == "sell":
+            net -= weight
+        conf_sum += event.confidence
+
+    direction = "neutral"
+    if net >= 3:
+        direction = "bullish"
+    elif net <= -3:
+        direction = "bearish"
+
+    confidence = int(round(conf_sum / len(events)))
+    return IntelSummary(pair=pair, direction=direction, confidence=confidence, total_events=len(events))
+
+
+def load_live_intel(pair: str, hours: int, limit: int) -> IntelResponse:
+    cache_key = f"{pair}:{hours}:{limit}"
+    now_ts = time.time()
+    cached = _intel_cache.get(cache_key)
+    if cached and (now_ts - cached[0] < INTEL_CACHE_TTL_SECONDS):
+        return IntelResponse(**cached[1])
+
+    gdelt_events = fetch_gdelt_events(pair=pair, hours=hours, per_query=max(10, limit // 2))
+    rss_events = fetch_rss_events(pair=pair, limit_per_feed=max(2, limit // 10))
+
+    merged = dedupe_events(gdelt_events + rss_events)
+    output_events: list[IntelEvent] = []
+    for raw in merged:
+        title = raw.get("title") or "Untitled"
+        source = raw.get("source") or raw.get("url") or "unknown"
+        published = raw.get("published_at")
+        impact_score, impact_direction, confidence, tags = score_event(pair, title, source, published)
+        output_events.append(
+            IntelEvent(
+                title=title,
+                source=normalize_domain(source) or source,
+                url=raw.get("url") or "",
+                published_at=(published or now_utc()).isoformat(),
+                pair=pair,
+                impact_score=impact_score,
+                impact_direction=impact_direction,
+                confidence=confidence,
+                tags=tags,
+            )
+        )
+
+    output_events.sort(key=lambda e: (e.impact_score, e.confidence, e.published_at), reverse=True)
+    limited = output_events[:limit]
+    summary = compute_summary(pair, limited)
+    response = IntelResponse(summary=summary, events=limited)
+    _intel_cache[cache_key] = (now_ts, response.dict())
+    return response
+
+
+# ---------- Routes ----------
+@app.get("/")
+def home():
+    if INDEX_HTML.exists():
+        return FileResponse(INDEX_HTML)
+    return {
+        "ok": True,
+        "message": "Turrner API is running. index.html not found in container image.",
+    }
+
+
+@app.get("/index.html")
+def index_file():
+    if not INDEX_HTML.exists():
+        raise HTTPException(status_code=404, detail="index.html not found")
+    return FileResponse(INDEX_HTML)
+
+
+@app.get("/api/health")
+def health():
+    # Replace with real network + wallet when you wire Ostium SDK
+    return {
+        "ok": True,
+        "app_env": APP_ENV,
+        "network": "arbitrum",
+        "address": "0x0000000000000000000000000000000000000000",
+        "index_present": INDEX_HTML.exists(),
+    }
+
+@app.get("/api/balance")
+def balance():
+    # Replace with Ostium SDK balance call
+    return {"usdc": 0}
+
+@app.get("/api/orders")
+def orders():
+    # Replace with Ostium SDK open orders call
+    return {"orders": []}
+
+@app.get("/api/positions")
+def positions():
+    # Replace with Ostium SDK open positions call
+    return {"positions": []}
+
+@app.post("/api/trade")
+def trade(req: TradeRequest):
+    side = req.side.upper()
+    if side not in ("BUY", "SELL"):
+        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
+
+    # ---- WHERE YOU PLUG OSTIUM SDK IN ----
+    # 1) Validate inputs
+    if req.collateral <= 0 or req.leverage <= 0:
+        raise HTTPException(status_code=400, detail="collateral and leverage must be > 0")
+
+    # 2) Place trade using Ostium SDK (server-side)
+    # tx_hash = ostium.place_trade(...)
+    # For now, return a fake tx so your UI works:
+    fake_tx = hex(int(time.time()))  # just a placeholder
+    return {"ok": True, "tx_hash": fake_tx}
+
+@app.get("/api/candles")
+def candles(symbol: str, tf: str = "1h", limit: int = 500):
+    if not TWELVEDATA_KEY:
+        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_KEY on server")
+
+    interval, _ = map_tf(tf)
+    td_symbol = map_symbol_to_twelvedata(symbol)
+
+    # Twelve Data time_series endpoint (common pattern)
+    params = {
+        "symbol": td_symbol,
+        "interval": interval,
+        "outputsize": min(max(limit, 10), 5000),
+        "apikey": TWELVEDATA_KEY,
+        "format": "JSON",
+    }
+    r = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=REQUEST_TIMEOUT)
+    data = r.json()
+
+    if r.status_code != 200 or "values" not in data:
+        raise HTTPException(status_code=502, detail=f"TwelveData error: {data}")
+
+    # Twelve Data usually returns newest-first; your chart expects oldest->newest.
+    values = list(reversed(data["values"]))
+
+    out = []
+    for v in values:
+        # datetime format varies; easiest: parse with dateutil in real code.
+        # Here we assume ISO-like and let Python parse in a minimal way:
+        # Better production parsing is recommended.
+        dt = v.get("datetime")
+        if not dt:
+            continue
+
+        # naive conversion: use requests to parse is too much here; keep it simple:
+        # Expect "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
+        try:
+            # If you want robust parsing: pip install python-dateutil and parse
+            import datetime as _dt
+            if " " in dt:
+                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
+            else:
+                ts = int(_dt.datetime.strptime(dt, "%Y-%m-%d").timestamp() * 1000)
+        except Exception:
+            continue
+
+        out.append({
+            "time": ts,
+            "open": v.get("open"),
+            "high": v.get("high"),
+            "low": v.get("low"),
+            "close": v.get("close"),
+        })
+
+    return {"candles": out}
+
+
+@app.get("/api/intel/pairs")
+def intel_pairs():
+    return {
+        "pairs": [
+            {"key": pair, "label": config["label"], "query_count": len(config["queries"])}
+            for pair, config in PAIR_INTEL_CONFIG.items()
+        ],
+        "sources": GOV_RSS_FEEDS,
+    }
+
+
+@app.get("/api/intel/live", response_model=IntelResponse)
+def intel_live(pair: str = "XRP", hours: int = 24, limit: int = 50):
+    pair = pair.upper().strip()
+    if pair not in PAIR_INTEL_CONFIG:
+        raise HTTPException(status_code=400, detail=f"Unsupported pair '{pair}'.")
+    if hours < 1 or hours > 168:
+        raise HTTPException(status_code=400, detail="hours must be in range 1..168")
+    if limit < 5 or limit > 200:
+        raise HTTPException(status_code=400, detail="limit must be in range 5..200")
+
+    try:
+        return load_live_intel(pair=pair, hours=hours, limit=limit)
+    except Exception as exc:
+        raise HTTPException(status_code=502, detail=f"Failed loading intelligence feed: {exc}") from exc
