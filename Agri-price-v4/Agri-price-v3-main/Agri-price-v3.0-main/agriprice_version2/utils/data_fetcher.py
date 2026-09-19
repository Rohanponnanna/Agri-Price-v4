"""
Data fetchers for ISRO Bhuvan NDVI, OpenWeather, and Agmarknet APMC.
Falls back to realistic synthetic data when APIs are unavailable (demo mode).
"""

import os, requests, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─── NDVI from ISRO Bhuvan ────────────────────────────────────────────────────
# Real endpoint: https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms
# Returns GeoTIFF; parse mean NDVI per region bounding box.
# Synthetic fallback mimics real NDVI seasonal cycle.

CROP_STATE_COORDS = {
    ("Tomato",  "Karnataka"):        (12.9716, 77.5946),
    ("Onion",   "Maharashtra"):      (19.9975, 73.7898),
    ("Wheat",   "Punjab"):           (30.9010, 75.8573),
    ("Potato",  "Uttar Pradesh"):    (26.8467, 80.9462),
    ("Rice",    "Andhra Pradesh"):   (16.5062, 80.6480),
    ("Cotton",  "Maharashtra"):      (20.7002, 77.0082),
    ("Soybean", "Madhya Pradesh"):   (23.2599, 77.4126),
    ("Maize",   "Karnataka"):        (15.3173, 75.7139),
}

# Base NDVI health profile per crop (healthy = 0.65–0.85, stressed = 0.3–0.5)
CROP_NDVI_PROFILE = {
    "Tomato":  (0.62, 0.08), "Onion":   (0.58, 0.07),
    "Wheat":   (0.72, 0.09), "Potato":  (0.68, 0.08),
    "Rice":    (0.75, 0.10), "Cotton":  (0.60, 0.09),
    "Soybean": (0.70, 0.09), "Maize":   (0.65, 0.08),
}


def fetch_ndvi_data(crop: str, state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch NDVI time series for a crop-state pair.
    Live: ISRO Bhuvan WMS GeoTIFF → mean NDVI extraction.
    Demo: Synthetic seasonal NDVI with realistic noise.
    """
    api_key = os.getenv("BHUVAN_TOKEN", "")
    coords  = CROP_STATE_COORDS.get((crop, state), (20.5937, 78.9629))

    if api_key:
        try:
            return _live_ndvi(crop, state, coords, days, api_key)
        except Exception:
            pass

    return _synthetic_ndvi(crop, days)


def _live_ndvi(crop, state, coords, days, api_key):
    """Parse ISRO Bhuvan NDVI — placeholder for actual WMS integration."""
    # Real implementation:
    # 1. Build WMS GetMap request for BBOX around coords
    # 2. Parse returned GeoTIFF with rasterio
    # 3. Compute zonal statistics (mean NDVI) for crop area
    # 4. Return as daily DataFrame
    raise NotImplementedError("Bhuvan WMS integration requires rasterio + API key")


def _synthetic_ndvi(crop: str, days: int) -> pd.DataFrame:
    """Realistic synthetic NDVI with seasonal pattern + noise."""
    mu, sigma = CROP_NDVI_PROFILE.get(crop, (0.65, 0.08))
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    # Seasonal sine + random walk
    t       = np.linspace(0, 2 * np.pi, days)
    trend   = mu + 0.08 * np.sin(t - np.pi / 3)
    noise   = np.random.normal(0, sigma * 0.25, days)
    ndvi    = np.clip(trend + noise, 0.1, 0.95)
    return pd.DataFrame({"date": [d.isoformat() for d in dates], "ndvi": ndvi.round(4)})


# ─── Weather from OpenWeather ─────────────────────────────────────────────────
STATE_COORDS = {
    "Karnataka":        (12.9716, 77.5946),
    "Maharashtra":      (19.0760, 72.8777),
    "Punjab":           (30.9010, 75.8573),
    "Madhya Pradesh":   (23.2599, 77.4126),
    "Uttar Pradesh":    (26.8467, 80.9462),
    "Andhra Pradesh":   (17.3850, 78.4867),
}

MONSOON_PROFILE = {
    "Karnataka":        {"base_rain": 3.2, "base_temp": 26, "humidity": 72},
    "Maharashtra":      {"base_rain": 2.8, "base_temp": 28, "humidity": 68},
    "Punjab":           {"base_rain": 1.1, "base_temp": 24, "humidity": 55},
    "Madhya Pradesh":   {"base_rain": 2.5, "base_temp": 27, "humidity": 65},
    "Uttar Pradesh":    {"base_rain": 2.0, "base_temp": 28, "humidity": 62},
    "Andhra Pradesh":   {"base_rain": 3.5, "base_temp": 30, "humidity": 75},
}


def fetch_weather_data(state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch historical + forecast climate variables.
    Live: OpenWeather One Call API 3.0
    Demo: Synthetic climate with monsoon seasonality.
    """
    api_key = os.getenv("OPENWEATHER_KEY", "")
    coords  = STATE_COORDS.get(state, (20.5937, 78.9629))

    if api_key:
        try:
            return _live_weather(coords, days, api_key)
        except Exception:
            pass

    return _synthetic_weather(state, days)


def _live_weather(coords, days, api_key):
    """OpenWeather One Call API integration."""
    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    rows = []
    for i in range(min(days, 5)):   # free tier limit; production uses bulk history
        dt   = int((datetime.now() - timedelta(days=days - i)).timestamp())
        resp = requests.get(f"{url}&dt={dt}", timeout=10)
        if resp.status_code != 200:
            raise ValueError(f"OpenWeather error {resp.status_code}")
        data = resp.json().get("data", [{}])[0]
        rows.append({
            "date":      (datetime.now() - timedelta(days=days - i)).date().isoformat(),
            "temp_max":  data.get("temp", 28),
            "temp_min":  data.get("feels_like", 22),
            "humidity":  data.get("humidity", 65),
            "rainfall":  data.get("rain", {}).get("1h", 0) * 24,
            "wind_kmh":  data.get("wind_speed", 12) * 3.6,
        })
    return pd.DataFrame(rows)


def _synthetic_weather(state: str, days: int) -> pd.DataFrame:
    profile = MONSOON_PROFILE.get(state, {"base_rain": 2.0, "base_temp": 27, "humidity": 65})
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    month = datetime.now().month
    # Indian monsoon simulation: Jun–Sep high rain
    monsoon_factor = 1.0 + 2.5 * max(0, np.sin(np.pi * (month - 6) / 4)) if 6 <= month <= 9 else 0.4

    temp_base = profile["base_temp"]
    rows = []
    for i, d in enumerate(dates):
        rain = max(0, np.random.exponential(profile["base_rain"] * monsoon_factor))
        rows.append({
            "date":     d.isoformat(),
            "temp_max": round(temp_base + random.uniform(-2, 3), 1),
            "temp_min": round(temp_base - random.uniform(4, 8), 1),
            "humidity": round(profile["humidity"] + random.uniform(-8, 8), 1),
            "rainfall": round(rain, 1),
            "wind_kmh": round(random.uniform(8, 25), 1),
        })
    return pd.DataFrame(rows)


# ─── Mandi prices from data.gov.in (Agmarknet feed) ──────────────────────────
# Real endpoint: "Current Daily Price of Various Commodities for Various
# Markets (Mandi)" — a data.gov.in resource sourced from Agmarknet.
#   GET https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
#       ?api-key=<key>&format=json&filters[state.keyword]=...
#       &filters[commodity.keyword]=...&filters[market.keyword]=...
#
# IMPORTANT: this government feed is a LIVE / CURRENT-DAY snapshot, not a
# deep historical archive — on any given day it only holds rows for markets
# that actually reported prices that day, and it has no date-range filter.
# So real-time integration here means:
#   1. Query it for the exact crop + mandi + state the user picked.
#   2. If enough genuine reporting days come back, use them directly.
#   3. If only a few (or one) real point comes back — normal for smaller
#      mandis — build the rest of the required history synthetically but
#      ANCHOR it so it ends exactly on today's real, live price.
#   4. If the key is missing/invalid/the feed is unreachable, fall back to
#      the fully synthetic demo series (unchanged behaviour).
# Get a free key at https://data.gov.in/user/register and set it as the
# MANDI_API_KEY environment variable (or paste it in the sidebar).

DATA_GOV_MANDI_RESOURCE = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_MANDI_URL      = f"https://api.data.gov.in/resource/{DATA_GOV_MANDI_RESOURCE}"

CROP_PRICE_RANGES = {
    "Tomato":  {"base": 1800, "vol": 600, "trend": 0.002},
    "Onion":   {"base": 2200, "vol": 400, "trend": -0.001},
    "Potato":  {"base": 1600, "vol": 300, "trend": 0.001},
    "Wheat":   {"base": 2100, "vol": 150, "trend": 0.0005},
    "Rice":    {"base": 3200, "vol": 250, "trend": 0.001},
    "Cotton":  {"base": 6500, "vol": 400, "trend": -0.0005},
    "Soybean": {"base": 4800, "vol": 350, "trend": 0.002},
    "Maize":   {"base": 1900, "vol": 200, "trend": 0.001},
}


def fetch_mandi_prices(crop: str, mandi: str, state: str, days: int = 30) -> pd.DataFrame:
    """
    Fetch APMC mandi price history.
    Live: data.gov.in / Agmarknet daily mandi price feed (needs MANDI_API_KEY).
    Demo: Realistic synthetic price with volatility model.

    The returned DataFrame always has columns:
        date, modal_price, min_price, max_price, arrivals_q
    and df.attrs["source"] is set to "live" or "demo" so the UI can show
    which one was actually used.
    """
    api_key = os.getenv("MANDI_API_KEY", "")
    if api_key:
        try:
            df = _live_mandi_prices(crop, mandi, state, days)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"[mandi] live API fetch failed ({e}); falling back to demo data")

    return _synthetic_mandi_prices(crop, days)


def _clean_mandi_name(mandi: str) -> str:
    """'Bangalore (APMC)' -> 'Bangalore' — the govt feed uses plain market names."""
    return mandi.split("(")[0].strip()


def _parse_price(val):
    try:
        return float(str(val).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_mandi_record(rec: dict):
    """Turn one raw data.gov.in record into a clean row, or None if unusable."""
    try:
        d = datetime.strptime(rec.get("arrival_date", ""), "%d/%m/%Y").date()
    except ValueError:
        return None
    modal = _parse_price(rec.get("modal_price"))
    if modal is None or modal <= 0:
        return None
    mn = _parse_price(rec.get("min_price"))
    mx = _parse_price(rec.get("max_price"))
    return {
        "date":        d.isoformat(),
        "modal_price": modal,
        "min_price":   mn if mn is not None else round(modal * 0.95, 2),
        "max_price":   mx if mx is not None else round(modal * 1.05, 2),
    }


def _query_data_gov(crop: str, state: str, market: str = None, limit: int = 300) -> list:
    """One call to the live data.gov.in mandi resource. Raises on hard failures."""
    api_key = os.getenv("MANDI_API_KEY", "")
    params = {
        "api-key": api_key,
        "format":  "json",
        "limit":   limit,
        "filters[state.keyword]":     state,
        "filters[commodity.keyword]": crop,
    }
    if market:
        params["filters[market.keyword]"] = market

    resp = requests.get(DATA_GOV_MANDI_URL, params=params, timeout=12)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("records", [])
    return [row for row in (_parse_mandi_record(r) for r in records) if row]


def _live_mandi_prices(crop: str, mandi: str, state: str, days: int) -> pd.DataFrame:
    market = _clean_mandi_name(mandi)

    parsed = _query_data_gov(crop, state, market=market)
    if not parsed:
        # Market naming in the feed sometimes differs slightly from our
        # dropdown label (e.g. "Bengaluru" vs "Bangalore (APMC)") — retry
        # across the whole state for this crop instead of failing outright.
        parsed = _query_data_gov(crop, state, market=None)
    if not parsed:
        raise ValueError(f"no live records for {crop} in {state}/{mandi}")

    live = (
        pd.DataFrame(parsed)
        .groupby("date", as_index=False)
        .agg(modal_price=("modal_price", "mean"),
             min_price=("min_price", "mean"),
             max_price=("max_price", "mean"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    # The feed doesn't publish arrival quantity — approximate it from the
    # min–max spread (a wider spread usually means more varied arrivals)
    # so downstream lag/rolling features still have something to work with.
    spread = (live["max_price"] - live["min_price"]).clip(lower=1)
    live["arrivals_q"] = (spread * 6 + 150).round(0)
    live.attrs["source"] = "live"

    latest_price   = float(live["modal_price"].iloc[-1])
    n_live_days    = len(live)
    enough_history = n_live_days >= max(5, days // 3)

    if enough_history:
        pad_needed = days - n_live_days
        if pad_needed > 0:
            pad = _synthetic_mandi_prices(crop, pad_needed)
            pad = pad[pad["date"] < live["date"].iloc[0]]
            live = pd.concat([pad, live], ignore_index=True)
        live = live.tail(days).reset_index(drop=True)
        live.attrs["source"] = "live"
        return live

    # Sparse reporting history for this mandi: generate a realistic series
    # for the full window, but anchor it so it ends on today's real price,
    # and splice in genuine values on any day we did get real data for.
    anchored = _synthetic_mandi_prices(crop, days)
    scale = latest_price / anchored["modal_price"].iloc[-1]
    for col in ["modal_price", "min_price", "max_price"]:
        anchored[col] = (anchored[col] * scale).round(2)

    live_by_date = live.set_index("date")
    for i, row in anchored.iterrows():
        if row["date"] in live_by_date.index:
            r = live_by_date.loc[row["date"]]
            anchored.loc[i, ["modal_price", "min_price", "max_price"]] = [
                r["modal_price"], r["min_price"], r["max_price"]
            ]
    anchored.attrs["source"] = "live-anchored"
    return anchored


def _synthetic_mandi_prices(crop: str, days: int) -> pd.DataFrame:
    """Geometric Brownian Motion price simulation matching real mandi volatility."""
    p     = CROP_PRICE_RANGES.get(crop, {"base": 2000, "vol": 300, "trend": 0.001})
    dates = [datetime.now().date() - timedelta(days=days - i) for i in range(days)]
    price = p["base"]
    prices = []

    # GBM with regime switching (supply shock events)
    for i in range(days):
        shock  = 1.0
        if random.random() < 0.05:       # 5% chance of supply/demand shock
            shock = random.choice([0.85, 1.15])
        daily_return = p["trend"] + random.normalvariate(0, p["vol"] / p["base"] * 0.4)
        price = max(price * (1 + daily_return) * shock, p["base"] * 0.4)
        prices.append(round(price, 2))

    # Add min/max spread (mandis report range)
    rows = []
    for d, modal in zip(dates, prices):
        spread = random.uniform(0.03, 0.10) * modal
        rows.append({
            "date":        d.isoformat(),
            "modal_price": modal,
            "min_price":   round(modal - spread, 2),
            "max_price":   round(modal + spread * 1.2, 2),
            "arrivals_q":  round(random.uniform(200, 2000), 0),
        })
    df = pd.DataFrame(rows)
    df.attrs["source"] = "demo"
    return df
