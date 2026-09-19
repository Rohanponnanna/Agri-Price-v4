"""
Real-Time Crop Price Forecasting for Agri-MSMEs
NDVI + Weather + Mandi Price → XGBoost + ARIMA Ensemble → WhatsApp Alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time, os, json, warnings
warnings.filterwarnings("ignore")

# Load keys from a local .env file (if present) BEFORE anything reads os.getenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils.data_fetcher import fetch_weather_data, fetch_mandi_prices, fetch_ndvi_data
from utils.feature_engineering import build_features
from models.ensemble import CropPriceForecast
from utils.whatsapp import send_whatsapp_alert, check_credentials, get_credentials, normalize_phone

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriPrice Forecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main palette */
    :root {
        --green: #1D9E75;
        --green-light: #E1F5EE;
        --amber: #BA7517;
        --amber-light: #FAEEDA;
        --red: #E24B4A;
        --red-light: #FCEBEB;
        --blue: #185FA5;
        --blue-light: #E6F1FB;
        --text: #2C2C2A;
        --muted: #5F5E5A;
        --border: rgba(136,135,128,0.2);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #F9F8F5;
        border-right: 1px solid var(--border);
    }

    /* Metric cards */
    .metric-card {
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: var(--text);
        line-height: 1.1;
    }
    .metric-delta {
        font-size: 13px;
        margin-top: 3px;
    }
    .delta-up   { color: var(--green); }
    .delta-down { color: var(--red); }

    /* Signal badge */
    .signal-buy  { background: var(--green-light); color: #085041;
                    border: 1px solid #9FE1CB; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }
    .signal-sell { background: var(--red-light);   color: #791F1F;
                    border: 1px solid #F7C1C1; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }
    .signal-hold { background: var(--amber-light); color: #633806;
                    border: 1px solid #FAC775; border-radius: 20px;
                    padding: 4px 14px; font-size: 13px; font-weight: 500; }

    /* Section header */
    .section-header {
        font-size: 15px;
        font-weight: 500;
        color: var(--text);
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* Status pills */
    .pill-green { background: var(--green-light); color: #085041;
                  border-radius: 4px; padding: 2px 8px; font-size: 11px; }
    .pill-amber { background: var(--amber-light); color: #633806;
                  border-radius: 4px; padding: 2px 8px; font-size: 11px; }

    /* Table styling */
    .forecast-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .forecast-table th {
        background: #F1EFE8; font-weight: 500; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.04em;
        padding: 8px 12px; text-align: left; color: var(--muted);
    }
    .forecast-table td { padding: 8px 12px; border-bottom: 1px solid var(--border); }

    /* WhatsApp button */
    .wa-btn {
        background: #25D366; color: #fff; border: none; border-radius: 8px;
        padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer;
        display: inline-flex; align-items: center; gap: 8px;
    }

    /* Hide streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
CROPS = {
    "Tomato": {"emoji": "🍅", "unit": "₹/Quintal", "season": "Year-round"},
    "Onion":  {"emoji": "🧅", "unit": "₹/Quintal", "season": "Oct–Mar"},
    "Potato": {"emoji": "🥔", "unit": "₹/Quintal", "season": "Nov–Apr"},
    "Wheat":  {"emoji": "🌾", "unit": "₹/Quintal", "season": "Mar–Apr"},
    "Rice":   {"emoji": "🌾", "unit": "₹/Quintal", "season": "Oct–Nov"},
    "Cotton": {"emoji": "🌱", "unit": "₹/Quintal", "season": "Oct–Jan"},
    "Soybean":{"emoji": "🌿", "unit": "₹/Quintal", "season": "Sep–Nov"},
    "Maize":  {"emoji": "🌽", "unit": "₹/Quintal", "season": "Sep–Oct"},
}

MANDIS = {
    "Karnataka": ["Bangalore (APMC)", "Hubballi", "Mysuru", "Davanagere", "Belgaum"],
    "Maharashtra": ["Mumbai (Vashi)", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Punjab":      ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain"],
    "Uttar Pradesh":  ["Lucknow", "Agra", "Varanasi", "Kanpur", "Meerut"],
    "Andhra Pradesh": ["Visakhapatnam", "Guntur", "Vijayawada", "Kurnool", "Tirupati"],
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriPrice Forecast")
    st.markdown("<p style='color:#5F5E5A;font-size:13px;margin-top:-8px'>Satellite + Mandi + Weather Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Select Crop & Location**")
    crop = st.selectbox("Crop", list(CROPS.keys()),
                        format_func=lambda x: f"{CROPS[x]['emoji']} {x}")
    state = st.selectbox("State", list(MANDIS.keys()))
    mandi = st.selectbox("APMC Mandi", MANDIS[state])

    st.divider()
    st.markdown("**Forecast Settings**")
    forecast_days = st.slider("Forecast horizon (days)", 3, 14, 7)
    confidence   = st.select_slider("Confidence interval", ["70%", "80%", "90%", "95%"], value="80%")
    show_history = st.slider("Historical data (days)", 14, 90, 30)

    st.divider()
    st.markdown("**WhatsApp Alerts**")
    phone = st.text_input("Farmer phone (+91XXXXXXXXXX)", placeholder="+919876543210")

    st.divider()
    st.markdown("**API Keys**")
    with st.expander("Configure (optional for demo)"):
        mandi_key = st.text_input("Mandi API key (data.gov.in)", type="password",
                                   value=os.getenv("MANDI_API_KEY",""),
                                   help="Free key from data.gov.in — powers live "
                                        "APMC mandi prices. Get one at "
                                        "https://data.gov.in/user/register")
        ow_key   = st.text_input("OpenWeather API key", type="password",
                                  value=os.getenv("OPENWEATHER_KEY",""))
        wa_token = st.text_input("WhatsApp token",      type="password",
                                  value=os.getenv("WHATSAPP_TOKEN",""))
        wa_phone_id = st.text_input("WA Phone ID",      type="password",
                                     value=os.getenv("WHATSAPP_PHONE_ID",""))
        # Apply immediately – no separate "Save" click required.
        for _k, _v in {
            "MANDI_API_KEY":     mandi_key,
            "OPENWEATHER_KEY":   ow_key,
            "WHATSAPP_TOKEN":    wa_token,
            "WHATSAPP_PHONE_ID": wa_phone_id,
        }.items():
            if _v and _v.strip():
                os.environ[_k] = _v.strip()

    run = st.button("🔍 Run Forecast", use_container_width=True, type="primary")


# ─── Main content ─────────────────────────────────────────────────────────────
# st.button() is only True for ONE rerun. Any other click (e.g. "Send WhatsApp
# Alert") triggers a rerun where `run` is False, which previously hit
# st.stop() and made the send button do nothing. We therefore keep the last
# forecast in st.session_state and reuse it on every later rerun.
if not run and "bundle" in st.session_state:
    _b = st.session_state["bundle"]
    crop, state, mandi = _b["crop"], _b["state"], _b["mandi"]
    forecast_days, confidence = _b["forecast_days"], _b["confidence"]

_mandi_key_set = bool(os.getenv("MANDI_API_KEY", ""))
_header_pill = (
    "<span class='pill-green'>● Live mandi key configured</span>"
    if _mandi_key_set else
    "<span class='pill-amber'>● Demo mode — add a Mandi API key for live prices</span>"
)
st.markdown(f"## {CROPS[crop]['emoji']} {crop} Price Forecast — {mandi}")
st.markdown(
    f"{_header_pill} &nbsp;"
    f"<span style='color:#5F5E5A;font-size:13px'>{mandi} &nbsp;·&nbsp; {state} &nbsp;·&nbsp; {forecast_days}-day horizon &nbsp;·&nbsp; {confidence} confidence</span>",
    unsafe_allow_html=True,
)
st.write("")

if not run and "bundle" not in st.session_state:
    # Landing state
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(
            "👈  Configure your crop, mandi, and location in the sidebar, then click **Run Forecast** to fetch live data and generate a 7-day price prediction.",
            icon="ℹ️",
        )
        if not _mandi_key_set:
            st.caption(
                "Add your data.gov.in **Mandi API key** in the sidebar's API Keys "
                "section to pull real, current APMC mandi prices instead of demo data."
            )
    with col_b:
        st.markdown("""
        **Data sources used**
        - 🛰 ISRO Bhuvan NDVI (crop health)
        - 🌦 OpenWeather API (climate)
        - 📊 data.gov.in / Agmarknet APMC (live mandi prices)
        - 🤖 XGBoost + ARIMA ensemble
        """)
    st.stop()


# ─── Data loading (only when "Run Forecast" is clicked) ───────────────────────
if run:
    with st.spinner("Fetching satellite, weather, and mandi data…"):
        progress = st.progress(0, text="Connecting to ISRO Bhuvan NDVI…")
        time.sleep(0.4)

        ndvi_df     = fetch_ndvi_data(crop, state, days=show_history)
        progress.progress(30, text="Fetching OpenWeather climate data…")
        time.sleep(0.3)

        weather_df  = fetch_weather_data(state, days=show_history)
        progress.progress(55, text="Scraping Agmarknet mandi prices…")
        time.sleep(0.4)

        mandi_df    = fetch_mandi_prices(crop, mandi, state, days=show_history)
        mandi_source = mandi_df.attrs.get("source", "demo")
        progress.progress(75, text="Building feature matrix…")
        time.sleep(0.2)

        features_df = build_features(ndvi_df, weather_df, mandi_df)
        progress.progress(90, text="Running ensemble model…")
        time.sleep(0.3)

        model    = CropPriceForecast()
        results  = model.forecast(features_df, horizon=forecast_days, ci=int(confidence.strip("%")))
        progress.progress(100, text="Done.")
        time.sleep(0.2)
        progress.empty()

    st.session_state["bundle"] = {
        "crop": crop, "state": state, "mandi": mandi,
        "forecast_days": forecast_days, "confidence": confidence,
        "results": results, "mandi_source": mandi_source,
    }

_bundle       = st.session_state["bundle"]
results       = _bundle["results"]
mandi_source  = _bundle["mandi_source"]


# ─── Unpack results ───────────────────────────────────────────────────────────
forecast   = results["forecast"]       # list of dicts: date, price, low, high
history    = results["history"]        # list of dicts: date, price
metrics    = results["metrics"]        # dict: rmse, mae, mape
signal     = results["signal"]         # BUY / SELL / HOLD
signal_reason = results["signal_reason"]

last_price = history[-1]["price"] if history else forecast[0]["price"]
next_price = forecast[0]["price"]
week_high  = max(f["high"] for f in forecast)
week_low   = min(f["low"]  for f in forecast)
pct_change = (forecast[-1]["price"] - last_price) / last_price * 100

# ─── Mandi data source indicator ───────────────────────────────────────────────
_source_copy = {
    "live":          ("🟢", f"Today's price for **{mandi}** is live from the data.gov.in mandi feed."),
    "live-anchored": ("🟡", f"Only sparse live reports for **{mandi}** — today's price is real; "
                            "the rest of the history is modeled and anchored to it."),
    "demo":          ("⚪", "Showing demo data. Add a Mandi API key in the sidebar for live prices."),
}
_icon, _msg = _source_copy.get(mandi_source, _source_copy["demo"])
st.caption(f"{_icon} {_msg}")

# ─── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

for col, label, value, delta, delta_pos in [
    (c1, "Today's Price",   f"₹{last_price:,.0f}",  f"+{pct_change:.1f}% projected", pct_change >= 0),
    (c2, "Tomorrow",        f"₹{next_price:,.0f}",  f"{'▲' if next_price>last_price else '▼'} ₹{abs(next_price-last_price):.0f}", next_price >= last_price),
    (c3, f"{forecast_days}-Day High", f"₹{week_high:,.0f}", "Upper band",  True),
    (c4, f"{forecast_days}-Day Low",  f"₹{week_low:,.0f}",  "Lower band",  False),
    (c5, "Forecast MAPE",   f"{metrics['mape']:.1f}%", f"RMSE ₹{metrics['rmse']:.0f}", metrics['mape'] < 8),
]:
    d_class = "delta-up" if delta_pos else "delta-down"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta {d_class}">{delta}</div>
    </div>""", unsafe_allow_html=True)

st.write("")

# ─── Signal banner ────────────────────────────────────────────────────────────
sig_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]
sig_icon  = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸"}[signal]
st.markdown(
    f"**Market signal:** &nbsp;<span class='{sig_class}'>{sig_icon} {signal}</span>"
    f"&nbsp; <span style='color:#5F5E5A;font-size:13px'>{signal_reason}</span>",
    unsafe_allow_html=True,
)
st.write("")


# ─── Price forecast ───────────────────────────────────────────────────────────
# ── 7 days before / 7 days after price table ──────────────────────────
st.markdown(
    "<div class='section-header'>Price outlook — 7 days before &amp; after today</div>",
    unsafe_allow_html=True,
)

N_BEFORE = 7
N_AFTER  = 7

hist_tail = history[-N_BEFORE:] if len(history) >= N_BEFORE else history
fc_head   = forecast[:N_AFTER]

rows = ""
prev_price = None

# Historical rows (oldest -> newest, ending at "today")
for i, h in enumerate(hist_tail):
    price = h["price"]
    if prev_price is None:
        chg_str, chg_color = "—", "#888780"
    else:
        chg = price - prev_price
        chg_str = f"{'▲' if chg>0 else ('▼' if chg<0 else '—')} ₹{abs(chg):,.0f}"
        chg_color = "#085041" if chg >= 0 else "#791F1F"
    prev_price = price

    is_today = (i == len(hist_tail) - 1)
    day_label = f"{h['date']} (Today)" if is_today else h["date"]
    row_style = "background:#F1EFE8;" if is_today else ""

    rows += f"""
    <tr style="{row_style}">
      <td><strong>{day_label}</strong></td>
      <td><span class="pill-amber">Actual</span></td>
      <td>₹{price:,.0f}</td>
      <td>—</td>
      <td style="color:{chg_color};font-weight:500">{chg_str}</td>
    </tr>"""

# Forecast rows (tomorrow -> +7 days)
for i, f in enumerate(fc_head):
    price = f["price"]
    chg = price - prev_price
    chg_str = f"{'▲' if chg>0 else ('▼' if chg<0 else '—')} ₹{abs(chg):,.0f}"
    chg_color = "#085041" if chg >= 0 else "#791F1F"
    prev_price = price

    day_label = "Tomorrow" if i == 0 else f["date"]
    rows += f"""
    <tr>
      <td><strong>{day_label}</strong></td>
      <td><span class="pill-green">Forecast</span></td>
      <td>₹{price:,.0f}</td>
      <td style="color:#888780">₹{f['low']:,.0f} – ₹{f['high']:,.0f}</td>
      <td style="color:{chg_color};font-weight:500">{chg_str}</td>
    </tr>"""

st.markdown(f"""
<table class="forecast-table">
  <thead><tr>
    <th>Date</th><th>Type</th><th>Price ₹/Q</th>
    <th>{confidence} Range</th><th>Day-on-day change</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

st.caption(
    f"Showing the last {len(hist_tail)} recorded day(s) up to today, followed by the next "
    f"{len(fc_head)} forecast day(s). Row highlighted in grey marks today."
)

csv_export = pd.DataFrame(
    [{"date": h["date"], "type": "Actual", "price": h["price"], "low": None, "high": None}
     for h in hist_tail]
    + [{"date": ("Tomorrow" if i == 0 else f["date"]), "type": "Forecast",
        "price": f["price"], "low": f["low"], "high": f["high"]}
       for i, f in enumerate(fc_head)]
)
st.download_button(
    "⬇️ Download this table (CSV)",
    csv_export.to_csv(index=False).encode(),
    f"{crop}_{mandi}_7day_before_after_{datetime.now():%Y%m%d}.csv",
    "text/csv",
)


# ─── WhatsApp section ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📱 Send WhatsApp Alert")

wa_col1, wa_col2 = st.columns([2, 1])

preview_msg = f"""🌾 *{crop} Price Alert — {mandi}*

📅 7-Day Forecast ({forecast[0]['date']} → {forecast[-1]['date']})

💰 *Today:* ₹{last_price:,.0f}/Quintal
📈 *7-Day High:* ₹{week_high:,.0f}/Quintal
📉 *7-Day Low:* ₹{week_low:,.0f}/Quintal

🎯 *Signal: {signal}* — {signal_reason}

📊 Source: NDVI + APMC + OpenWeather
🤖 Model: XGBoost + ARIMA Ensemble (MAPE: {metrics['mape']:.1f}%)

_AgriPrice Forecast System — Powered by Anthropic AI_"""

with wa_col1:
    st.text_area("Message preview", preview_msg, height=260, disabled=True)

with wa_col2:
    st.markdown("**Cooperative list**")
    farmer_numbers = st.text_area(
        "Phone numbers (one per line)",
        placeholder="+919876543210\n+919812345678\n+919898989898",
        height=110,
    )
    batch_send = st.checkbox("Send to all cooperative members")

    wa_mode_label = st.selectbox(
        "Delivery mode",
        ["Auto (text, then template fallback)", "Template only", "Text only"],
        help="WhatsApp only delivers free-form text if the farmer messaged your "
             "business number in the last 24 hours. Otherwise an approved "
             "template is required. 'Auto' handles this for you.",
    )
    wa_mode = {"Auto (text, then template fallback)": "auto",
               "Template only": "template", "Text only": "text"}[wa_mode_label]

    with st.expander("Template settings"):
        tpl_name = st.text_input("Template name",
                                 value=os.getenv("WHATSAPP_TEMPLATE_NAME", "hello_world"))
        tpl_lang = st.text_input("Template language code",
                                 value=os.getenv("WHATSAPP_TEMPLATE_LANG", "en_US"))
        st.caption(
            "`hello_world` is Meta's built-in test template (no variables). "
            "For your own template, the body must contain 4 variables: "
            "{{1}} crop, {{2}} price, {{3}} signal, {{4}} mandi."
        )

    btn_test, btn_send = st.columns(2)
    test_clicked = btn_test.button("🔌 Test keys", use_container_width=True)
    send_clicked = btn_send.button("📲 Send", use_container_width=True, type="primary")

# Credentials come from the sidebar fields first, then env / .env / secrets
_cred = get_credentials(wa_token or None, wa_phone_id or None)

if test_clicked:
    with st.spinner("Checking WhatsApp credentials…"):
        chk = check_credentials(_cred["token"], _cred["phone_id"])
    if chk["ok"]:
        st.success(
            f"✅ Credentials valid — sender: **{chk.get('display_phone_number','')}** "
            f"({chk.get('verified_name','')})"
        )
    else:
        st.error(f"❌ {chk['error']}")
        if chk.get("hint"):
            st.info(chk["hint"], icon="💡")

if send_clicked:
    numbers = []
    if phone:
        numbers.append(phone)
    if batch_send and farmer_numbers:
        numbers.extend([n.strip() for n in farmer_numbers.strip().split("\n") if n.strip()])

    if not numbers:
        st.warning("Please enter at least one phone number (sidebar field, or tick "
                   "'Send to all cooperative members' and fill the list).")
    elif not _cred["token"] or not _cred["phone_id"]:
        st.error("WhatsApp token / Phone ID missing. Add them under "
                 "**API Keys** in the sidebar (or in .env / Streamlit secrets).")
    else:
        template_params = [crop, f"₹{last_price:,.0f}/Quintal", signal, mandi]
        with st.spinner(f"Sending to {len(numbers)} number(s)…"):
            results_wa = send_whatsapp_alert(
                numbers, preview_msg, mode=wa_mode,
                template_name=tpl_name, template_lang=tpl_lang,
                template_params=template_params,
                token=_cred["token"], phone_id=_cred["phone_id"],
            )
        ok = [r for r in results_wa if r.get("success")]
        bad = [r for r in results_wa if not r.get("success")]
        if ok:
            st.success(f"✅ WhatsApp accepted the message for {len(ok)} number(s).")
            for r in ok:
                note = f" — {r['note']}" if r.get("note") else ""
                st.caption(f"+{r['phone']} · {r['mode']} · id {r.get('message_id','')[:24]}…{note}")
            st.caption("'Accepted' means Meta queued it. If it doesn't arrive, the "
                       "recipient may be outside the 24-hour window or not on your "
                       "test-number allow-list — use Template mode.")
        for r in bad:
            st.error(f"❌ {r['phone']}: {r.get('error','Unknown error')}"
                     + (f"  (code {r['code']})" if r.get("code") else ""))
            if r.get("hint"):
                st.info(r["hint"], icon="💡")


# ─── Download ─────────────────────────────────────────────────────────────────
st.divider()
dl_col1, dl_col2, _ = st.columns([1, 1, 3])

forecast_df = pd.DataFrame(forecast)
with dl_col1:
    st.download_button(
        "⬇️ Download forecast CSV",
        forecast_df.to_csv(index=False).encode(),
        f"{crop}_{mandi}_forecast_{datetime.now():%Y%m%d}.csv",
        "text/csv",
        use_container_width=True,
    )
with dl_col2:
    combined = {
        "crop": crop, "mandi": mandi, "state": state,
        "generated_at": datetime.now().isoformat(),
        "signal": signal, "metrics": metrics,
        "forecast": forecast,
    }
    st.download_button(
        "⬇️ Download full JSON",
        json.dumps(combined, indent=2, default=str).encode(),
        f"{crop}_{mandi}_full_{datetime.now():%Y%m%d}.json",
        "application/json",
        use_container_width=True,
    )

# Footer
st.markdown(
    "<div style='text-align:center;color:#B4B2A9;font-size:11px;margin-top:2rem'>"
    "AgriPrice Forecast · ISRO Bhuvan NDVI · Agmarknet APMC · OpenWeather · "
    "XGBoost + ARIMA Ensemble · Meta WhatsApp Cloud API"
    "</div>",
    unsafe_allow_html=True,
)
