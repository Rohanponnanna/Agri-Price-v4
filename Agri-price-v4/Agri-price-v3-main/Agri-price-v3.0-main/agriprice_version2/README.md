---
title: AgriPrice Forecast
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
---

# 🌾 AgriPrice Forecast — Real-Time Crop Price Prediction

Multimodal ML system for India's agricultural MSMEs.

Combines ISRO Bhuvan NDVI satellite data, Agmarknet APMC mandi price feeds,
and OpenWeather climate data to generate a 7-day crop price forecast,
delivered via WhatsApp Cloud API.

## Quick Start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables

Set these in Streamlit Cloud → Settings → Secrets,
or in Hugging Face Spaces → Settings → Repository Secrets:

```
MANDI_API_KEY = "your_key"
OPENWEATHER_KEY = "your_key"
WHATSAPP_TOKEN = "your_token"
WHATSAPP_PHONE_ID = "your_phone_number_id"
WHATSAPP_TEMPLATE_NAME = "hello_world"   # or your approved template
WHATSAPP_TEMPLATE_LANG = "en_US"
```

For local runs you can instead copy `.env.example` to `.env` — it is loaded automatically.

## WhatsApp troubleshooting

Use **🔌 Test keys** in the app to verify token + phone-number ID without sending anything.

| Meta error | Meaning / fix |
|---|---|
| 190 | Token expired/invalid. 24-hour test tokens expire daily — create a System User permanent token. |
| 100 | Wrong Phone Number ID (use the *Phone number ID* from API Setup, not the phone number or WABA ID). |
| 131030 | Recipient not on the test-number allow-list (API Setup → "To" → add & verify the number). |
| 131047 | 24-hour window closed. Use *Template* mode (the default *Auto* mode falls back automatically). |
| 132001 | Template name / language code doesn't match an approved template. |


The app runs in demo mode with realistic synthetic data
if no API keys are provided.

## Folder Structure

```
agriprice-forecast/        ← repo root
├── app.py                 ← Streamlit entry point
├── requirements.txt       ← dependencies
├── README.md
├── .env.example
├── models/
│   ├── __init__.py
│   └── ensemble.py        ← XGBoost + ARIMA blender
└── utils/
    ├── __init__.py
    ├── data_fetcher.py    ← ISRO / OpenWeather / Agmarknet
    ├── feature_engineering.py
    └── whatsapp.py        ← Meta Cloud API
```

## Stack

- ISRO Bhuvan NDVI API
- Agmarknet APMC mandi prices
- OpenWeather One Call API
- XGBoost + ARIMA ensemble
- Meta WhatsApp Cloud API
- Streamlit / Hugging Face Spaces
