"""
WhatsApp alert delivery via Meta WhatsApp Cloud API.

Why messages often silently fail (and how this module handles it)
-----------------------------------------------------------------
1. Free-form *text* messages are only delivered if the farmer has messaged
   your business number in the last 24 hours (Meta error 131047). Outside that
   window only an approved *template* message is delivered.
   -> mode="auto" tries text first, then falls back to a template.
2. With a Meta *test* number, the recipient must be added to the
   "To" allow-list in the Meta developer dashboard (error 131030).
3. Phone numbers must be digits only, with country code and no "+" / spaces.
   -> `normalize_phone()` cleans this (10-digit numbers get +91).
4. Expired / temporary tokens (error 190) — 24-hour tokens from the API Setup
   page stop working next day; create a System User permanent token.

Environment variables
---------------------
  WHATSAPP_TOKEN          Meta access token                        (required)
  WHATSAPP_PHONE_ID       WhatsApp Business *phone number ID*      (required)
  WHATSAPP_TEMPLATE_NAME  approved template name (default: hello_world)
  WHATSAPP_TEMPLATE_LANG  template language code (default: en_US)
  WHATSAPP_API_VERSION    Graph API version        (default: v21.0)
  WHATSAPP_DEFAULT_CC     country code for 10-digit numbers (default: 91)
"""

import os
import re
from typing import Dict, List, Optional

import requests

DEFAULT_TIMEOUT = 20

# Meta error code -> human friendly hint
ERROR_HINTS = {
    190:    "Access token is invalid or expired. Generate a new (permanent) token "
            "from a System User in Meta Business Manager.",
    100:    "Invalid parameter — usually a wrong Phone Number ID. Use the *Phone "
            "number ID* from WhatsApp → API Setup (not the phone number itself, "
            "and not the WhatsApp Business Account ID).",
    10:     "Token lacks the 'whatsapp_business_messaging' permission.",
    200:    "Token lacks permission for this phone number ID.",
    131030: "Recipient is not in the allowed list. With a Meta test number, add "
            "this phone under API Setup → 'To' and verify it with the OTP.",
    131047: "24-hour window closed — the farmer has not messaged your number in "
            "the last 24 h. Send an approved template message instead (or ask "
            "the farmer to send 'Hi' to your business number first).",
    131026: "Message undeliverable — the number may not be on WhatsApp.",
    131056: "Too many messages to this number in a short time — try later.",
    131042: "Payment method missing / billing issue on the WhatsApp Business Account.",
    132000: "Template parameter count doesn't match the approved template.",
    132001: "Template name/language not found or not approved. Check the exact "
            "name and language code in WhatsApp Manager → Message templates.",
    132005: "Template text too long after parameter substitution.",
    132012: "Template parameter format mismatch.",
    133010: "The sender phone number is not registered on the Cloud API.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_credentials(token: Optional[str] = None,
                    phone_id: Optional[str] = None) -> Dict[str, str]:
    """Explicit args win, then env vars. Values are stripped (pasted tokens
    frequently carry a trailing space/newline, which makes Meta reject them)."""
    token = (token or os.getenv("WHATSAPP_TOKEN", "") or "").strip().strip('"').strip("'")
    phone_id = (phone_id or os.getenv("WHATSAPP_PHONE_ID", "") or "").strip().strip('"').strip("'")
    return {"token": token, "phone_id": phone_id}


def normalize_phone(raw: str, default_cc: Optional[str] = None) -> str:
    """
    Return the number in the digits-only international format Meta expects.
      '+91 98765 43210' -> '919876543210'
      '9876543210'      -> '919876543210'
      '09876543210'     -> '919876543210'
    """
    default_cc = default_cc or os.getenv("WHATSAPP_DEFAULT_CC", "91")
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return ""
    if raw.strip().startswith("00"):          # 0091...
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = default_cc + digits[1:]
    elif len(digits) == 10:
        digits = default_cc + digits
    return digits


def _base_url(phone_id: str) -> str:
    version = os.getenv("WHATSAPP_API_VERSION", "v21.0").strip() or "v21.0"
    return f"https://graph.facebook.com/{version}/{phone_id}"


def _post(url: str, token: str, payload: dict) -> dict:
    """POST to Graph API and return a normalised result dict."""
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        return {"success": False, "error": f"Network error: {exc}", "code": None}

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.ok and data.get("messages"):
        return {"success": True,
                "message_id": data["messages"][0].get("id", ""),
                "status_code": resp.status_code}

    err = data.get("error", {}) if isinstance(data, dict) else {}
    code = err.get("code")
    sub = err.get("error_subcode")
    details = (err.get("error_data") or {}).get("details", "")
    msg = err.get("message") or f"HTTP {resp.status_code}"
    if details:
        msg = f"{msg} — {details}"
    hint = ERROR_HINTS.get(code) or ERROR_HINTS.get(sub, "")
    return {"success": False, "error": msg, "code": code, "hint": hint,
            "status_code": resp.status_code}


# ── Public API ────────────────────────────────────────────────────────────────
def check_credentials(token: Optional[str] = None,
                      phone_id: Optional[str] = None) -> dict:
    """
    Verify the token + phone number ID pair (no message is sent).
    Returns {ok, display_phone_number, verified_name, error, hint}.
    """
    cred = get_credentials(token, phone_id)
    if not cred["token"] or not cred["phone_id"]:
        return {"ok": False, "error": "WHATSAPP_TOKEN or WHATSAPP_PHONE_ID not set"}
    try:
        resp = requests.get(
            _base_url(cred["phone_id"]),
            headers={"Authorization": f"Bearer {cred['token']}"},
            params={"fields": "display_phone_number,verified_name,quality_rating"},
            timeout=DEFAULT_TIMEOUT,
        )
        data = resp.json()
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "error": f"Network error: {exc}"}

    if resp.ok and "id" in data:
        return {"ok": True,
                "display_phone_number": data.get("display_phone_number", ""),
                "verified_name": data.get("verified_name", ""),
                "quality_rating": data.get("quality_rating", "")}
    err = data.get("error", {})
    return {"ok": False, "error": err.get("message", f"HTTP {resp.status_code}"),
            "hint": ERROR_HINTS.get(err.get("code"), "")}


def send_text_message(phone: str, message: str,
                      token: Optional[str] = None,
                      phone_id: Optional[str] = None) -> dict:
    cred = get_credentials(token, phone_id)
    to = normalize_phone(phone)
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }
    res = _post(_base_url(cred["phone_id"]) + "/messages", cred["token"], payload)
    res.update({"phone": to, "mode": "text"})
    return res


def send_template_message(phone: str,
                          template_name: Optional[str] = None,
                          language: Optional[str] = None,
                          body_params: Optional[List[str]] = None,
                          token: Optional[str] = None,
                          phone_id: Optional[str] = None) -> dict:
    """
    Send an approved template. `body_params` fill {{1}}, {{2}}… in the template
    body. For Meta's built-in 'hello_world' template no parameters are sent.
    """
    cred = get_credentials(token, phone_id)
    to = normalize_phone(phone)
    template_name = (template_name or os.getenv("WHATSAPP_TEMPLATE_NAME", "hello_world")).strip()
    language = (language or os.getenv("WHATSAPP_TEMPLATE_LANG", "en_US")).strip()

    template = {"name": template_name, "language": {"code": language}}
    if body_params and template_name != "hello_world":
        template["components"] = [{
            "type": "body",
            "parameters": [{"type": "text", "text": str(p).replace("\n", " ")[:1000]}
                           for p in body_params],
        }]
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "template",
        "template": template,
    }
    res = _post(_base_url(cred["phone_id"]) + "/messages", cred["token"], payload)
    res.update({"phone": to, "mode": f"template:{template_name}"})
    return res


def send_whatsapp_alert(phone_numbers: List[str],
                        message: str,
                        mode: str = "auto",
                        template_name: Optional[str] = None,
                        template_lang: Optional[str] = None,
                        template_params: Optional[List[str]] = None,
                        token: Optional[str] = None,
                        phone_id: Optional[str] = None) -> List[dict]:
    """
    Send an alert to one or more phone numbers.

    mode:
      "text"     – free-form text only (needs the 24-hour window)
      "template" – approved template only (works any time)
      "auto"     – text first; if Meta reports the 24-hour window is closed,
                   automatically retry with the template

    Returns a list of dicts: {phone, success, mode, message_id | error, hint}
    """
    cred = get_credentials(token, phone_id)
    if not cred["token"] or not cred["phone_id"]:
        return [{"phone": p, "success": False, "mode": mode,
                 "error": "WHATSAPP_TOKEN or WHATSAPP_PHONE_ID not set"}
                for p in phone_numbers]

    results, seen = [], set()
    for raw in phone_numbers:
        to = normalize_phone(raw)
        if not to or len(to) < 11:
            results.append({"phone": raw, "success": False, "mode": mode,
                            "error": "Invalid phone number — use the format +919876543210"})
            continue
        if to in seen:                             # avoid duplicate sends
            continue
        seen.add(to)

        def _template():
            return send_template_message(to, template_name, template_lang,
                                         template_params, cred["token"], cred["phone_id"])

        if mode == "template":
            res = _template()
        else:
            res = send_text_message(to, message, cred["token"], cred["phone_id"])
            if mode == "auto" and not res["success"] and res.get("code") in (131047, 131051, 470):
                fallback = _template()
                fallback["note"] = "24-hour window closed — sent as template instead"
                res = fallback
        results.append(res)
    return results


def send_template_alert(phone: str, crop: str, price: float,
                        signal: str, mandi: str) -> dict:
    """
    Backwards-compatible helper for the 'agri_price_alert' template.
    Parameters: {{1}} crop, {{2}} price, {{3}} signal, {{4}} mandi
    """
    return send_template_message(
        phone,
        template_name=os.getenv("WHATSAPP_TEMPLATE_NAME", "agri_price_alert"),
        language=os.getenv("WHATSAPP_TEMPLATE_LANG", "en"),
        body_params=[crop, f"₹{price:,.0f}/Quintal", signal, mandi],
    )
