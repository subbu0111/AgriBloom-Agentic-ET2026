"""
OpenRouter LLM Client for AgriBloom Agentic
Generates expert agricultural advisory using Gemini via OpenRouter API.
Supports multimodal input — sends crop photos directly to Gemini for analysis.
Falls back to template responses if API is unavailable.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


def _get_api_key() -> Optional[str]:
    """Get OpenRouter API key from environment."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        logger.warning("OPENROUTER_API_KEY not set — LLM features disabled")
        return None
    return key


def _image_to_base64(image) -> Optional[str]:
    """Convert PIL Image to base64 data URL for multimodal API."""
    try:
        from PIL import Image
        if image is None:
            return None
        if not isinstance(image, Image.Image):
            return None

        # Resize to max 512px to save tokens/bandwidth
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size))

        # Convert to JPEG bytes
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=80)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"Could not encode image: {e}")
        return None


def _build_system_prompt(lang: str) -> str:
    """Build the system prompt for the agricultural advisor."""
    lang_names = {
        "en": "English", "hi": "Hindi", "te": "Telugu", "kn": "Kannada",
        "ta": "Tamil", "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
        "bn": "Bengali", "or": "Odia",
    }
    lang_name = lang_names.get(lang, "English")

    return f"""You are AgriBloom AI — an expert Indian agricultural advisor and plant pathologist. You provide practical, actionable advice to Indian farmers.

IMPORTANT: If a crop photo is attached, ANALYZE THE PHOTO DIRECTLY to identify the plant, disease, and health status. Your visual diagnosis takes priority over any model-predicted labels.

RULES:
1. Respond ENTIRELY in {lang_name} language (use the script of that language, not transliteration)
2. Use emojis for section headers (🌾 💊 💰 🌤️ 🌱 ⚠️ 📋)
3. Use NUMBERED LISTS (1. 2. 3.) for all points — NEVER use asterisks (*) or bullet points
4. Be specific with product names, dosages (g/L, ml/L), and timings
5. Include local market intelligence and selling advice
6. Mention relevant government schemes (PM-KISAN, PMFBY, eNAM, KCC)
7. Give weather-based timing advice covering the ENTIRE crop growth cycle (typically 3-4 months) — use the 14-day forecast data provided plus your knowledge of typical seasonal weather patterns for the region
8. Suggest next season crops based on region and current month
9. Keep response under 500 words — farmers need concise advice
10. Use ₹ for prices, quintal for weight, hectare for area
11. End with nearest KVK contact suggestion

FORMAT your response as:
🌾 [Advisory Report Title]
═══════════════════════
🔬 [Disease Detection — YOUR visual diagnosis of the photo]
💊 [Immediate Treatment — numbered steps with products + dosage]
💰 [Market Intelligence — current price, MSP comparison, when to sell]
🌤️ [Seasonal Weather Outlook — cover next 3 months or full crop duration, not just 14 days]
🌱 [Next Season Recommendation — what to grow based on region]
📋 [Government Schemes — PM-KISAN, PMFBY, KCC, eNAM]
⚠️ [Important Warnings]"""


def _build_user_content(context: dict[str, Any]) -> list:
    """Build multimodal user content with image + text."""
    disease = context.get("disease", {})
    weather = context.get("weather", {})
    forecast = context.get("forecast", [])
    market = context.get("market", {})
    location = context.get("location", {})
    crop_calendar = context.get("crop_calendar", {})
    treatment = context.get("treatment", "")
    user_query = context.get("user_query", "")
    image = context.get("image")

    # Format 14-day forecast
    forecast_text = ""
    if forecast:
        forecast_lines = []
        for day in forecast[:14]:
            forecast_lines.append(
                f"  Day {day.get('day', '?')}: {day.get('temp_max', '?')}°C, "
                f"Rain: {day.get('rain', 0)}mm, Humidity: {day.get('humidity', '?')}%"
            )
        forecast_text = "\n".join(forecast_lines)

    # Format crop calendar
    calendar_text = ""
    if crop_calendar:
        current_season = crop_calendar.get("current_season", "")
        recommended_crops = ", ".join(crop_calendar.get("recommended_crops", []))
        next_season = crop_calendar.get("next_season", "")
        next_crops = ", ".join(crop_calendar.get("next_season_crops", []))
        schemes = "\n".join(f"  - {s}" for s in crop_calendar.get("schemes", []))
        calendar_text = f"""
Current Season: {current_season}
Recommended Crops: {recommended_crops}
Next Season: {next_season}
Next Season Crops: {next_crops}
Government Schemes:
{schemes}"""

    # Crop growth durations (days)
    crop_durations = {
        "rice": "120-150 days (4-5 months)",
        "maize": "90-120 days (3-4 months)",
        "wheat": "120-150 days (4-5 months)",
        "tomato": "90-120 days (3-4 months)",
        "potato": "90-120 days (3-4 months)",
        "sugarcane": "270-365 days (9-12 months)",
        "ragi": "90-120 days (3-4 months)",
        "cotton": "150-180 days (5-6 months)",
    }
    detected_crop = market.get('crop', 'unknown')
    crop_duration = crop_durations.get(detected_crop, "90-120 days (3-4 months)")

    text_prompt = f"""Analyze this crop photo and farming situation. Provide expert advisory:

MODEL PRE-SCREENING (may be inaccurate — use YOUR visual analysis):
  Model Label: {disease.get('label', 'unknown')}
  Original Label: {disease.get('original_label', 'unknown')}
  Confidence: {disease.get('confidence', 0):.0%}
  Known Treatment: {treatment}

FARMER LOCATION:
  State: {location.get('state', 'Unknown')}
  District: {location.get('district', 'Unknown')}

CURRENT WEATHER:
  Temperature: {weather.get('temp_c', 'N/A')}°C
  Humidity: {weather.get('humidity', 'N/A')}%
  Rainfall: {weather.get('rain_mm', 0)}mm
  Wind: {weather.get('wind_kmh', 'N/A')} km/h

14-DAY WEATHER FORECAST (use this + your regional knowledge to project full season):
{forecast_text or '  Not available'}

CROP GROWTH DURATION: {crop_duration}
(Project weather outlook for the FULL crop cycle based on region + season)

MARKET DATA:
  Crop: {detected_crop}
  Current Price: ₹{market.get('modal_price', 'N/A')}/quintal
  MSP: ₹{market.get('msp', 'N/A')}/quintal
  Mandi: {market.get('mandi', 'Local')}
  Price Trend: {market.get('trend', 'stable')}

CROP CALENDAR & SCHEMES:
{calendar_text or '  Not available'}

FARMER QUERY: {user_query or 'Please analyze this crop photo and give full advisory.'}

IMPORTANT: 
1. Look at the photo carefully. Identify the actual plant and disease. The model label above may be wrong — trust YOUR visual analysis.
2. Use NUMBERED LISTS (1. 2. 3.) for all points. Do NOT use asterisks (*) or bullet points.
3. Cover seasonal weather for the FULL crop growth cycle ({crop_duration}), not just 14 days."""

    # Build multimodal content
    content_parts = []

    # Try to add image
    image_b64 = _image_to_base64(image)
    if image_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": image_b64},
        })
        logger.info("Sending crop photo to Gemini for visual analysis")

    # Add text
    content_parts.append({
        "type": "text",
        "text": text_prompt,
    })

    return content_parts


def generate_llm_response(
    context: dict[str, Any],
    lang: str = "en",
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """
    Generate advisory response using OpenRouter LLM with multimodal support.

    Args:
        context: Dictionary with disease, weather, market, location, image data
        lang: Language code for response
        model: OpenRouter model identifier

    Returns:
        Generated response text, or None if API call fails
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    system_prompt = _build_system_prompt(lang)
    user_content = _build_user_content(context)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://huggingface.co/spaces/subbu0111/AgriBloom-Agentic",
        "X-Title": "AgriBloom Agentic",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    try:
        logger.info(f"Calling OpenRouter API ({model})...")
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        logger.info(f"LLM response received ({len(content)} chars)")
        return content

    except requests.exceptions.Timeout:
        logger.error("OpenRouter API timeout (45s)")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"OpenRouter API failed: {e}")
        return None


__all__ = ["generate_llm_response"]
