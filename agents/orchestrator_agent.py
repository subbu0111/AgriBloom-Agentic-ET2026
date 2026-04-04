"""
Orchestrator Agent - Request Router and Session Manager
Routes requests to appropriate agents based on input analysis
Maintains conversation history and session state
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Supported languages with metadata
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "tts_code": "en"},
    "hi": {"name": "Hindi", "native": "हिंदी", "tts_code": "hi"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "tts_code": "kn"},
    "te": {"name": "Telugu", "native": "తెలుగు", "tts_code": "te"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "tts_code": "ta"},
}

# Crop keywords for routing
CROP_KEYWORDS = {
    "maize": ["maize", "corn", "makka", "ಮೆಕ್ಕೆಜೋಳ", "మొక్కజొన్న", "சோளம்"],
    "tomato": ["tomato", "tamatar", "ಟೊಮ್ಯಾಟೊ", "టమాటా", "தக்காளி"],
    "potato": ["potato", "aloo", "ಆಲೂಗಡ್ಡೆ", "బంగాళదుంప", "உருளைக்கிழங்கு"],
    "rice": ["rice", "paddy", "dhan", "chawal", "ಅಕ್ಕಿ", "వరి", "நெல்"],
    "wheat": ["wheat", "gehun", "ಗೋಧಿ", "గోధుమ", "கோதுமை"],
    "ragi": ["ragi", "finger millet", "nachni", "ರಾಗಿ", "రాగి", "கேழ்வரகு"],
    "sugarcane": ["sugarcane", "ganna", "ಕಬ್ಬು", "చెరకు", "கரும்பு"],
}

# Intent keywords
INTENT_KEYWORDS = {
    "disease": ["disease", "blight", "rust", "rot", "infection", "problem", "issue", "leaf", "yellowing", "spots"],
    "weather": ["weather", "rain", "temperature", "forecast", "mausam"],
    "market": ["price", "market", "mandi", "sell", "buy", "rate", "cost"],
    "treatment": ["treatment", "spray", "medicine", "fungicide", "pesticide", "cure", "solution"],
}


def _detect_language(text: str, default: str = "en") -> str:
    """Detect language from text based on script detection."""
    if not text:
        return default

    # Simple script detection
    for char in text:
        code = ord(char)
        # Devanagari (Hindi)
        if 0x0900 <= code <= 0x097F:
            return "hi"
        # Kannada
        if 0x0C80 <= code <= 0x0CFF:
            return "kn"
        # Telugu
        if 0x0C00 <= code <= 0x0C7F:
            return "te"
        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"

    return default


def _detect_crop_from_text(text: str) -> str | None:
    """Detect crop type from text using keywords."""
    text_lower = text.lower()

    for crop, keywords in CROP_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return crop

    return None


def _detect_intent(text: str, has_image: bool) -> str:
    """Detect user intent from text and context."""
    text_lower = text.lower() if text else ""

    if has_image:
        return "disease_detection"

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return intent

    return "general_inquiry"


def _determine_route(
    has_image: bool,
    intent: str,
) -> Literal["vision_first", "knowledge_first"]:
    """Determine the processing route based on input analysis."""
    # Vision-first route when image is provided
    if has_image:
        return "vision_first"

    # Knowledge-first for non-image queries
    return "knowledge_first"


def run_orchestrator(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Orchestrate request routing and session management.

    Input state keys:
        - image: PIL Image (optional)
        - user_text: User's query text
        - user_language: Language code
        - offline: Offline mode flag
        - lat, lon: Location coordinates

    Output state keys:
        - chat_history: Updated conversation history
        - route: "vision_first" or "knowledge_first"
        - detected_intent: User intent classification
        - detected_crop: Crop mentioned in text (if any)
        - lang: Normalized language code
        - status: "orchestrated"
    """
    # Extract inputs
    user_text = (state.get("user_text") or "").strip()
    user_language = state.get("user_language", "en")
    has_image = state.get("image") is not None
    offline = bool(state.get("offline", False))

    # Initialize or get chat history
    chat_history = state.get("chat_history", [])

    # Normalize language
    if user_language not in SUPPORTED_LANGUAGES:
        detected_lang = _detect_language(user_text, "en")
        user_language = detected_lang

    # Detect intent and crop
    intent = _detect_intent(user_text, has_image)
    detected_crop = _detect_crop_from_text(user_text)

    # Determine route
    route = _determine_route(has_image, intent)

    # Log session event
    session_event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "orchestrator_received",
        "text": user_text[:100] if user_text else None,  # Truncate for privacy
        "language": user_language,
        "has_image": has_image,
        "offline": offline,
        "route": route,
        "intent": intent,
        "detected_crop": detected_crop,
    }
    chat_history.append(session_event)

    logger.info(
        f"Orchestrator: route={route}, intent={intent}, "
        f"lang={user_language}, crop={detected_crop}, offline={offline}"
    )

    return {
        **state,
        "chat_history": chat_history,
        "route": route,
        "detected_intent": intent,
        "detected_crop": detected_crop,
        "lang": user_language,
        "status": "orchestrated",
    }


# Export
__all__ = ["run_orchestrator", "SUPPORTED_LANGUAGES", "CROP_KEYWORDS"]
