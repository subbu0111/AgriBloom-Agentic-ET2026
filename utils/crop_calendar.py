"""
Crop Calendar & Government Schemes for Indian Agriculture
State-wise crop recommendations by season with ICAR guidelines.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# Indian agricultural seasons
SEASONS = {
    "kharif": {"months": [6, 7, 8, 9, 10], "name": "Kharif (Monsoon)", "sowing": "Jun-Jul", "harvest": "Oct-Nov"},
    "rabi": {"months": [11, 12, 1, 2, 3], "name": "Rabi (Winter)", "sowing": "Oct-Nov", "harvest": "Mar-Apr"},
    "zaid": {"months": [3, 4, 5, 6], "name": "Zaid (Summer)", "sowing": "Mar-Apr", "harvest": "Jun-Jul"},
}


# State-wise crop recommendations by season (ICAR guidelines)
STATE_CROP_CALENDAR = {
    "Telangana": {
        "kharif": ["Rice (Paddy)", "Cotton", "Maize", "Red Gram (Tur)", "Soybean", "Chillies"],
        "rabi": ["Rice (2nd crop)", "Bengal Gram", "Groundnut", "Sunflower", "Maize", "Jowar"],
        "zaid": ["Watermelon", "Muskmelon", "Green Gram (Moong)", "Cucumber", "Vegetables"],
    },
    "Andhra Pradesh": {
        "kharif": ["Rice", "Cotton", "Maize", "Groundnut", "Sugarcane", "Tobacco"],
        "rabi": ["Rice", "Bengal Gram", "Black Gram", "Sunflower", "Chillies"],
        "zaid": ["Vegetables", "Watermelon", "Green Gram", "Sesame"],
    },
    "Karnataka": {
        "kharif": ["Rice", "Ragi", "Maize", "Sugarcane", "Cotton", "Groundnut"],
        "rabi": ["Wheat", "Jowar", "Bengal Gram", "Sunflower", "Safflower"],
        "zaid": ["Vegetables", "Watermelon", "Cucumber", "Moong Dal"],
    },
    "Tamil Nadu": {
        "kharif": ["Rice", "Cotton", "Sugarcane", "Groundnut", "Millets"],
        "rabi": ["Rice (Samba)", "Pulses", "Oil Seeds", "Banana"],
        "zaid": ["Vegetables", "Sesame", "Green Gram"],
    },
    "Maharashtra": {
        "kharif": ["Rice", "Soybean", "Cotton", "Sugarcane", "Tur Dal", "Jowar"],
        "rabi": ["Wheat", "Gram", "Jowar", "Onion", "Safflower"],
        "zaid": ["Vegetables", "Watermelon", "Cucumber"],
    },
    "Uttar Pradesh": {
        "kharif": ["Rice", "Sugarcane", "Maize", "Bajra", "Soybean"],
        "rabi": ["Wheat", "Mustard", "Potato", "Gram", "Pea", "Lentil"],
        "zaid": ["Watermelon", "Muskmelon", "Moong", "Vegetables"],
    },
    "Punjab": {
        "kharif": ["Rice", "Cotton", "Maize", "Sugarcane", "Bajra"],
        "rabi": ["Wheat", "Mustard", "Potato", "Gram", "Sunflower"],
        "zaid": ["Moong", "Vegetables", "Fodder crops"],
    },
    "Madhya Pradesh": {
        "kharif": ["Soybean", "Rice", "Maize", "Cotton", "Tur Dal"],
        "rabi": ["Wheat", "Gram", "Mustard", "Linseed", "Lentil"],
        "zaid": ["Vegetables", "Moong", "Watermelon"],
    },
    "Gujarat": {
        "kharif": ["Cotton", "Groundnut", "Rice", "Bajra", "Castor"],
        "rabi": ["Wheat", "Mustard", "Gram", "Cumin", "Potato"],
        "zaid": ["Vegetables", "Watermelon", "Sesame"],
    },
    "West Bengal": {
        "kharif": ["Rice (Aman)", "Jute", "Maize", "Vegetables"],
        "rabi": ["Rice (Boro)", "Potato", "Mustard", "Wheat", "Vegetables"],
        "zaid": ["Vegetables", "Sesame", "Moong"],
    },
    "Rajasthan": {
        "kharif": ["Bajra", "Moth Bean", "Guar", "Groundnut", "Sesame"],
        "rabi": ["Wheat", "Mustard", "Gram", "Barley", "Cumin"],
        "zaid": ["Watermelon", "Cucumber", "Moong"],
    },
    "Odisha": {
        "kharif": ["Rice", "Maize", "Groundnut", "Cotton", "Vegetables"],
        "rabi": ["Rice (2nd)", "Pulses", "Mustard", "Potato", "Vegetables"],
        "zaid": ["Vegetables", "Watermelon", "Green Gram"],
    },
}

# Default for states not listed
DEFAULT_CROPS = {
    "kharif": ["Rice", "Maize", "Cotton", "Groundnut", "Pulses"],
    "rabi": ["Wheat", "Mustard", "Gram", "Potato", "Vegetables"],
    "zaid": ["Watermelon", "Moong Dal", "Vegetables", "Cucumber"],
}


# Government schemes for Indian farmers (2025-26)
GOVT_SCHEMES = [
    {
        "name": "PM-KISAN",
        "benefit": "₹6,000/year direct transfer (3 installments of ₹2,000)",
        "eligibility": "All land-holding farmer families",
        "url": "pmkisan.gov.in",
    },
    {
        "name": "PMFBY (Pradhan Mantri Fasal Bima Yojana)",
        "benefit": "Crop insurance at 2% premium (Kharif), 1.5% (Rabi), 5% (Horticulture)",
        "eligibility": "All farmers with crop loans + voluntary for others",
        "url": "pmfby.gov.in",
    },
    {
        "name": "eNAM (National Agriculture Market)",
        "benefit": "Sell crops online to any mandi in India — better price discovery",
        "eligibility": "All farmers with Aadhaar + bank account",
        "url": "enam.gov.in",
    },
    {
        "name": "PM-KUSUM (Solar Pump Scheme)",
        "benefit": "60% subsidy on solar pumps (2HP to 10HP)",
        "eligibility": "Farmers with irrigation needs",
        "url": "mnre.gov.in",
    },
    {
        "name": "Soil Health Card Scheme",
        "benefit": "Free soil testing with crop-wise fertilizer recommendations",
        "eligibility": "All farmers — visit nearest KVK",
        "url": "soilhealth.dac.gov.in",
    },
    {
        "name": "KCC (Kisan Credit Card)",
        "benefit": "Crop loan at 4% interest (with subsidy) for up to ₹3 lakh",
        "eligibility": "All farmers, sharecroppers, tenant farmers",
        "url": "Apply at any bank branch",
    },
]


def get_current_season() -> dict[str, Any]:
    """Get the current agricultural season based on month."""
    month = datetime.now().month

    for season_key, season_info in SEASONS.items():
        if month in season_info["months"]:
            return {"key": season_key, **season_info}

    return {"key": "rabi", **SEASONS["rabi"]}


def get_next_season() -> dict[str, Any]:
    """Get the next agricultural season."""
    current = get_current_season()
    season_order = ["kharif", "rabi", "zaid"]
    current_idx = season_order.index(current["key"])
    next_key = season_order[(current_idx + 1) % 3]
    return {"key": next_key, **SEASONS[next_key]}


def get_crop_calendar(state: str) -> dict[str, Any]:
    """
    Get comprehensive crop calendar for a state.

    Returns:
        Dictionary with current season, recommended crops, next season info,
        and applicable government schemes.
    """
    current_season = get_current_season()
    next_season = get_next_season()

    # Get state-specific or default crops
    state_crops = STATE_CROP_CALENDAR.get(state, DEFAULT_CROPS)

    current_crops = state_crops.get(current_season["key"], DEFAULT_CROPS[current_season["key"]])
    next_crops = state_crops.get(next_season["key"], DEFAULT_CROPS[next_season["key"]])

    # Format scheme info
    scheme_summaries = [
        f"{s['name']}: {s['benefit']}" for s in GOVT_SCHEMES
    ]

    return {
        "current_season": current_season["name"],
        "current_sowing": current_season["sowing"],
        "current_harvest": current_season["harvest"],
        "recommended_crops": current_crops,
        "next_season": next_season["name"],
        "next_season_sowing": next_season["sowing"],
        "next_season_crops": next_crops,
        "schemes": scheme_summaries,
        "schemes_detailed": GOVT_SCHEMES,
        "state": state,
    }


__all__ = ["get_crop_calendar", "get_current_season", "get_next_season", "GOVT_SCHEMES"]
