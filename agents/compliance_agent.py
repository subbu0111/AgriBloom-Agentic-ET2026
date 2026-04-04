"""
Compliance Agent - FSSAI/ICAR Regulatory Guardrails
Enforces Indian agricultural chemical regulations and safety standards
Generates audit trails for traceability
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default compliance rules (fallback if JSON missing)
DEFAULT_RULES = {
    "banned_inputs": {
        "pesticides": [
            # Banned pesticides in India (as per CIB&RC)
            "endosulfan", "methyl parathion", "monocrotophos", "phosphamidon",
            "carbofuran", "phorate", "triazophos", "diazinon", "fenthion",
            "dichlorvos", "methyl bromide", "ethylene dibromide", "aldrin",
            "chlordane", "dieldrin", "heptachlor", "mirex", "toxaphene",
            "lindane", "ddt", "barium carbonate", "ethyl mercury chloride",
            "sodium cyanide", "captafol", "nicotine sulfate", "pentachlorophenol",
            "benzene hexachloride", "carbaryl", "paraquat dimethyl sulphate",
        ],
        "fertilizers": [
            # Restricted fertilizers
            "urea above 46%", "unregistered bio-fertilizer",
        ],
        "growth_regulators": [
            "calcium carbide", "ethephon above limit", "oxytocin for crops",
        ],
    },
    "max_residue_limits": {
        # MRL in mg/kg as per FSSAI
        "carbendazim": 0.5,
        "mancozeb": 3.0,
        "propiconazole": 0.1,
        "chlorpyrifos": 0.2,
        "imidacloprid": 0.5,
        "cypermethrin": 0.5,
    },
    "required_disclaimers": [
        "Validate all recommendations with your local Krishi Vigyan Kendra (KVK).",
        "Follow integrated pest management (IPM) practices.",
        "Wear protective equipment when handling agrochemicals.",
        "Observe pre-harvest interval (PHI) before crop harvesting.",
        "This is AI-generated advice. Consult agricultural officer for critical decisions.",
    ],
    "regional_disclaimers": {
        "hi": [
            "सभी सिफारिशों को अपने स्थानीय कृषि विज्ञान केंद्र (KVK) से सत्यापित करें।",
            "यह AI-जनित सलाह है। महत्वपूर्ण निर्णयों के लिए कृषि अधिकारी से परामर्श करें।",
        ],
        "kn": [
            "ಎಲ್ಲಾ ಶಿಫಾರಸುಗಳನ್ನು ನಿಮ್ಮ ಸ್ಥಳೀಯ KVK ಯಿಂದ ಮಾನ್ಯಮಾಡಿ.",
            "ಇದು AI-ಉತ್ಪಾದಿಸಿದ ಸಲಹೆ. ನಿರ್ಣಾಯಕ ನಿರ್ಧಾರಗಳಿಗೆ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ.",
        ],
        "te": [
            "మీ స్థానిక KVK నుండి అన్ని సిఫార్సులను ధృవీకరించండి.",
            "ఇది AI-ఉత్పత్తి సలహా. క్లిష్టమైన నిర్ణయాల కోసం వ్యవసాయ అధికారిని సంప్రదించండి.",
        ],
        "ta": [
            "உங்கள் உள்ளூர் KVK இலிருந்து அனைத்து பரிந்துரைகளையும் சரிபார்க்கவும்.",
            "இது AI-உருவாக்கிய ஆலோசனை. முக்கிய முடிவுகளுக்கு விவசாய அதிகாரியை அணுகவும்.",
        ],
    },
    "icar_guidelines": {
        "spray_timing": "Avoid spraying during flowering to protect pollinators.",
        "water_interval": "Maintain 24-48 hour interval between irrigation and spraying.",
        "mixing": "Never mix more than 2 pesticides. Test compatibility first.",
    },
    "meta": {
        "version": "2.0.0",
        "last_updated": "2026-03-25",
        "source": "CIB&RC, FSSAI, ICAR Guidelines",
    },
}


def _load_rules(path: str = "compliance/compliance_rules.json") -> dict[str, Any]:
    """Load compliance rules from JSON file with fallback."""
    rules_path = Path(path)

    if rules_path.exists():
        try:
            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded compliance rules v{rules.get('meta', {}).get('version', 'unknown')}")
            return rules
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse compliance rules: {e}")

    logger.warning("Using default compliance rules")
    return DEFAULT_RULES


def _check_banned_substances(text: str, rules: dict) -> list[str]:
    """Check for banned substances in text."""
    banned_lists = rules.get("banned_inputs", {})
    text_lower = text.lower()

    violations = []

    for category, substances in banned_lists.items():
        for substance in substances:
            if substance.lower() in text_lower:
                violations.append(f"{substance} ({category})")

    return violations


def _check_mrl_compliance(recommendations: list[str], rules: dict) -> list[dict]:
    """Check if mentioned chemicals exceed MRL limits."""
    mrl_limits = rules.get("max_residue_limits", {})
    warnings = []

    for rec in recommendations:
        rec_lower = rec.lower()
        for chemical, limit in mrl_limits.items():
            if chemical.lower() in rec_lower:
                warnings.append({
                    "chemical": chemical,
                    "mrl_limit": f"{limit} mg/kg",
                    "warning": f"Ensure {chemical} residue stays below {limit} mg/kg at harvest",
                })

    return warnings


def _get_disclaimers(rules: dict, lang: str = "en") -> list[str]:
    """Get disclaimers in the appropriate language."""
    if lang != "en" and lang in rules.get("regional_disclaimers", {}):
        return rules["regional_disclaimers"][lang]
    return rules.get("required_disclaimers", [])


def run_compliance(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Check recommendations against FSSAI/ICAR compliance rules.

    Input state keys:
        - knowledge: Dict with recommendations
        - user_text: User's query text
        - treatment: Disease treatment recommendation
        - lang: Language code

    Output state keys:
        - compliance: Compliance report with status
        - status: "compliance_complete" or "compliance_blocked"
    """
    rules = _load_rules()
    lang = state.get("lang", "en")

    # Gather all text to check
    recommendations = state.get("recommendations", [])
    treatment = state.get("treatment", "")
    user_text = state.get("user_text", "")

    all_text = " ".join([
        user_text,
        treatment,
        " ".join(recommendations),
    ])

    # Check for banned substances
    violations = _check_banned_substances(all_text, rules)

    # Check MRL compliance
    mrl_warnings = _check_mrl_compliance(recommendations, rules)

    # Get appropriate disclaimers
    disclaimers = _get_disclaimers(rules, lang)

    # Build compliance report
    allowed = len(violations) == 0
    risk_level = "low" if allowed and len(mrl_warnings) == 0 else ("high" if not allowed else "medium")

    compliance_report = {
        "allowed": allowed,
        "risk_level": risk_level,
        "violations": violations,
        "mrl_warnings": mrl_warnings,
        "disclaimers": disclaimers,
        "icar_guidelines": rules.get("icar_guidelines", {}),
        "rule_version": rules.get("meta", {}).get("version", "unknown"),
        "checked_at": datetime.now().isoformat(),
    }

    # Log compliance check
    logger.info(
        f"Compliance check: allowed={allowed}, violations={len(violations)}, "
        f"mrl_warnings={len(mrl_warnings)}, risk={risk_level}"
    )

    return {
        **state,
        "compliance": compliance_report,
        "status": "compliance_complete" if allowed else "compliance_blocked",
    }


# Export
__all__ = ["run_compliance", "DEFAULT_RULES"]
