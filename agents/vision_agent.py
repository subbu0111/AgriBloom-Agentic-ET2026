"""
Vision Agent - Production Crop Disease Detection
Uses ViT (Vision Transformer) with GPU acceleration for RTX 4060
Supports: Maize, Tomato, Potato, Rice, Wheat, Ragi, Sugarcane
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

from utils.onnx_inference import ONNXVisionEngine

logger = logging.getLogger(__name__)

# Full disease classes for Indian crops
TARGET_CLASSES = [
    # Maize diseases (PlantVillage)
    "maize_blight",
    "maize_common_rust",
    "maize_gray_leaf_spot",
    "maize_healthy",
    # Tomato diseases (PlantVillage)
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites",
    "tomato_target_spot",
    "tomato_mosaic_virus",
    "tomato_yellow_leaf_curl",
    "tomato_healthy",
    # Potato diseases (PlantVillage)
    "potato_early_blight",
    "potato_late_blight",
    "potato_healthy",
    # Rice diseases (Kaggle)
    "rice_bacterial_leaf_blight",
    "rice_brown_spot",
    "rice_leaf_blast",
    "rice_leaf_scald",
    "rice_narrow_brown_spot",
    "rice_healthy",
    # Wheat diseases (Kaggle)
    "wheat_brown_rust",
    "wheat_leaf_rust",
    "wheat_septoria",
    "wheat_yellow_rust",
    "wheat_healthy",
    # Ragi/Finger Millet diseases (Kaggle)
    "ragi_blast",
    "ragi_brown_spot",
    "ragi_healthy",
    # Sugarcane diseases (Kaggle)
    "sugarcane_bacterial_blight",
    "sugarcane_red_rot",
    "sugarcane_rust",
    "sugarcane_healthy",
]

# Crop-specific class mappings
CROP_CLASSES = {
    "maize": [c for c in TARGET_CLASSES if c.startswith("maize_")],
    "tomato": [c for c in TARGET_CLASSES if c.startswith("tomato_")],
    "potato": [c for c in TARGET_CLASSES if c.startswith("potato_")],
    "rice": [c for c in TARGET_CLASSES if c.startswith("rice_")],
    "wheat": [c for c in TARGET_CLASSES if c.startswith("wheat_")],
    "ragi": [c for c in TARGET_CLASSES if c.startswith("ragi_")],
    "sugarcane": [c for c in TARGET_CLASSES if c.startswith("sugarcane_")],
}

# Disease treatment recommendations (multilingual keys)
# Maps to actual model labels from config.json id2label
DISEASE_TREATMENTS = {
    # Maize diseases - actual model labels
    "maize_(maize)_northern_leaf_blight": {
        "en": "Apply Mancozeb 75% WP at 2.5g/L. Remove infected leaves. Improve field drainage.",
        "hi": "मैंकोज़ेब 75% WP को 2.5 ग्राम/लीटर पर लगाएं। संक्रमित पत्तियों को हटाएं।",
        "kn": "ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP ಅನ್ನು 2.5g/L ನಲ್ಲಿ ಹಾಕಿ। ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "te": "మాంకోజెబ్ 75% WP ను 2.5g/L వద్ద వర్తించండి. సోకిన ఆకులను తొలగించండి.",
        "ta": "மாங்கோசெப் 75% WP ஐ 2.5g/L இல் பயன்படுத்தவும். பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    },
    "maize_(maize)_common_rust_": {
        "en": "Spray Propiconazole 25% EC at 1ml/L at early stage. Ensure proper spacing.",
        "hi": "प्रारंभिक अवस्था में प्रोपिकोनाज़ोल 25% EC को 1ml/L पर छिड़कें।",
        "kn": "ಆರಂಭಿಕ ಹಂತದಲ್ಲಿ ಪ್ರೋಪಿಕೋನಾಜೋಲ್ 25% EC ಅನ್ನು 1ml/L ನಲ್ಲಿ ಸಿಂಪಡಿಸಿ.",
        "te": "ప్రారంభ దశలో ప్రొపికోనజోల్ 25% EC ను 1ml/L వద్ద స్ప్రే చేయండి.",
        "ta": "ஆரம்ப நிலையில் ப்ரோபிகோனசோல் 25% EC ஐ 1ml/L இல் தெளிக்கவும்.",
    },
    "maize_(maize)_cercospora_leaf_spot_gray_leaf_spot": {
        "en": "Apply Mancozeb or Chlorothalonil. Maintain proper plant spacing.",
        "hi": "मैंकोज़ेब या क्लोरोथालोनिल लगाएं। उचित पौधों की दूरी बनाए रखें।",
        "kn": "ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ ಕ್ಲೋರೋಥಲೋನಿಲ್ ಹಾಕಿ.",
        "te": "మాంకోజెబ్ లేదా క్లోరోథలోనిల్ వర్తించండి.",
        "ta": "மாங்கோசெப் அல்லது குளோரோதலோனில் பயன்படுத்தவும்.",
    },
    "maize_(maize)_healthy": {
        "en": "Crop is healthy. Continue regular monitoring and balanced fertilization.",
        "hi": "फसल स्वस्थ है। नियमित निगरानी और संतुलित उर्वरक जारी रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా పర్యవేక్షణ కొనసాగించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பை தொடரவும்.",
    },
    # Tomato diseases - actual model labels
    "tomato_bacterial_spot": {
        "en": "Apply Copper-based fungicide. Remove infected plants. Avoid overhead irrigation.",
        "hi": "तांबा-आधारित कवकनाशी लगाएं। संक्रमित पौधों को हटाएं।",
        "kn": "ತಾಮ್ರ-ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕ ಹಾಕಿ.",
        "te": "రాగి ఆధారిత శిలీంద్ర నాశిని వర్తించండి.",
        "ta": "காப்பர் அடிப்படையிலான பூஞ்சைக்கொல்லி பயன்படுத்தவும்.",
    },
    "tomato_early_blight": {
        "en": "Apply Chlorothalonil or Copper-based fungicide. Stake plants for air circulation.",
        "hi": "क्लोरोथालोनिल या तांबा-आधारित कवकनाशी लगाएं। हवा के संचार के लिए पौधों को सहारा दें।",
        "kn": "ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ತಾಮ್ರ-ಆಧಾರಿತ ಶಿಲೀಂಧ್ರನಾಶಕ ಹಾಕಿ.",
        "te": "క్లోరోథలోనిల్ లేదా రాగి ఆధారిత శిలీంద్ర నాశిని వర్తించండి.",
        "ta": "குளோரோதலோனில் அல்லது காப்பர் அடிப்படையிலான பூஞ்சைக்கொல்லி பயன்படுத்தவும்.",
    },
    "tomato_late_blight": {
        "en": "Emergency: Apply Metalaxyl + Mancozeb immediately. Remove all infected plants.",
        "hi": "आपातकालीन: तुरंत मेटालैक्सिल + मैंकोज़ेब लगाएं। सभी संक्रमित पौधों को हटाएं।",
        "kn": "ತುರ್ತು: ತಕ್ಷಣ ಮೆಟಾಲಾಕ್ಸಿಲ್ + ಮ್ಯಾಂಕೋಜೆಬ್ ಹಾಕಿ. ಸೋಂಕಿತ ಎಲ್ಲಾ ಗಿಡಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.",
        "te": "అత్యవసరం: వెంటనే మెటాలాక్సిల్ + మాంకోజెబ్ వర్తించండి. సోకిన అన్ని మొక్కలను తొలగించండి.",
        "ta": "அவசரம்: உடனடியாக மெட்டாலாக்சில் + மாங்கோசெப் பயன்படுத்தவும். பாதிக்கப்பட்ட செடிகளை அகற்றவும்.",
    },
    "tomato_leaf_mold": {
        "en": "Improve ventilation. Apply Mancozeb. Remove affected leaves.",
        "hi": "वेंटिलेशन में सुधार करें। मैंकोज़ेब लगाएं। प्रभावित पत्तियां हटाएं।",
        "kn": "ವಾತಾಯನ ಸುಧಾರಿಸಿ. ಮ್ಯಾಂಕೋಜೆಬ್ ಹಾಕಿ.",
        "te": "వెంటిలేషన్ మెరుగుపరచండి. మాంకోజెబ్ వర్తించండి.",
        "ta": "காற்றோட்டத்தை மேம்படுத்தவும். மாங்கோசெப் பயன்படுத்தவும்.",
    },
    "tomato_septoria_leaf_spot": {
        "en": "Apply Chlorothalonil. Remove lower infected leaves. Ensure good drainage.",
        "hi": "क्लोरोथालोनिल लगाएं। नीचे की संक्रमित पत्तियां हटाएं।",
        "kn": "ಕ್ಲೋರೋಥಲೋನಿಲ್ ಹಾಕಿ. ಕೆಳಗಿನ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ.",
        "te": "క్లోరోథలోనిల్ వర్తించండి. కింది సోకిన ఆకులను తొలగించండి.",
        "ta": "குளோரோதலோனில் பயன்படுத்தவும். கீழே பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    },
    "tomato_spider_mites_two_spotted_spider_mite": {
        "en": "Apply Abamectin or Dicofol. Increase humidity. Use predatory mites if available.",
        "hi": "एबामेक्टिन या डाइकोफोल लगाएं। आर्द्रता बढ़ाएं।",
        "kn": "ಅಬಾಮೆಕ್ಟಿನ್ ಅಥವಾ ಡೈಕೋಫೋಲ್ ಹಾಕಿ.",
        "te": "అబామెక్టిన్ లేదా డైకోఫాల్ వర్తించండి.",
        "ta": "அபாமெக்டின் அல்லது டைகோபால் பயன்படுத்தவும்.",
    },
    "tomato_target_spot": {
        "en": "Apply Chlorothalonil or Mancozeb. Remove infected plant debris.",
        "hi": "क्लोरोथालोनिल या मैंकोज़ेब लगाएं। संक्रमित पौधों के अवशेष हटाएं।",
        "kn": "ಕ್ಲೋರೋಥಲೋನಿಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ ಹಾಕಿ.",
        "te": "క్లోరోథలోనిల్ లేదా మాంకోజెబ్ వర్తించండి.",
        "ta": "குளோரோதலோனில் அல்லது மாங்கோசெப் பயன்படுத்தவும்.",
    },
    "tomato_mosaic_virus": {
        "en": "Remove infected plants. Disinfect tools. Use virus-free seeds next season.",
        "hi": "संक्रमित पौधों को हटाएं। उपकरण कीटाणुरहित करें।",
        "kn": "ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ತೆಗೆದುಹಾಕಿ. ಉಪಕರಣಗಳನ್ನು ಸೋಂಕುರಹಿತಗೊಳಿಸಿ.",
        "te": "సోకిన మొక్కలను తొలగించండి. పరికరాలను క్రిమిసంహారం చేయండి.",
        "ta": "பாதிக்கப்பட்ட செடிகளை அகற்றவும். கருவிகளை கிருமிநீக்கம் செய்யவும்.",
    },
    "tomato_yellow_leaf_curl_virus": {
        "en": "Control whitefly vectors with Imidacloprid. Remove infected plants.",
        "hi": "इमिडाक्लोप्रिड से सफेद मक्खी को नियंत्रित करें। संक्रमित पौधों को हटाएं।",
        "kn": "ಇಮಿಡಾಕ್ಲೋಪ್ರಿಡ್ ಜೊತೆ ಬಿಳಿ ನೊಣಗಳನ್ನು ನಿಯಂತ್ರಿಸಿ.",
        "te": "ఇమిడాక్లోప్రిడ్‌తో తెల్ల ఈగలను నియంత్రించండి.",
        "ta": "இமிடாக்ளோபிரிட் மூலம் வெள்ளை ஈக்களை கட்டுப்படுத்தவும்.",
    },
    "tomato_yellowleaf": {
        "en": "Check for nutrient deficiency. Apply balanced NPK fertilizer.",
        "hi": "पोषक तत्वों की कमी की जांच करें। संतुलित NPK उर्वरक लगाएं।",
        "kn": "ಪೋಷಕಾಂಶ ಕೊರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ. ಸಮತೋಲಿತ NPK ಗೊಬ್ಬರ ಹಾಕಿ.",
        "te": "పోషక లోపాన్ని తనిఖీ చేయండి. సమతుల్య NPK ఎరువు వర్తించండి.",
        "ta": "ஊட்டச்சத்து குறைபாட்டை சரிபார்க்கவும். சமநிலையான NPK உரம் பயன்படுத்தவும்.",
    },
    "tomato_healthy": {
        "en": "Crop is healthy. Ensure proper staking and maintain good air circulation.",
        "hi": "फसल स्वस्थ है। उचित स्टेकिंग और अच्छा वायु संचार बनाए रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಸರಿಯಾದ ಸ್ಟೇಕಿಂಗ್ ಖಚಿತಪಡಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. సరైన స్టేకింగ్ నిర్ధారించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. சரியான ஸ்டேக்கிங் உறுதிசெய்யவும்.",
    },
    # Potato diseases - actual model labels
    "potato_early_blight": {
        "en": "Apply Chlorothalonil at 2g/L. Remove lower infected leaves. Improve air circulation.",
        "hi": "क्लोरोथालोनिल 2g/L पर लगाएं। नीचे की संक्रमित पत्तियां हटाएं।",
        "kn": "ಕ್ಲೋರೋಥಲೋನಿಲ್ 2g/L ಹಾಕಿ. ಕೆಳಗಿನ ಸೋಂಕಿತ ಎಲೆಗಳನ್ನು ತೆಗೆಯಿರಿ.",
        "te": "క్లోరోథలోనిల్ 2g/L వద్ద వర్తించండి. కింది సోకిన ఆకులను తొలగించండి.",
        "ta": "குளோரோதலோனில் 2g/L இல் பயன்படுத்தவும். கீழே பாதிக்கப்பட்ட இலைகளை அகற்றவும்.",
    },
    "potato__early_blight": {
        "en": "Apply Chlorothalonil at 2g/L. Remove lower infected leaves.",
        "hi": "क्लोरोथालोनिल 2g/L पर लगाएं।",
        "kn": "ಕ್ಲೋರೋಥಲೋನಿಲ್ 2g/L ಹಾಕಿ.",
        "te": "క్లోరోథలోనిల్ 2g/L వద్ద వర్తించండి.",
        "ta": "குளோரோதலோனில் 2g/L இல் பயன்படுத்தவும்.",
    },
    "potato_late_blight": {
        "en": "Critical: Spray Cymoxanil 8% + Mancozeb 64% WP. Destroy infected tubers.",
        "hi": "गंभीर: साइमोक्सानिल 8% + मैंकोज़ेब 64% WP छिड़कें। संक्रमित कंदों को नष्ट करें।",
        "kn": "ನಿರ್ಣಾಯಕ: ಸೈಮೋಕ್ಸಾನಿಲ್ 8% + ಮ್ಯಾಂಕೋಜೆಬ್ 64% WP ಸಿಂಪಡಿಸಿ.",
        "te": "క్లిష్టమైనది: సైమోక్సానిల్ 8% + మాంకోజెబ్ 64% WP స్ప్రే చేయండి.",
        "ta": "முக்கியமானது: சைமோக்சானில் 8% + மாங்கோசெப் 64% WP தெளிக்கவும்.",
    },
    "potato__late_blight": {
        "en": "Critical: Spray Cymoxanil 8% + Mancozeb 64% WP.",
        "hi": "गंभीर: साइमोक्सानिल 8% + मैंकोज़ेब 64% WP छिड़कें।",
        "kn": "ನಿರ್ಣಾಯಕ: ಸೈಮೋಕ್ಸಾನಿಲ್ 8% + ಮ್ಯಾಂಕೋಜೆಬ್ 64% WP ಸಿಂಪಡಿಸಿ.",
        "te": "క్లిష్టమైనది: సైమోక్సానిల్ 8% + మాంకోజెబ్ 64% WP స్ప్రే చేయండి.",
        "ta": "முக்கியமானது: சைமோக்சானில் 8% + மாங்கோசெப் 64% WP தெளிக்கவும்.",
    },
    "potato_healthy": {
        "en": "Crop is healthy. Maintain proper earthing up and monitor for pests.",
        "hi": "फसल स्वस्थ है। उचित मिट्टी चढ़ाना और कीटों की निगरानी जारी रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ಸರಿಯಾದ ಮಣ್ಣು ಹಾಕುವಿಕೆ ನಿರ್ವಹಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. సరైన ఎర్తింగ్ అప్ నిర్వహించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. சரியான மண் அணைத்தல் பராமரிக்கவும்.",
    },
    "potato__healthy": {
        "en": "Crop is healthy. Maintain proper earthing up.",
        "hi": "फसल स्वस्थ है।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ.",
        "te": "పంట ఆరోగ్యంగా ఉంది.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது.",
    },
    # Rice diseases
    "rice_leaf_aug": {
        "en": "Rice Leaf Disease: Spray Tricyclazole 75% WP at 0.6g/L OR Carbendazim 50% WP at 1g/L. Apply Urea 25kg/acre + Potash 15kg/acre. Drain excess water, maintain 5cm water level.",
        "hi": "धान की पत्ती का रोग: ट्राइसाइक्लाज़ोल 75% WP 0.6g/L या कार्बेंडाज़िम 50% WP 1g/L छिड़कें। यूरिया 25kg/एकड़ + पोटाश 15kg/एकड़ डालें। अतिरिक्त पानी निकालें।",
        "kn": "ಅಕ್ಕಿ ಎಲೆ ರೋಗ: ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75% WP 0.6g/L ಅಥವಾ ಕಾರ್ಬೆಂಡಾಜಿಮ್ 50% WP 1g/L ಸಿಂಪಡಿಸಿ. ಯೂರಿಯಾ 25kg/ಎಕರೆ + ಪೊಟ್ಯಾಷ್ 15kg/ಎಕರೆ ಹಾಕಿ. ಹೆಚ್ಚಿನ ನೀರು ಹರಿಸಿ.",
        "te": "వరి ఆకు వ్యాధి: ట్రైసైక్లజోల్ 75% WP 0.6g/L లేదా కార్బెండజిమ్ 50% WP 1g/L స్ప్రే చేయండి. యూరియా 25kg/ఎకరం + పొటాష్ 15kg/ఎకరం వేయండి. అధిక నీటిని తీసివేయండి.",
        "ta": "நெல் இலை நோய்: ட்ரைசைக்லசோல் 75% WP 0.6g/L அல்லது கார்பெண்டாசிம் 50% WP 1g/L தெளிக்கவும். யூரியா 25kg/ஏக்கர் + பொட்டாஷ் 15kg/ஏக்கர் இடவும். அதிகப்படியான நீரை வெளியேற்றவும்.",
        "pa": "ਝੋਨੇ ਦੀ ਪੱਤੀ ਦੀ ਬਿਮਾਰੀ: ਟ੍ਰਾਈਸਾਈਕਲਾਜ਼ੋਲ 75% WP 0.6g/L ਛਿੜਕਾਅ ਕਰੋ। ਯੂਰੀਆ 25kg/ਏਕੜ + ਪੋਟਾਸ਼ 15kg/ਏਕੜ ਪਾਓ।",
        "gu": "ચોખાના પાનનો રોગ: ટ્રાયસાયક્લેઝોલ 75% WP 0.6g/L છંટકાવ કરો. યુરિયા 25kg/એકર + પોટાશ 15kg/એકર આપો.",
        "mr": "तांदूळ पानाचा आजार: ट्रायसायक्लॅझोल 75% WP 0.6g/L फवारा. युरिया 25kg/एकर + पोटॅश 15kg/एकर द्या.",
        "bn": "ধান পাতার রোগ: ট্রাইসাইক্লাজোল 75% WP 0.6g/L স্প্রে করুন। ইউরিয়া 25kg/একর + পটাশ 15kg/একর দিন।",
        "or": "ଧାନ ପତ୍ର ରୋଗ: ଟ୍ରାଇସାଇକ୍ଲାଜୋଲ 75% WP 0.6g/L ସ୍ପ୍ରେ କରନ୍ତୁ। ୟୁରିଆ 25kg/ଏକର + ପୋଟାସ 15kg/ଏକର ଦିଅନ୍ତୁ।",
    },
    "rice_blast": {
        "en": "Rice Blast: Spray Tricyclazole 75% WP at 0.6g/L. Stop nitrogen fertilizer. Apply Potash 20kg/acre to strengthen plants.",
        "hi": "धान का ब्लास्ट: ट्राइसाइक्लाज़ोल 75% WP 0.6g/L छिड़कें। नाइट्रोजन रोकें। पोटाश 20kg/एकड़ डालें।",
        "kn": "ಅಕ್ಕಿ ಬ್ಲಾಸ್ಟ್: ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75% WP 0.6g/L ಸಿಂಪಡಿಸಿ. ನೈಟ್ರೋಜನ್ ನಿಲ್ಲಿಸಿ. ಪೊಟ್ಯಾಷ್ 20kg/ಎಕರೆ ಹಾಕಿ.",
        "te": "వరి బ్లాస్ట్: ట్రైసైక్లజోల్ 75% WP 0.6g/L స్ప్రే చేయండి. నైట్రోజన్ ఆపండి. పొటాష్ 20kg/ఎకరం వేయండి.",
        "ta": "நெல் பிளாஸ்ட்: ட்ரைசைக்லசோல் 75% WP 0.6g/L தெளிக்கவும். நைட்ரஜன் நிறுத்தவும். பொட்டாஷ் 20kg/ஏக்கர் இடவும்.",
    },
    "rice_brown_spot": {
        "en": "Rice Brown Spot: Spray Mancozeb 75% WP at 2.5g/L. Apply Zinc Sulphate 25kg/acre. Ensure good drainage.",
        "hi": "धान का भूरा धब्बा: मैंकोज़ेब 75% WP 2.5g/L छिड़कें। जिंक सल्फेट 25kg/एकड़ डालें।",
        "kn": "ಅಕ್ಕಿ ಕಂದು ಚುಕ್ಕೆ: ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP 2.5g/L ಸಿಂಪಡಿಸಿ. ಜಿಂಕ್ ಸಲ್ಫೇಟ್ 25kg/ಎಕರೆ ಹಾಕಿ.",
        "te": "వరి బ్రౌన్ స్పాట్: మాంకోజెబ్ 75% WP 2.5g/L స్ప్రే చేయండి. జింక్ సల్ఫేట్ 25kg/ఎకరం వేయండి.",
        "ta": "நெல் பழுப்பு புள்ளி: மாங்கோசெப் 75% WP 2.5g/L தெளிக்கவும். ஜிங்க் சல்பேட் 25kg/ஏக்கர் இடவும்.",
    },
    # Ragi diseases - actual model labels
    "ragi_blast": {
        "en": "Spray Tricyclazole 75% WP at 0.6g/L. Avoid excess nitrogen fertilizer.",
        "hi": "ट्राइसाइक्लाज़ोल 75% WP को 0.6g/L पर छिड़कें। अधिक नाइट्रोजन उर्वरक से बचें।",
        "kn": "ಟ್ರೈಸೈಕ್ಲಾಜೋಲ್ 75% WP ಅನ್ನು 0.6g/L ನಲ್ಲಿ ಸಿಂಪಡಿಸಿ.",
        "te": "ట్రైసైక్లజోల్ 75% WP ను 0.6g/L వద్ద స్ప్రే చేయండి.",
        "ta": "ட்ரைசைக்லசோல் 75% WP ஐ 0.6g/L இல் தெளிக்கவும்.",
    },
    "ragi_rust": {
        "en": "Apply Mancozeb or Propiconazole. Ensure proper plant spacing.",
        "hi": "मैंकोज़ेब या प्रोपिकोनाज़ोल लगाएं।",
        "kn": "ಮ್ಯಾಂಕೋಜೆಬ್ ಅಥವಾ ಪ್ರೋಪಿಕೋನಾಜೋಲ್ ಹಾಕಿ.",
        "te": "మాంకోజెబ్ లేదా ప్రొపికోనజోల్ వర్తించండి.",
        "ta": "மாங்கோசெப் அல்லது ப்ரோபிகோனசோல் பயன்படுத்தவும்.",
    },
    "ragi_healthy": {
        "en": "Crop is healthy. Continue regular monitoring.",
        "hi": "फसल स्वस्थ है। नियमित निगरानी जारी रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా పర్యవేక్షణ కొనసాగించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பை தொடரவும்.",
    },
    # Sugarcane diseases - actual model labels
    "sugarcane_redrot": {
        "en": "Remove and burn infected stalks. Treat setts with Carbendazim before planting.",
        "hi": "संक्रमित डंठलों को हटाकर जलाएं। रोपण से पहले सेट्स को कार्बेन्डाजिम से उपचारित करें।",
        "kn": "ಸೋಂಕಿತ ಕಾಂಡಗಳನ್ನು ತೆಗೆದು ಸುಡಿ. ನಾಟಿ ಮಾಡುವ ಮೊದಲು ಕಾರ್ಬೆಂಡಾಜಿಮ್ ಚಿಕಿತ್ಸೆ ನೀಡಿ.",
        "te": "సోకిన కాండాలను తొలగించి కాల్చండి. నాటే ముందు కార్బెండజిమ్‌తో చికిత్స చేయండి.",
        "ta": "பாதிக்கப்பட்ட தண்டுகளை அகற்றி எரிக்கவும். நடவு செய்வதற்கு முன் கார்பெண்டாசிம் சிகிச்சை அளிக்கவும்.",
    },
    "sugarcane_rust": {
        "en": "Apply Propiconazole or Mancozeb. Use resistant varieties.",
        "hi": "प्रोपिकोनाज़ोल या मैंकोज़ेब लगाएं।",
        "kn": "ಪ್ರೋಪಿಕೋನಾಜೋಲ್ ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ ಹಾಕಿ.",
        "te": "ప్రొపికోనజోల్ లేదా మాంకోజెబ్ వర్తించండి.",
        "ta": "ப்ரோபிகோனசோல் அல்லது மாங்கோசெப் பயன்படுத்தவும்.",
    },
    "sugarcane_mosaic": {
        "en": "Remove infected plants. Use virus-free planting material. Control aphid vectors.",
        "hi": "संक्रमित पौधों को हटाएं। वायरस-मुक्त रोपण सामग्री का उपयोग करें।",
        "kn": "ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ತೆಗೆದುಹಾಕಿ. ವೈರಸ್-ಮುಕ್ತ ನಾಟಿ ಸಾಮಗ್ರಿ ಬಳಸಿ.",
        "te": "సోకిన మొక్కలను తొలగించండి. వైరస్-రహిత నాటు పదార్థం ఉపయోగించండి.",
        "ta": "பாதிக்கப்பட்ட செடிகளை அகற்றவும். வைரஸ் இல்லாத நடவு பொருட்களை பயன்படுத்தவும்.",
    },
    "sugarcane_yellow": {
        "en": "Check for nutrient deficiency. Apply micronutrients. Ensure proper drainage.",
        "hi": "पोषक तत्वों की कमी की जांच करें। सूक्ष्म पोषक तत्व लगाएं।",
        "kn": "ಪೋಷಕಾಂಶ ಕೊರತೆಯನ್ನು ಪರಿಶೀಲಿಸಿ. ಸೂಕ್ಷ್ಮ ಪೋಷಕಾಂಶಗಳನ್ನು ಹಾಕಿ.",
        "te": "పోషక లోపాన్ని తనిఖీ చేయండి. సూక్ష్మ పోషకాలు వర్తించండి.",
        "ta": "ஊட்டச்சத்து குறைபாட்டை சரிபார்க்கவும். நுண்ணூட்டச்சத்துக்களை பயன்படுத்தவும்.",
    },
    "sugarcane_healthy": {
        "en": "Crop is healthy. Continue regular monitoring and maintain proper irrigation.",
        "hi": "फसल स्वस्थ है। नियमित निगरानी जारी रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా పర్యవేక్షణ కొనసాగించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பை தொடரவும்.",
    },
    # Wheat diseases - actual model labels
    "wheat_yellow_rust": {
        "en": "Apply Propiconazole 25% EC at 0.1%. Use resistant varieties for next season.",
        "hi": "प्रोपिकोनाज़ोल 25% EC को 0.1% पर लगाएं। अगले मौसम के लिए प्रतिरोधी किस्मों का उपयोग करें।",
        "kn": "ಪ್ರೋಪಿಕೋನಾಜೋಲ್ 25% EC ಅನ್ನು 0.1% ನಲ್ಲಿ ಹಾಕಿ.",
        "te": "ప్రొపికోనజోల్ 25% EC ను 0.1% వద్ద వర్తించండి.",
        "ta": "ப்ரோபிகோனசோல் 25% EC ஐ 0.1% இல் பயன்படுத்தவும்.",
    },
    "wheat_healthy": {
        "en": "Crop is healthy. Continue regular monitoring.",
        "hi": "फसल स्वस्थ है। नियमित निगरानी जारी रखें।",
        "kn": "ಬೆಳೆ ಆರೋಗ್ಯಕರವಾಗಿದೆ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
        "te": "పంట ఆరోగ్యంగా ఉంది. క్రమం తప్పకుండా పర్యవేక్షణ కొనసాగించండి.",
        "ta": "பயிர் ஆரோக்கியமாக உள்ளது. வழக்கமான கண்காணிப்பை தொடரவும்.",
    },
    # Uncertain/unknown detection
    "uncertain_detection": {
        "en": "⚠️ Unable to clearly identify the crop disease. Please upload a clearer photo of the affected leaf. Tips: Good lighting, close-up of leaf, avoid shadows.",
        "hi": "⚠️ फसल की बीमारी स्पष्ट रूप से पहचान नहीं हो पाई। कृपया प्रभावित पत्ती की स्पष्ट फोटो अपलोड करें।",
        "kn": "⚠️ ಬೆಳೆ ರೋಗವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಗುರುತಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಪ್ರಭಾವಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "te": "⚠️ పంట వ్యాధిని స్పష్టంగా గుర్తించలేకపోయాము. దయచేసి ప్రభావిత ఆకు యొక్క స్పష్టమైన ఫోటో అప్‌లోడ్ చేయండి.",
        "ta": "⚠️ பயிர் நோயை தெளிவாக அடையாளம் காண முடியவில்லை. பாதிக்கப்பட்ட இலையின் தெளிவான புகைப்படத்தை பதிவேற்றவும்.",
    },
    "unknown": {
        "en": "Please upload a clear photo of the crop leaf to get disease diagnosis and treatment advice.",
        "hi": "रोग निदान और उपचार सलाह प्राप्त करने के लिए कृपया फसल की पत्ती की स्पष्ट फोटो अपलोड करें।",
        "kn": "ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಸಲಹೆ ಪಡೆಯಲು ದಯವಿಟ್ಟು ಬೆಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
        "te": "వ్యాధి నిర్ధారణ మరియు చికిత్స సలహా పొందడానికి దయచేసి పంట ఆకు యొక్క స్పష్టమైన ఫోటో అప్‌లోడ్ చేయండి.",
        "ta": "நோய் கண்டறிதல் மற்றும் சிகிச்சை ஆலோசனையைப் பெற பயிர் இலையின் தெளிவான புகைப்படத்தை பதிவேற்றவும்.",
    },
}


def get_device() -> torch.device:
    """Get optimal device for inference."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class VisionEngine:
    """
    Production Vision Engine using ViT with GPU acceleration.
    Loads fine-tuned model from checkpoint or falls back to base ViT.
    """

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = get_device()

        logger.info(f"Loading Vision model from: {model_name_or_path}")
        logger.info(f"Using device: {self.device}")

        is_local = Path(model_name_or_path).exists() and _has_model_weights(Path(model_name_or_path))

        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=is_local,
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            local_files_only=is_local,
            ignore_mismatched_sizes=True,
        )

        # Move to GPU if available
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get number of labels and id2label from model config
        self.num_labels = self.model.config.num_labels
        self.id2label = self.model.config.id2label
        
        logger.info(f"Model loaded with {self.num_labels} classes on {self.device}")
        logger.info(f"Label mapping: {list(self.id2label.values())[:5]}...")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """
        Run inference on a single image.
        Returns prediction with confidence and disease info.
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        # Use model's own id2label mapping (try both int and string keys)
        label = self.id2label.get(idx) or self.id2label.get(str(idx)) or f"unknown_class_{idx}"

        return {
            "label": label,
            "confidence": confidence,
            "class_index": idx,
            "all_probs": probs.tolist()[:10],  # Top 10 for debugging
            "device": str(self.device),
            "source": "vit_gpu" if "cuda" in str(self.device) else "vit_cpu",
        }

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Batch inference for multiple images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=-1).cpu().numpy()

        results = []
        for i, p in enumerate(probs):
            idx = int(np.argmax(p))
            label = self.id2label.get(str(idx), f"unknown_class_{idx}")
            results.append({
                "label": label,
                "confidence": float(p[idx]),
                "source": "vit_batch",
            })

        return results


# Global engine instances
_ENGINE: Optional[VisionEngine] = None
_ENGINE_KEY: str = ""
_ONNX_ENGINE: Optional[ONNXVisionEngine] = None


def _has_model_weights(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint directory has actual model weight files."""
    weight_files = ["model.safetensors", "pytorch_model.bin", "model.bin"]
    for f in weight_files:
        fpath = checkpoint_dir / f
        if fpath.exists() and fpath.stat().st_size > 1000:  # >1KB = real file
            return True
    return False


def _resolve_model_path(model_dir: Optional[str]) -> str:
    """Resolve the best model path to use."""
    # Priority 1: Explicitly provided path
    if model_dir and Path(model_dir).exists() and _has_model_weights(Path(model_dir)):
        return model_dir

    # Priority 2: Environment variable
    env_dir = os.getenv("AGRIBLOOM_VISION_MODEL_DIR", "").strip()
    if env_dir and Path(env_dir).exists() and _has_model_weights(Path(env_dir)):
        return env_dir

    # Priority 3: Production fine-tuned checkpoint (vit_crop_disease)
    prod_checkpoint = Path("models/checkpoints/vit_crop_disease")
    if prod_checkpoint.exists() and _has_model_weights(prod_checkpoint):
        return str(prod_checkpoint)

    # Priority 4: Development checkpoint (day4_vit)
    dev_checkpoint = Path("models/checkpoints/day4_vit")
    if dev_checkpoint.exists() and _has_model_weights(dev_checkpoint):
        return str(dev_checkpoint)

    # Priority 5: PlantVillage plant disease model from HuggingFace Hub
    logger.info("No local model weights found — using PlantVillage ViT from HF Hub")
    return "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"


def _get_engine(model_dir: Optional[str] = None) -> VisionEngine:
    """Get or create the vision engine singleton."""
    global _ENGINE, _ENGINE_KEY

    key = _resolve_model_path(model_dir)
    if _ENGINE is None or _ENGINE_KEY != key:
        _ENGINE = VisionEngine(model_name_or_path=key)
        _ENGINE_KEY = key

    return _ENGINE


def _get_onnx_engine() -> ONNXVisionEngine:
    """Get or create ONNX engine singleton."""
    global _ONNX_ENGINE

    if _ONNX_ENGINE is None:
        onnx_path = Path("models/vision/vit_base_patch16_224.onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        _ONNX_ENGINE = ONNXVisionEngine(str(onnx_path))

    return _ONNX_ENGINE


def get_treatment(disease_label: str, lang: str = "en") -> str:
    """Get treatment recommendation for a disease."""
    treatments = DISEASE_TREATMENTS.get(disease_label, {})
    return treatments.get(lang, treatments.get("en", "Consult local agricultural officer for treatment guidance."))


def run_vision(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: Run vision inference on crop image.

    Input state keys:
        - image: PIL Image (required for inference)
        - offline: bool (use ONNX if True)
        - model_dir: optional model path override
        - lang: language code for treatment

    Output state keys:
        - disease_prediction: dict with label, confidence, source
        - crop_type: detected crop type
        - treatment: recommended treatment
        - status: "vision_complete"
    """
    image = state.get("image")

    # Handle missing image
    if image is None:
        return {
            **state,
            "disease_prediction": {
                "label": "unknown",
                "confidence": 0.0,
                "source": "no_image",
            },
            "crop_type": "unknown",
            "status": "vision_skipped",
        }

    offline = bool(state.get("offline", False))
    lang = state.get("lang", "en")

    try:
        if offline:
            # Use ONNX for offline inference
            engine = _get_onnx_engine()
            pred = engine.get_prediction(image, TARGET_CLASSES)
        else:
            # Use PyTorch ViT for online inference (with GPU)
            engine = _get_engine(state.get("model_dir"))
            pred = engine.predict(image)

    except Exception as e:
        logger.error(f"Vision inference failed: {e}")
        # Return error state instead of fake prediction
        return {
            **state,
            "disease_prediction": {
                "label": "error",
                "confidence": 0.0,
                "source": "inference_error",
                "error": str(e),
            },
            "crop_type": "unknown",
            "status": "vision_error",
        }

    # Extract crop type from label
    label = pred.get("label", "unknown")
    confidence = pred.get("confidence", 0.0)
    
    # Keep original label for LLM context even if uncertain
    pred["original_label"] = label
    
    # Only mark as uncertain if confidence is very low
    if confidence < 0.15:
        label = "uncertain_detection"
        pred["label"] = label
    
    crop_type = label.split("_")[0] if "_" in label else "unknown"

    # Get treatment recommendation
    treatment = get_treatment(label, lang)

    return {
        **state,
        "disease_prediction": pred,
        "crop_type": crop_type,
        "treatment": treatment,
        "status": "vision_complete",
    }


# Export for direct use
__all__ = [
    "run_vision",
    "VisionEngine",
    "TARGET_CLASSES",
    "CROP_CLASSES",
    "get_treatment",
]
