---
title: AgriBloom Agentic ET2026
emoji: 🌾
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
python_version: "3.11"
pinned: true
license: mit
short_description: AI crop disease detection for Indian farmers
---

# AgriBloom Agentic 🌾

**Agricultural Advisory Agents — AI-Powered Crop Disease Detection & Advisory**

AI-powered crop disease detection and advisory system for Indian farmers.
Supports **10 Indian languages**, powered by a multi-agent pipeline.

## Features
- 🔬 **Disease Detection**: 54 crop disease classes (Maize, Tomato, Potato, Rice, Wheat, Ragi, Sugarcane)
- 🌐 **10 Languages**: English, Hindi, Kannada, Telugu, Tamil, Punjabi, Gujarati, Marathi, Bengali, Odia
- 🌤️ **Live Weather**: Real-time weather via Open-Meteo API (free, no key needed)
- 💰 **Market Prices**: MSP + Mandi price advisory
- 🎙️ **Voice Output**: Text-to-speech in local languages
- 📄 **PDF Report**: Downloadable compliance audit report
- 📈 **Bloom Simulator**: Crop health prediction chart
- 📶 **Offline Mode**: Works without internet

## How to Use
1. Select your **language**
2. Upload a **crop leaf photo**
3. Select your **State and District**
4. Click **GET ADVICE**

## Tech Stack
- Multi-agent pipeline: LangGraph (5 agents)
- Vision: ViT (Vision Transformer) from HuggingFace
- UI: Gradio 5.x
- Voice: gTTS
- Weather: Open-Meteo (free, no API key)
