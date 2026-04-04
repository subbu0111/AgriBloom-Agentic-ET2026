"""
AgriBloom Agentic - Premium Gradio UI
Voice AI for Indian Farmers
Beautiful dark green theme with animations and Tailwind styling
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import gradio as gr
from PIL import Image

logger = logging.getLogger(__name__)

# Whisper model for voice transcription
_WHISPER_MODEL = None


def _get_whisper_model():
    """Lazy load Whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
            logger.info("Loading Whisper base model...")
            _WHISPER_MODEL = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return None
    return _WHISPER_MODEL


LANGUAGE_MAP = {
    "English": "en",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Hindi (हिंदी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Gujarati (ગુજરાતી)": "gu",
    "Marathi (मराठी)": "mr",
    "Bengali (বাংলা)": "bn",
    "Odia (ଓଡ଼ିଆ)": "or",
}

LANGUAGE_CODES = {
    "en": "en",
    "kn": "kn",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "pa": "pa",
    "gu": "gu",
    "mr": "mr",
    "bn": "bn",
    "or": "or",
}

# Indian States and Districts for easy selection
INDIAN_LOCATIONS = {
    "Karnataka": ["Bengaluru", "Mysuru", "Davangere", "Belgaum", "Hubli", "Mangaluru", "Tumkur", "Shimoga"],
    "Andhra Pradesh": ["Vijayawada", "Guntur", "Visakhapatnam", "Tirupati", "Kurnool", "Nellore", "Kadapa"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Ooty", "Erode", "Thanjavur"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam", "Nalgonda"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Kolhapur", "Solapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Junagadh"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda", "Mohali", "Ferozepur"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer", "Alwar"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Rewa"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Allahabad", "Meerut", "Gorakhpur"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga", "Purnia"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri", "Kharagpur"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur", "Puri"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam", "Palakkad"],
}

# District coordinates for weather lookup
DISTRICT_COORDS = {
    "Bengaluru": (12.9716, 77.5946), "Mysuru": (12.2958, 76.6394), "Davangere": (14.4644, 75.9218),
    "Belgaum": (15.8497, 74.4977), "Hubli": (15.3647, 75.1240), "Mangaluru": (12.9141, 74.8560),
    "Vijayawada": (16.5062, 80.6480), "Guntur": (16.3067, 80.4365), "Visakhapatnam": (17.6868, 83.2185),
    "Hyderabad": (17.3850, 78.4867), "Warangal": (17.9689, 79.5941), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Madurai": (9.9252, 78.1198), "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882), "Ahmedabad": (23.0225, 72.5714), "Surat": (21.1702, 72.8311),
    "Ludhiana": (30.9010, 75.8573), "Amritsar": (31.6340, 74.8723), "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462), "Patna": (25.5941, 85.1376), "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245), "Kochi": (9.9312, 76.2673),
}

def _get_coords(district: str) -> tuple:
    """Get coordinates for a district."""
    return DISTRICT_COORDS.get(district, (14.4644, 75.9218))  # Default: Davangere

def _lang_name_to_code(selected: str) -> str:
    """Convert language name to code."""
    return LANGUAGE_MAP.get(selected, "en")


def _transcribe_audio(audio_path: str | None, text: str, lang_code: str) -> str:
    """Transcribe audio if no text provided."""
    if text and text.strip():
        return text.strip()

    if not audio_path:
        return ""

    model = _get_whisper_model()
    if model is None:
        return "[Whisper not available]"

    try:
        result = model.transcribe(audio_path, language=lang_code)
        transcript = result.get("text", "").strip()
        logger.info(f"Transcribed audio: {transcript[:100]}...")
        return transcript
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "[Transcription error]"


def _load_css() -> str:
    """Load custom CSS."""
    css_file = Path("ui/static/styles.css")
    if css_file.exists():
        return css_file.read_text(encoding="utf-8")
    return ""


def _get_tailwind_head() -> str:
    """Get Tailwind CSS configuration."""
    return """
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              agri: {
                50: '#f0fdf4',
                100: '#dcfce7',
                200: '#bbf7d0',
                300: '#86efac',
                400: '#4ade80',
                500: '#22c55e',
                600: '#16a34a',
                700: '#166534',
                800: '#14532d',
                900: '#052e16',
              }
            },
            animation: {
              'wave': 'wave 1.5s ease-in-out infinite',
              'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              'float': 'float 3s ease-in-out infinite',
            },
            keyframes: {
              wave: {
                '0%, 100%': { transform: 'scaleY(0.5)' },
                '50%': { transform: 'scaleY(1)' },
              },
              float: {
                '0%, 100%': { transform: 'translateY(0px)' },
                '50%': { transform: 'translateY(-10px)' },
              }
            }
          }
        }
      }
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      body { font-family: 'Inter', sans-serif; }
      .gradio-container { max-width: 1400px !important; }
    </style>
    """


def create_hero_html() -> str:
    """Create hero banner HTML."""
    return """
    <div class="relative overflow-hidden bg-gradient-to-br from-agri-900 via-agri-800 to-agri-700 rounded-2xl p-8 mb-6 shadow-2xl">
        <!-- Animated background pattern -->
        <div class="absolute inset-0 opacity-10">
            <div class="absolute top-0 left-0 w-40 h-40 bg-agri-400 rounded-full blur-3xl animate-pulse-slow"></div>
            <div class="absolute bottom-0 right-0 w-60 h-60 bg-agri-300 rounded-full blur-3xl animate-pulse-slow" style="animation-delay: 1s;"></div>
        </div>

        <div class="relative z-10 text-center">
            <div class="flex justify-center mb-4">
                <span class="text-5xl animate-float">🌾</span>
            </div>
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-3 tracking-tight">
                AgriBloom Agentic
            </h1>
            <p class="text-xl text-agri-200 mb-6">
                Voice AI for Indian Farmers | Multi-Agent Agricultural Advisory
            </p>

            <!-- Voice waveform animation -->
            <div class="flex justify-center items-end space-x-1 h-8 mb-4">
                <div class="w-1 bg-agri-400 rounded-full animate-wave" style="height: 50%; animation-delay: 0s;"></div>
                <div class="w-1 bg-agri-300 rounded-full animate-wave" style="height: 70%; animation-delay: 0.1s;"></div>
                <div class="w-1 bg-agri-400 rounded-full animate-wave" style="height: 100%; animation-delay: 0.2s;"></div>
                <div class="w-1 bg-agri-300 rounded-full animate-wave" style="height: 80%; animation-delay: 0.3s;"></div>
                <div class="w-1 bg-agri-400 rounded-full animate-wave" style="height: 60%; animation-delay: 0.4s;"></div>
                <div class="w-1 bg-agri-300 rounded-full animate-wave" style="height: 90%; animation-delay: 0.5s;"></div>
                <div class="w-1 bg-agri-400 rounded-full animate-wave" style="height: 40%; animation-delay: 0.6s;"></div>
            </div>

            <div class="flex flex-wrap justify-center gap-3">
                <span class="px-3 py-1 bg-agri-600/50 text-agri-100 rounded-full text-sm">🔬 Vision AI</span>
                <span class="px-3 py-1 bg-agri-600/50 text-agri-100 rounded-full text-sm">🛡️ Compliance</span>
                <span class="px-3 py-1 bg-agri-600/50 text-agri-100 rounded-full text-sm">🌱 Bloom Simulator</span>
                <span class="px-3 py-1 bg-agri-600/50 text-agri-100 rounded-full text-sm">🎙️ Voice I/O</span>
                <span class="px-3 py-1 bg-agri-600/50 text-agri-100 rounded-full text-sm">📶 Offline Mode</span>
            </div>
        </div>
    </div>
    """


def create_status_html(status: str, is_error: bool = False) -> str:
    """Create status badge HTML."""
    color = "red" if is_error else "green"
    icon = "❌" if is_error else "✅"
    return f"""
    <div class="flex items-center gap-2 p-3 bg-{color}-50 border border-{color}-200 rounded-lg">
        <span>{icon}</span>
        <span class="text-{color}-700 font-medium">{status}</span>
    </div>
    """


def launch_app(run_pipeline: Callable[..., dict[str, Any]]) -> None:
    """
    Launch the AgriBloom Gradio application.

    Args:
        run_pipeline: Function to run the agent pipeline
    """
    custom_css = _load_css()

    # Additional custom CSS
    additional_css = """
    .ab-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    .ab-card-dark {
        background: linear-gradient(135deg, #14532d 0%, #166534 100%);
        border: 1px solid #22c55e;
        color: white;
    }
    .ab-btn-primary {
        background: linear-gradient(135deg, #166534 0%, #14532d 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .ab-btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(22, 101, 52, 0.3) !important;
    }
    .ab-demo-btn {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%) !important;
    }
    """

    with gr.Blocks(
        title="AgriBloom Agentic - Voice AI for Indian Farmers",
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate",
        ),
    ) as demo:
        # Hero Banner
        gr.HTML(create_hero_html())

        # Main UI - Simple and Farmer-Friendly
        with gr.Row(equal_height=False):
            # Left Panel - Inputs
            with gr.Column(scale=5):
                gr.Markdown("## 📸 Upload Your Crop Photo / మీ పంట ఫోటో అప్‌లోడ్ చేయండి")

                with gr.Group(elem_classes=["ab-card"]):
                    # LANGUAGE SELECTOR - PROMINENT AT TOP
                    language = gr.Dropdown(
                        choices=list(LANGUAGE_MAP.keys()),
                        value="English",
                        label="🌐 Select Your Language / మీ భాషను ఎంచుకోండి",
                        info="Choose your preferred language for advice",
                    )
                    
                    image_input = gr.Image(
                        type="pil",
                        label="📷 Take Photo or Upload / ఫోటో తీయండి",
                        height=280,
                        sources=["upload", "webcam"],
                    )

                    text_input = gr.Textbox(
                        lines=2,
                        label="📝 Describe Problem (Optional) / సమస్య వివరించండి",
                        placeholder="Example: My tomato leaves are turning yellow...",
                    )

                # Location - Simple District Selection
                gr.Markdown("### 📍 Your Location / మీ ప్రదేశం")
                with gr.Group(elem_classes=["ab-card"]):
                    with gr.Row():
                        state_input = gr.Dropdown(
                            choices=list(INDIAN_LOCATIONS.keys()),
                            value="Telangana",
                            label="State / రాష్ట్రం",
                        )
                        district_input = gr.Dropdown(
                            choices=INDIAN_LOCATIONS["Telangana"],
                            value="Hyderabad",
                            label="District / జిల్లా",
                        )
                    
                    offline_mode = gr.Checkbox(
                        value=False,
                        label="📶 Offline Mode (No Internet) / ఆఫ్‌లైన్ మోడ్",
                    )

                submit_btn = gr.Button(
                    "🌾 GET ADVICE / సలహా పొందండి",
                    variant="primary",
                    size="lg",
                    elem_classes=["ab-btn-primary"],
                )

            # Right Panel - Results
            with gr.Column(scale=5):
                gr.Markdown("## 📋 Your Advisory / మీ సలహా")

                with gr.Group(elem_classes=["ab-card"]):
                    response_output = gr.Textbox(
                        lines=12,
                        label="🌾 Disease & Treatment / వ్యాధి మరియు చికిత్స",
                        show_label=True,
                    )

                    voice_output = gr.Audio(
                        label="🔊 Listen to Advice / సలహా వినండి",
                        type="filepath",
                    )

                gr.Markdown("### 📈 Crop Health Prediction")
                with gr.Group(elem_classes=["ab-card"]):
                    bloom_plot = gr.Plot(
                        label="Before → After Treatment",
                    )

                with gr.Row():
                    audit_file = gr.File(label="📄 Download Report PDF")
                    
                status_output = gr.Markdown("*Ready! Upload a crop photo to get advice.*")

        # Footer
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 12px; text-align: center;">
            <p style="font-weight: bold; color: #166534;">🌾 AgriBloom Agentic - AI Advisory for Indian Farmers</p>
            <p style="color: #15803d; font-size: 14px;">Supports: Maize, Tomato, Potato, Rice, Wheat, Ragi, Sugarcane</p>
        </div>
        """)

        # Event handlers
        def process_query(
            image,
            text_value,
            language_name,
            state,
            district,
            offline,
        ):
            """Process user query through the pipeline."""
            lang_code = _lang_name_to_code(language_name)
            
            # Get coordinates from district
            lat, lon = _get_coords(district)

            # Get image path if available
            image_path = ""
            if hasattr(image, "filename"):
                image_path = str(image.filename)

            try:
                result = run_pipeline(
                    image=image,
                    image_path=image_path,
                    user_text=text_value or "",
                    user_language=lang_code,
                    lang=lang_code,
                    offline=bool(offline),
                    lat=float(lat),
                    lon=float(lon),
                    user_state=state or "Telangana",
                    user_district=district or "Hyderabad",
                )

                status = f"✅ **Analysis Complete!** | Disease detected with {result.get('disease_prediction', {}).get('confidence', 0)*100:.0f}% confidence"

                return (
                    result.get("final_response", "No response generated."),
                    result.get("voice_output_path"),
                    result.get("bloom_figure"),
                    result.get("audit_pdf_path"),
                    status,
                )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                return (
                    f"Error processing request: {str(e)}",
                    None,
                    None,
                    None,
                    f"❌ **Error**: {str(e)}",
                )

        # Connect event handlers - simplified
        submit_btn.click(
            fn=process_query,
            inputs=[
                image_input,
                text_input,
                language,
                state_input,
                district_input,
                offline_mode,
            ],
            outputs=[
                response_output,
                voice_output,
                bloom_plot,
                audit_file,
                status_output,
            ],
            show_progress="full",
        )
        
        # Update districts when state changes
        def update_districts(state):
            districts = INDIAN_LOCATIONS.get(state, ["Select District"])
            return gr.Dropdown(choices=districts, value=districts[0])
        
        state_input.change(
            fn=update_districts,
            inputs=[state_input],
            outputs=[district_input],
        )

    # Launch server
    logger.info("Launching AgriBloom Agentic UI on http://0.0.0.0:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )


if __name__ == "__main__":
    # Test launch with mock pipeline
    def mock_pipeline(**kwargs):
        return {
            "final_response": "Test response",
            "status": "output_complete",
        }

    launch_app(mock_pipeline)
