"""
AgriBloom Agentic - Hugging Face Spaces Entry Point
"""
import sys
import logging
from pathlib import Path

# =====================================================================
# Fix gradio_client bug: schema parsing crashes on bool values
# =====================================================================
try:
    import gradio_client.utils as _gc_utils

    # Patch get_type(schema) — 1 arg
    _orig_get_type = _gc_utils.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "bool"
        return _orig_get_type(schema)
    _gc_utils.get_type = _safe_get_type

    # Patch _json_schema_to_python_type(schema, defs) — 2 args
    _orig_private = _gc_utils._json_schema_to_python_type
    def _safe_private(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _orig_private(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_private

    # Patch json_schema_to_python_type(schema) — 1 arg only
    _orig_public = _gc_utils.json_schema_to_python_type
    def _safe_public(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        if defs is not None:
            return _orig_public(schema)
        return _orig_public(schema)
    _gc_utils.json_schema_to_python_type = _safe_public

    print("[PATCH] Applied gradio_client schema fixes")
except Exception as e:
    print(f"[PATCH] Could not patch: {e}")
# =====================================================================

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("agribloom")

logger.info("=" * 60)
logger.info("AgriBloom Agentic - Starting on Hugging Face Spaces")
logger.info("=" * 60)

try:
    from main import run_pipeline
    logger.info("Pipeline loaded successfully")
except Exception as e:
    logger.error(f"Failed to load pipeline: {e}")
    def run_pipeline(**kwargs):
        return {
            "final_response": f"Pipeline error: {e}",
            "status": "error",
            "disease_prediction": {"confidence": 0.0, "label": "unknown"},
        }

from ui.app import launch_app
launch_app(run_pipeline)
