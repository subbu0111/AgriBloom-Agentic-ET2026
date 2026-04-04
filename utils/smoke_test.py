from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import run_pipeline


def run_smoke_test() -> None:
    sample_image = Image.new("RGB", (256, 256), color=(40, 140, 60))

    result = run_pipeline(
        image=sample_image,
        image_path="maize_healthy_demo.jpg",
        user_text="Leaves are yellowing, suggest safe treatment",
        user_language="hi",
        offline=True,
        lat=17.385,
        lon=78.4867,
    )

    response = result.get("final_response", "")
    voice_path = result.get("voice_output_path", "")
    audit_pdf = result.get("audit_pdf_path", "")

    assert response, "Expected non-empty advisory text"
    assert Path(voice_path).exists(), f"Voice output missing: {voice_path}"
    assert Path(audit_pdf).exists(), f"Audit PDF missing: {audit_pdf}"

    print("Smoke test passed.")
    print(f"Status: {result.get('status')}")
    print(f"Prediction: {result.get('disease_prediction')}")
    print(f"Audio: {voice_path}")
    print(f"Audit: {audit_pdf}")


if __name__ == "__main__":
    run_smoke_test()
