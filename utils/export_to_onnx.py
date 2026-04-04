#!/usr/bin/env python3
"""Export ViT model to ONNX for inference"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

def export_to_onnx(model_dir: str = "models/checkpoints/vit_crop_disease"):
    """Export trained model to ONNX format"""
    model_path = Path(model_dir)
    output_path = model_path / "model.onnx"

    print(f"[EXPORT] Loading model from {model_path}")
    model = AutoModelForImageClassification.from_pretrained(str(model_path))
    processor = AutoImageProcessor.from_pretrained(str(model_path))

    # Load label map
    with open(model_path / "label_map.json") as f:
        label_data = json.load(f)

    num_classes = len(label_data["labels"])
    print(f"[MODEL] Classes: {num_classes} | Image size: {processor.size}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, processor.size['height'], processor.size['width'])

    # Export to ONNX
    print(f"[EXPORT] Exporting to {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✓ ONNX exported: {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")

    # Save config for inference
    config = {
        "model_path": str(output_path),
        "num_classes": num_classes,
        "labels": label_data["labels"],
        "image_size": {"height": processor.size['height'], "width": processor.size['width']},
        "accuracy": 0.9064
    }

    config_path = model_path / "onnx_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved: {config_path}")

if __name__ == "__main__":
    export_to_onnx()
