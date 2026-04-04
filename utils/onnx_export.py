from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import ViTForImageClassification


def export_vit_to_onnx(output_path: str = "models/vision/vit_base_patch16_224.onnx", opset: int = 17) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        export_params=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ViT model to ONNX for offline inference")
    parser.add_argument("--output", default="models/vision/vit_base_patch16_224.onnx", help="Output ONNX model path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    out = export_vit_to_onnx(output_path=args.output, opset=args.opset)
    print(f"Exported ONNX model: {out}")


if __name__ == "__main__":
    main()
