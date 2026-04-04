"""
Generate train/val/test manifests from all crop disease datasets.
Creates CSV files for training the Vision Transformer model.
"""
from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path
from typing import TypedDict

import pandas as pd

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ImageRecord(TypedDict):
    image_path: str
    label: str
    crop: str
    disease: str


def normalize_label(crop: str, disease: str) -> str:
    """Normalize label to standard format."""
    c = crop.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    d = disease.lower().strip().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")

    # Handle common naming variations
    c = c.replace("corn_maize", "maize").replace("corn", "maize")
    c = c.replace("paddy", "rice").replace("millei", "ragi")
    c = c.replace("finger_millet", "ragi").replace("nachni", "ragi")

    # Clean up disease names
    d = d.replace("gray_leaf_spot", "gray_spot").replace("grey_leaf_spot", "gray_spot")
    d = d.replace("powdery_mildew", "powdery_mildew").replace("northern_leaf_blight", "leaf_blight")

    if d.startswith(c + "_"):
        return d
    return f"{c}_{d}" if d and d != "unknown" else f"{c}_healthy"


def extract_from_kaggle_format(dataset_root: Path) -> list[ImageRecord]:
    """
    Extract from Kaggle format: Crop___Disease/images
    Used by wheat2 and other Kaggle datasets.
    """
    records = []
    if not dataset_root.exists():
        return records

    for class_dir in sorted(dataset_root.rglob("*")):
        if not class_dir.is_dir() or "__" not in class_dir.name:
            continue

        parts = class_dir.name.split("___")
        if len(parts) != 2:
            continue

        crop = parts[0].strip()
        disease = parts[1].strip()

        label = normalize_label(crop, disease)

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(
                    ImageRecord(
                        image_path=str(img_path),
                        label=label,
                        crop=crop.lower(),
                        disease=disease.lower(),
                    )
                )

    return records


def extract_from_plantvillage_format(dataset_root: Path) -> list[ImageRecord]:
    """
    Extract from PlantVillage format with nested PlantVillage directories.
    """
    records = []
    if not dataset_root.exists():
        return records

    # Look for class directories (skip the nested PlantVillage folder)
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        # Skip the nested PlantVillage directory, go deeper
        if class_dir.name == "PlantVillage":
            return extract_from_plantvillage_format(class_dir)

        class_name = class_dir.name

        # Parse PlantVillage style: Crop__disease or Crop_disease
        if "__" in class_name:
            parts = class_name.split("__")
            crop = parts[0].strip()
            disease = parts[1].strip()
        elif "_" in class_name:
            parts = class_name.split("_")
            crop = parts[0].strip()
            disease = "_".join(parts[1:]).strip() if len(parts) > 1 else "healthy"
        else:
            continue

        label = normalize_label(crop, disease)

        # Get all images in subdirectories
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(
                    ImageRecord(
                        image_path=str(img_path),
                        label=label,
                        crop=crop.lower(),
                        disease=disease.lower(),
                    )
                )

    return records


def extract_from_simple_crop_format(dataset_root: Path, crop_name: str) -> list[ImageRecord]:
    """
    Extract from simple format: disease_class/images
    Used by ragi, sugarcane, etc.
    """
    records = []
    if not dataset_root.exists():
        return records

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        disease = class_dir.name.strip()
        label = normalize_label(crop_name, disease)

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(
                    ImageRecord(
                        image_path=str(img_path),
                        label=label,
                        crop=crop_name.lower(),
                        disease=disease.lower(),
                    )
                )

    return records


def main() -> None:
    """Generate manifests from all datasets."""
    print("=" * 60)
    print("AgriBloom Manifest Generation (Day 7)")
    print("=" * 60)

    data_root = ROOT / "data" / "raw"
    manifests_dir = ROOT / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Collect images from all datasets with proper parsing
    all_records: list[ImageRecord] = []

    print("\n[INFO] Scanning datasets...")

    # PlantVillage format
    pv_records = extract_from_plantvillage_format(data_root / "plantvillage" / "PlantVillage")
    print(f"   PlantVillage: {len(pv_records)} images")
    all_records.extend(pv_records)

    # Kaggle format (wheat2)
    wheat2_records = extract_from_kaggle_format(data_root / "wheat2")
    print(f"   Wheat2 (Kaggle): {len(wheat2_records)} images")
    all_records.extend(wheat2_records)

    # Wheat (might be mixed format)
    wheat_records = extract_from_kaggle_format(data_root / "wheat")
    if not wheat_records:  # Try simple format
        wheat_records = extract_from_simple_crop_format(data_root / "wheat", "wheat")
    print(f"   Wheat: {len(wheat_records)} images")
    all_records.extend(wheat_records)

    # Simple crop formats
    rice_records = extract_from_simple_crop_format(data_root / "rice", "rice")
    print(f"   Rice: {len(rice_records)} images")
    all_records.extend(rice_records)

    sugarcane_records = extract_from_simple_crop_format(data_root / "sugarcane", "sugarcane")
    print(f"   Sugarcane: {len(sugarcane_records)} images")
    all_records.extend(sugarcane_records)

    ragi_records = extract_from_simple_crop_format(data_root / "ragi" / "dataset", "ragi")
    print(f"   Ragi: {len(ragi_records)} images")
    all_records.extend(ragi_records)

    print(f"\n[OK] Total images: {len(all_records)}")

    if not all_records:
        print("[ERROR] No images found!")
        return

    # Label distribution
    label_counts = {}
    for record in all_records:
        label = record["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"[STATS] {len(label_counts)} unique disease classes:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_records)

    train_size = int(len(all_records) * 0.70)
    val_size = int(len(all_records) * 0.15)

    train_records = all_records[:train_size]
    val_records = all_records[train_size : train_size + val_size]
    test_records = all_records[train_size + val_size :]

    print(f"\n[SPLIT] Train/Val/Test:")
    print(f"   Train: {len(train_records)} (70%)")
    print(f"   Val:   {len(val_records)} (15%)")
    print(f"   Test:  {len(test_records)} (15%)")

    # Write manifests
    for split_name, records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        csv_path = manifests_dir / f"{split_name}.csv"
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"[OK] {csv_path.name}")

    # Save label mapping
    label_index = {label: idx for idx, label in enumerate(sorted(label_counts.keys()))}
    index_path = manifests_dir / "label_index.json"
    index_path.write_text(json.dumps(label_index, indent=2), encoding="utf-8")
    print(f"[OK] label_index.json ({len(label_index)} classes)")

    print("\n" + "=" * 60)
    print("Ready to train! Run: python utils/train_eval_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
