"""
Production ViT Training Pipeline for Crop Disease Detection
Optimized for NVIDIA RTX 4060 with Mixed Precision Training

Supports: Maize, Tomato, Potato, Rice, Wheat, Ragi, Sugarcane
Data sources: PlantVillage + Kaggle datasets
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Target classes (36 total for full dataset)
TARGET_CLASSES = [
    "maize_blight", "maize_common_rust", "maize_gray_leaf_spot", "maize_healthy",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
    "tomato_leaf_mold", "tomato_septoria_leaf_spot", "tomato_spider_mites",
    "tomato_target_spot", "tomato_mosaic_virus", "tomato_yellow_leaf_curl", "tomato_healthy",
    "potato_early_blight", "potato_late_blight", "potato_healthy",
    "rice_bacterial_leaf_blight", "rice_brown_spot", "rice_leaf_blast",
    "rice_leaf_scald", "rice_narrow_brown_spot", "rice_healthy",
    "wheat_brown_rust", "wheat_leaf_rust", "wheat_septoria", "wheat_yellow_rust", "wheat_healthy",
    "ragi_blast", "ragi_brown_spot", "ragi_healthy",
    "sugarcane_bacterial_blight", "sugarcane_red_rot", "sugarcane_rust", "sugarcane_healthy",
]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get optimal device with GPU info logging."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        return device
    logger.warning("CUDA not available. Training on CPU (will be slow).")
    return torch.device("cpu")


def normalize_label(crop: str, disease: str) -> str:
    """Normalize label to standard format."""
    c = crop.lower().strip().replace(" ", "_").replace("-", "_")
    d = disease.lower().strip().replace(" ", "_").replace("-", "_")

    # Handle common naming variations
    c = c.replace("corn", "maize").replace("paddy", "rice")
    c = c.replace("finger_millet", "ragi").replace("nachni", "ragi")

    if d.startswith(c + "_"):
        return d
    return f"{c}_{d}"


@dataclass
class TrainingConfig:
    """Training configuration with optimal defaults for RTX 4060."""
    epochs: int = 15
    batch_size: int = 16  # Optimal for 8GB VRAM
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    freeze_backbone: bool = False
    use_mixed_precision: bool = True  # FP16 for faster training
    num_workers: int = 4
    seed: int = 42
    save_total_limit: int = 3


@dataclass
class Example:
    """Single training example."""
    image_path: str
    label: str
    crop: str = ""
    source: str = ""


class CropDiseaseDataset(Dataset):
    """
    Dataset for crop disease images with augmentation support.
    Handles PlantVillage and Kaggle dataset formats.
    """

    def __init__(
        self,
        examples: list[Example],
        processor: ViTImageProcessor,
        label_to_id: dict[str, int],
        augment: bool = False,
    ) -> None:
        self.examples = examples
        self.processor = processor
        self.label_to_id = label_to_id
        self.augment = augment

        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        try:
            image = Image.open(example.image_path).convert("RGB")

            # Apply augmentation during training
            if self.augment:
                image = self.train_transform(image)

            # Process through ViT processor
            encoded = self.processor(images=image, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)

            label_id = self.label_to_id.get(example.label, 0)

            return {
                "pixel_values": pixel_values,
                "labels": torch.tensor(label_id, dtype=torch.long),
            }
        except Exception as e:
            logger.warning(f"Error loading image {example.image_path}: {e}")
            # Return a placeholder
            return {
                "pixel_values": torch.zeros(3, 224, 224),
                "labels": torch.tensor(0, dtype=torch.long),
            }


def load_manifest(csv_path: Path) -> list[Example]:
    """Load examples from CSV manifest."""
    if not csv_path.exists():
        logger.warning(f"Manifest not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    examples = []

    for _, row in df.iterrows():
        image_path = Path(str(row["image_path"]))

        # Handle relative paths
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.exists():
            continue

        crop = str(row.get("crop", "unknown"))
        disease = str(row.get("disease", "unknown"))
        source = str(row.get("source", "unknown"))

        label = normalize_label(crop, disease)

        examples.append(Example(
            image_path=str(image_path),
            label=label,
            crop=crop,
            source=source,
        ))

    logger.info(f"Loaded {len(examples)} examples from {csv_path}")
    return examples


def create_weighted_sampler(examples: list[Example], label_to_id: dict[str, int]) -> WeightedRandomSampler:
    """Create weighted sampler for class imbalance."""
    label_counts = {}
    for ex in examples:
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1

    total = len(examples)
    class_weights = {label: total / count for label, count in label_counts.items()}

    sample_weights = [class_weights[ex.label] for ex in examples]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.should_stop


def train(
    train_csv: Path,
    val_csv: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """
    Main training function with GPU optimization.

    Args:
        train_csv: Path to training manifest CSV
        val_csv: Path to validation manifest CSV
        output_dir: Directory to save checkpoints
        config: Training configuration

    Returns:
        Training summary dictionary
    """
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    use_amp = config.use_mixed_precision and device.type == "cuda"

    # Load data
    train_examples = load_manifest(train_csv)
    val_examples = load_manifest(val_csv)

    if not train_examples:
        raise ValueError(f"No valid training examples found in {train_csv}")
    if not val_examples:
        raise ValueError(f"No valid validation examples found in {val_csv}")

    # Build label mapping from data
    all_labels = sorted(set(ex.label for ex in train_examples + val_examples))
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    logger.info(f"Training with {len(all_labels)} classes:")
    for label in all_labels:
        count = sum(1 for ex in train_examples if ex.label == label)
        logger.info(f"  {label}: {count} samples")

    # Load model and processor
    base_model = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(base_model)

    model = ViTForImageClassification.from_pretrained(
        base_model,
        num_labels=len(all_labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    # Freeze backbone if requested
    if config.freeze_backbone:
        logger.info("Freezing ViT backbone - only training classifier head")
        for param in model.vit.parameters():
            param.requires_grad = False

    model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,}")

    # Create datasets and dataloaders
    train_dataset = CropDiseaseDataset(
        examples=train_examples,
        processor=processor,
        label_to_id=label_to_id,
        augment=True,
    )

    val_dataset = CropDiseaseDataset(
        examples=val_examples,
        processor=processor,
        label_to_id=label_to_id,
        augment=False,
    )

    # Weighted sampler for class imbalance
    sampler = create_weighted_sampler(train_examples, label_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 2,  # Restart every 2 epochs
        T_mult=2,
        eta_min=1e-7,
    )

    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None

    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)

    # Training loop
    best_val_acc = 0.0
    history = []

    logger.info(f"Starting training for {config.epochs} epochs")
    logger.info(f"Mixed precision: {use_amp}, Gradient accumulation: {config.gradient_accumulation_steps}")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Mixed precision forward pass
            if use_amp:
                with autocast():
                    outputs = model(pixel_values=pixel_values)
                    loss = F.cross_entropy(outputs.logits, labels)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(pixel_values=pixel_values)
                loss = F.cross_entropy(outputs.logits, labels)
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            train_loss += loss.item() * config.gradient_accumulation_steps
            preds = torch.argmax(outputs.logits, dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if use_amp:
                    with autocast():
                        outputs = model(pixel_values=pixel_values)
                        loss = F.cross_entropy(outputs.logits, labels)
                else:
                    outputs = model(pixel_values=pixel_values)
                    loss = F.cross_entropy(outputs.logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "time_seconds": round(epoch_time, 1),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best model! Saving to {output_dir}")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

            # Save label map (convert numpy int64 to int for JSON serialization)
            label_map = {
                "labels": [int(x) for x in all_labels],
                "label_to_id": {k: int(v) for k, v in label_to_id.items()}
            }
            (output_dir / "label_map.json").write_text(
                json.dumps(label_map, indent=2), encoding="utf-8"
            )

        # Early stopping check
        if early_stopping(val_acc):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Clear GPU cache periodically
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save training history and summary
    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    summary = {
        "best_val_acc": round(best_val_acc, 4),
        "final_train_acc": round(train_acc, 4),
        "num_classes": len(label_to_id),
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
        "epochs_trained": len(history),
        "config": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "mixed_precision": use_amp,
            "freeze_backbone": config.freeze_backbone,
        },
        "device": str(device),
        "checkpoint_dir": str(output_dir),
    }

    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    return summary


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train ViT model for crop disease detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument("--train-csv", type=str, default="data/manifests/train.csv",
                        help="Path to training manifest CSV")
    parser.add_argument("--val-csv", type=str, default="data/manifests/val.csv",
                        help="Path to validation manifest CSV")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints/vit_crop_disease",
                        help="Directory to save model checkpoints")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (16 optimal for RTX 4060 8GB)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--grad-accumulation", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze ViT backbone (faster training)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Early stopping patience")

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accumulation,
        freeze_backbone=args.freeze_backbone,
        use_mixed_precision=not args.no_amp,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping_patience=args.early_stopping,
    )

    summary = train(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        output_dir=Path(args.output_dir),
        config=config,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
