from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.vision_agent import run_vision
def _ece(confidences: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    if len(confidences) == 0:
        return 1.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        if not np.any(mask):
            continue
        ece += abs(float(np.mean(correctness[mask])) - float(np.mean(confidences[mask]))) * (np.sum(mask) / len(confidences))
    return float(ece)


def eval_with_trained_model(manifest_csv: str, model_dir: str, out_dir: str) -> dict[str, Any]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_csv}")

    frame = pd.read_csv(manifest_path)
    rows = []
    for _, row in frame.iterrows():
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        pred_state = run_vision(
            {
                "image": image,
                "image_path": str(image_path),
                "offline": False,
                "allow_path_hints": False,
                "model_dir": model_dir,
            }
        )
        pred = pred_state.get("disease_prediction", {})

        crop = str(row.get("crop", "unknown"))
        disease = str(row.get("disease", "unknown"))
        true_label = disease if disease.startswith(crop + "_") else f"{crop}_{disease}"
        confidence = float(pred.get("confidence", 0.0))
        pred_label = str(pred.get("label", "unknown"))
        rows.append(
            {
                "crop": crop,
                "disease": disease,
                "image_path": str(image_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": int(pred_label == true_label),
            }
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_file = out_path / "prediction_rows.csv"
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No evaluable rows found in manifest.")
    frame.to_csv(pred_file, index=False)

    y_true = frame["true_label"].tolist()
    y_pred = frame["pred_label"].tolist()
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv = out_path / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv)

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Greens")
    plt.title("AgriBloom Day 4 Trained Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=60, ha="right", fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    plt.tight_layout()
    cm_png = out_path / "confusion_matrix.png"
    fig.savefig(cm_png, dpi=160)
    plt.close(fig)

    conf = frame["confidence"].astype(float).to_numpy()
    corr = frame["correct"].astype(int).to_numpy()
    prob_true, prob_pred = calibration_curve(corr, conf, n_bins=8, strategy="uniform")
    cal_df = pd.DataFrame({"mean_predicted_confidence": prob_pred, "fraction_of_positives": prob_true})
    cal_csv = out_path / "calibration_curve.csv"
    cal_df.to_csv(cal_csv, index=False)

    fig2 = plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Day 4 Calibration Curve (Trained)")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    cal_png = out_path / "calibration_curve.png"
    fig2.savefig(cal_png, dpi=160)
    plt.close(fig2)

    per_crop = (
        frame.groupby("crop")
        .agg(samples=("crop", "count"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )
    per_crop["accuracy"] = per_crop["accuracy"].round(4)
    per_crop["mean_confidence"] = per_crop["mean_confidence"].round(4)
    per_crop_csv = out_path / "per_crop_metrics.csv"
    per_crop.to_csv(per_crop_csv, index=False)

    overall_acc = float(frame["correct"].mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    ece = _ece(confidences=conf, correctness=corr)
    leaderboard_score = round((overall_acc * 60.0) + (macro_f1 * 30.0) + ((1.0 - ece) * 10.0), 3)

    summary = {
        "project": "AgriBloom Agentic",
        "trained_model_dir": model_dir,
        "total_samples": int(len(frame)),
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "leaderboard_score": leaderboard_score,
        "files": {
            "prediction_rows": str(pred_file),
            "per_crop_metrics": str(per_crop_csv),
            "confusion_matrix_csv": str(cm_csv),
            "confusion_matrix_png": str(cm_png),
            "calibration_curve_csv": str(cal_csv),
            "calibration_curve_png": str(cal_png),
        },
    }

    summary_path = out_path / "trained_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4 evaluate fine-tuned vision model")
    parser.add_argument("--manifest-csv", default="data/manifests/test.csv")
    parser.add_argument("--model-dir", default="models/checkpoints/day4_vit")
    parser.add_argument("--out-dir", default="models/reports/day4")
    args = parser.parse_args()

    summary = eval_with_trained_model(manifest_csv=args.manifest_csv, model_dir=args.model_dir, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
