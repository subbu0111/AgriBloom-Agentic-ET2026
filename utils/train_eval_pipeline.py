from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, check=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command train + eval pipeline")
    parser.add_argument("--train-csv", default="data/manifests/train.csv")
    parser.add_argument("--val-csv", default="data/manifests/val.csv")
    parser.add_argument("--test-csv", default="data/manifests/test.csv")
    parser.add_argument("--output-dir", default="models/checkpoints/vit_crop_disease")
    parser.add_argument("--report-dir", default="models/reports/evaluation")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--demo-sanity", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    train_cmd = [
        sys.executable,
        str(project_root / "utils" / "train_vision.py"),
        "--train-csv",
        args.train_csv,
        "--val-csv",
        args.val_csv,
        "--output-dir",
        args.output_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
    ]
    if args.demo_sanity:
        train_cmd.append("--demo-sanity")

    eval_cmd = [
        sys.executable,
        str(project_root / "utils" / "evaluate.py"),
        "--manifest-csv",
        args.test_csv,
        "--model-dir",
        args.output_dir,
        "--out-dir",
        args.report_dir,
    ]

    _run(train_cmd)
    _run(eval_cmd)

    summary = Path(args.report_dir) / "trained_eval_summary.json"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"))
    else:
        print(json.dumps({"warning": "No summary generated"}, indent=2))


if __name__ == "__main__":
    main()
