"""One-click runner for beginners.

This script installs missing dependencies first, then runs:
1) synthetic data generation
2) model training/evaluation
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from dependency_bootstrap import ensure_dependencies


ALL_REQUIREMENTS = {
    "numpy": "numpy>=1.24",
    "pandas": "pandas>=2.0",
    "sklearn": "scikit-learn>=1.3",
    "matplotlib": "matplotlib>=3.7",
    "torch": "torch>=2.2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Install dependencies and run full baseline pipeline.")
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data/synthetic_moisture.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=18)
    args = parser.parse_args()

    ensure_dependencies(ALL_REQUIREMENTS)

    print("\n[Pipeline] Step 1/2: Generate synthetic dataset...")
    subprocess.check_call(
        [
            sys.executable,
            "generate_data.py",
            "--n-samples",
            str(args.n_samples),
            "--seed",
            str(args.seed),
            "--output",
            args.data_path,
        ]
    )

    print("\n[Pipeline] Step 2/2: Train and evaluate model...")
    subprocess.check_call(
        [
            sys.executable,
            "train.py",
            "--data-path",
            args.data_path,
            "--output-dir",
            args.output_dir,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
        ]
    )

    print("\n[Pipeline] All done. Artifacts are saved in the outputs directory.")


if __name__ == "__main__":
    main()
