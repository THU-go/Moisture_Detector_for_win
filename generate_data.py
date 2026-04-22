import argparse
from pathlib import Path

from dependency_bootstrap import ensure_dependencies

ensure_dependencies(
    {
        "numpy": "numpy>=1.24",
        "pandas": "pandas>=2.0",
    }
)

import numpy as np
import pandas as pd


def generate_synthetic_dataset(n_samples: int, random_seed: int) -> pd.DataFrame:
    """
    Generate synthetic data for pipeline validation only.

    Physical intuition (simplified):
    - Moisture (target) is the latent variable we want.
    - Detector voltage tends to decrease when moisture increases
      because higher water content can increase attenuation.
    - Temperature and humidity introduce drift/disturbance.
    - Add nonlinear terms and random noise to mimic real measurements.

    NOTE: This is NOT real experimental data.
    """
    rng = np.random.default_rng(random_seed)

    # Simulate environmental factors
    temperature = rng.uniform(5.0, 45.0, size=n_samples)  # degree C
    humidity = rng.uniform(15.0, 95.0, size=n_samples)  # %RH

    # Latent true moisture (%) in a plausible range for root-zone monitoring
    # Make moisture partly dependent on environment + randomness
    base_moisture = (
        28.0
        + 0.22 * humidity
        - 0.18 * temperature
        + 5.0 * np.sin(humidity / 18.0)
        + rng.normal(0.0, 4.5, size=n_samples)
    )
    moisture = np.clip(base_moisture, 3.0, 65.0)

    # Microwave detector voltage model (V), inverse relation with moisture
    # Add drift terms and weak nonlinear interactions
    voltage = (
        2.65
        - 0.024 * moisture
        + 0.0022 * temperature
        - 0.0015 * humidity
        + 0.00006 * temperature * humidity
        + 0.018 * np.sin(moisture / 6.0)
        + rng.normal(0.0, 0.045, size=n_samples)
    )
    voltage = np.clip(voltage, 0.25, 2.95)

    # Final label includes small extra measurement uncertainty
    measured_moisture = np.clip(moisture + rng.normal(0.0, 1.2, size=n_samples), 0.0, 70.0)

    return pd.DataFrame(
        {
            "voltage": voltage,
            "temperature": temperature,
            "humidity": humidity,
            "moisture": measured_moisture,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic moisture monitoring dataset.")
    parser.add_argument("--n-samples", type=int, default=6000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_moisture.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset(n_samples=args.n_samples, random_seed=args.seed)
    df.to_csv(output_path, index=False)

    print("=== Synthetic Dataset Generation Complete ===")
    print("IMPORTANT: This CSV is synthetic data for pipeline validation/debugging only.")
    print(f"Saved to: {output_path.resolve()}")
    print(f"Shape: {df.shape}")
    print("\nColumn statistics:")
    print(df.describe().round(3))


if __name__ == "__main__":
    main()
