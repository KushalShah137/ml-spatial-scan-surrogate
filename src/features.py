"""
Feature Engineering Pipeline
Constructs the 5-feature spatial scan vector and target p-value for surrogate modeling.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add repo root to sys.path if running as a standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


FEATURE_COLUMNS = [
    "mu_Z",            # Expected cases (population proxy)
    "obs_exp_ratio",   # n_Z / mu_Z (observed-to-expected ratio)
    "radius",          # Great-circle radius (km)
    "area",            # Circular window area (km^2)
    "case_density",    # n_Z / area (local case density proxy)
]
TARGET_COLUMN = "p_value"


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 5-feature vector per the spatial scan surrogate paper specification.
    """
    # 1. mu_Z (population proxy)
    mu_Z = df["mu_Z"].astype(float)

    # 2. n_Z / mu_Z (observed-to-expected ratio)
    obs_exp_ratio = df["n_Z"].astype(float) / np.maximum(mu_Z, 1e-12)

    # 3. radius
    radius = df["radius"].astype(float)

    # 4. area
    area = df["area"].astype(float)

    # 5. n_Z / area (local case density proxy; 0 if area is 0 for k=1 single county)
    case_density = np.where(area > 0, df["n_Z"].astype(float) / np.maximum(area, 1e-12), 0.0)

    model_ready_df = pd.DataFrame(
        {
            "center_fips": df["center_fips"].astype(str),
            "center_county": df["center_county"].astype(str),
            "k": df["k"].astype(int),
            "n_Z": df["n_Z"].astype(float),
            "mu_Z": mu_Z,
            "obs_exp_ratio": obs_exp_ratio,
            "radius": radius,
            "area": area,
            "case_density": case_density,
            "p_value": df[TARGET_COLUMN].astype(float),
        }
    )

    return model_ready_df


def build_model_ready_data(
    input_csv_path: Path = None, output_csv_path: Path = None
) -> pd.DataFrame:
    """
    Loads labeled_windows.csv, constructs feature vector, and saves model_ready.csv.
    """
    data_dir = repo_root / "data"

    if input_csv_path is None:
        input_csv_path = data_dir / "labeled_windows.csv"
    if output_csv_path is None:
        output_csv_path = data_dir / "model_ready.csv"

    if not input_csv_path.exists():
        from src.mc_labeling import run_monte_carlo_labeling
        print(f"{input_csv_path} not found. Running MC labeling...")
        labeled_df = run_monte_carlo_labeling(output_csv_path=input_csv_path)
    else:
        labeled_df = pd.read_csv(input_csv_path, dtype={"center_fips": str})

    model_ready_df = extract_features(labeled_df)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready_df.to_csv(output_csv_path, index=False)

    print(f"Constructed {len(model_ready_df)} model-ready feature rows.")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target:   {TARGET_COLUMN}")
    print(f"Saved to: {output_csv_path}")

    return model_ready_df


if __name__ == "__main__":
    build_model_ready_data()
