"""
Spatial Scan Engine
Implements Kulldorff's circular spatial scan statistic using k-NN candidate windows.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def haversine_distance_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Computes pairwise great-circle distance matrix (in km) using the Haversine formula.
    """
    R = 6371.0  # Mean Earth radius in kilometers
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    # Ensure numerical safety inside arcsin
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def compute_kulldorff_lr(n_Z: float, mu_Z: float, N: float) -> tuple[float, float]:
    """
    Calculates Kulldorff's likelihood ratio statistic for Poisson model:
    LR(Z) = (n_Z/mu_Z)^n_Z * ((N-n_Z)/(N-mu_Z))^(N-n_Z), only when n_Z > mu_Z.
    
    Returns (LR(Z), log_LR(Z)).
    """
    if n_Z <= mu_Z or mu_Z <= 0:
        return 1.0, 0.0

    # Log-likelihood ratio calculation for numeric stability
    term1 = n_Z * np.log(n_Z / mu_Z)
    if N > n_Z and (N - mu_Z) > 0:
        term2 = (N - n_Z) * np.log((N - n_Z) / (N - mu_Z))
    else:
        term2 = 0.0

    log_lr = term1 + term2

    # Prevent float64 overflow when computing exp(log_lr)
    if log_lr > 700:
        lr = np.inf
    else:
        lr = float(np.exp(log_lr))

    return lr, float(log_lr)


def generate_candidate_windows(
    county_df: pd.DataFrame, max_k: int = 20
) -> pd.DataFrame:
    """
    Generates candidate circular windows via k-nearest-neighbor county sets (k = 1 to max_k)
    per county centroid using great-circle distance.
    """
    N = county_df["total_deaths"].sum()
    P_total = county_df["avg_population"].sum()

    lats = county_df["lat"].values
    lons = county_df["lon"].values
    dist_matrix = haversine_distance_matrix(lats, lons)

    num_counties = len(county_df)
    windows = []

    for i in range(num_counties):
        center_row = county_df.iloc[i]
        center_fips = center_row["fips"]
        center_county = center_row["county_name"]

        # Sort all counties by distance to county i
        sorted_indices = np.argsort(dist_matrix[i])

        for k in range(1, min(max_k + 1, num_counties + 1)):
            k_indices = sorted_indices[:k]
            sub = county_df.iloc[k_indices]

            n_Z = float(sub["total_deaths"].sum())
            p_Z = float(sub["avg_population"].sum())
            mu_Z = float(N * (p_Z / P_total))

            # Radius is distance from center county to the k-th neighbor (0 for k=1)
            radius = float(dist_matrix[i, sorted_indices[k - 1]])
            area = float(np.pi * (radius ** 2))

            lr, log_lr = compute_kulldorff_lr(n_Z, mu_Z, N)

            windows.append(
                {
                    "center_fips": center_fips,
                    "center_county": center_county,
                    "k": k,
                    "n_Z": n_Z,
                    "mu_Z": mu_Z,
                    "radius": radius,
                    "area": area,
                    "LR(Z)": lr,
                    "log_lr": log_lr,
                }
            )

    return pd.DataFrame(windows)


def run_scan_engine(
    input_csv_path: Path = None,
    output_csv_path: Path = None,
    max_k: int = 20,
) -> pd.DataFrame:
    """
    Runs spatial scan engine on cleaned county data, saves candidate windows,
    and prints summary statistics.
    """
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    if input_csv_path is None:
        input_csv_path = data_dir / "county_clean.csv"

    if output_csv_path is None:
        output_csv_path = data_dir / "candidate_windows.csv"

    if not input_csv_path.exists():
        from src.data_prep import prep_mortality_data
        print(f"{input_csv_path} not found. Running data prep first...")
        county_df = prep_mortality_data(output_csv_path=input_csv_path)
    else:
        county_df = pd.read_csv(input_csv_path, dtype={"fips": str})

    windows_df = generate_candidate_windows(county_df, max_k=max_k)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    windows_df.to_csv(output_csv_path, index=False)

    # Print summary
    county_count = len(county_df)
    window_count = len(windows_df)
    top5 = windows_df.sort_values(by="log_lr", ascending=False).head(5)

    print("\n" + "=" * 70)
    print("SPATIAL SCAN ENGINE SUMMARY")
    print("=" * 70)
    print(f"County count:            {county_count}")
    print(f"Candidate window count:  {window_count}")
    print("-" * 70)
    print("Top 5 Windows by Kulldorff LR(Z):")
    print("-" * 70)
    display_cols = ["center_county", "center_fips", "k", "n_Z", "mu_Z", "radius", "LR(Z)", "log_lr"]
    print(top5[display_cols].to_string(index=False))
    print("=" * 70 + "\n")

    return windows_df


if __name__ == "__main__":
    run_scan_engine()
