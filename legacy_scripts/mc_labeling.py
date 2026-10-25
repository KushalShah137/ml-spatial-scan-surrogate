"""
Monte Carlo Significance Testing and Window Labeling
Vectorized Monte Carlo replication engine for Kulldorff's spatial scan statistic.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

# Add repo root to sys.path if running as a standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.scan_engine import haversine_distance_matrix


def build_membership_matrix(
    county_df: pd.DataFrame, max_k: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs each candidate window's county membership as a binary matrix W (n_windows x n_counties).
    
    Returns:
        W: binary membership matrix of shape (n_windows, n_counties)
        dist_matrix: pairwise distance matrix in km
    """
    num_counties = len(county_df)
    num_windows = num_counties * max_k
    W = np.zeros((num_windows, num_counties), dtype=np.float64)

    lats = county_df["lat"].values
    lons = county_df["lon"].values
    dist_matrix = haversine_distance_matrix(lats, lons)

    window_idx = 0
    for i in range(num_counties):
        sorted_indices = np.argsort(dist_matrix[i])
        for k in range(1, max_k + 1):
            k_indices = sorted_indices[:k]
            W[window_idx, k_indices] = 1.0
            window_idx += 1

    return W, dist_matrix


def run_monte_carlo_labeling(
    county_csv_path: Path = None,
    candidate_windows_csv_path: Path = None,
    output_csv_path: Path = None,
    R: int = 999,
    random_seed: int = 42,
    max_k: int = 20,
) -> pd.DataFrame:
    """
    Performs vectorized Monte Carlo significance testing across R replications and computes empirical p-values.
    """
    t_start = time.time()
    data_dir = repo_root / "data"

    if county_csv_path is None:
        county_csv_path = data_dir / "county_clean.csv"
    if candidate_windows_csv_path is None:
        candidate_windows_csv_path = data_dir / "candidate_windows.csv"
    if output_csv_path is None:
        output_csv_path = data_dir / "labeled_windows.csv"

    # Ensure required input files exist
    if not county_csv_path.exists():
        from src.data_prep import prep_mortality_data
        county_df = prep_mortality_data(output_csv_path=county_csv_path)
    else:
        county_df = pd.read_csv(county_csv_path, dtype={"fips": str})

    if not candidate_windows_csv_path.exists():
        from src.scan_engine import run_scan_engine
        cand_df = run_scan_engine(
            input_csv_path=county_csv_path,
            output_csv_path=candidate_windows_csv_path,
            max_k=max_k,
        )
    else:
        cand_df = pd.read_csv(candidate_windows_csv_path, dtype={"center_fips": str})

    # Fixed parameters
    N = int(county_df["total_deaths"].sum())
    P_total = float(county_df["avg_population"].sum())
    county_probs = (county_df["avg_population"] / P_total).values

    # Reconstruct binary membership matrix W (n_windows x n_counties)
    W, _ = build_membership_matrix(county_df, max_k=max_k)
    num_windows = W.shape[0]

    # Vector of expected cases per window under the null: mu_Z shape (n_windows, 1)
    mu_Z = (W @ (N * county_probs))[:, None]

    # Draw R multinomial samples across all counties in one vectorized generation: shape (R, n_counties)
    rng = np.random.default_rng(random_seed)
    sim_cases = rng.multinomial(N, county_probs, size=R)  # shape (R, n_counties)

    # Compute n_Z for every window across all replicates simultaneously: shape (n_windows, R)
    n_Z = W @ sim_cases.T  # shape (n_windows, R)

    # Compute vectorized log_lr for every window across all replicates
    log_lr = np.zeros_like(n_Z, dtype=np.float64)
    mask = n_Z > mu_Z

    if np.any(mask):
        n_Z_m = n_Z[mask]
        mu_Z_m = np.broadcast_to(mu_Z, n_Z.shape)[mask]

        term1 = n_Z_m * np.log(n_Z_m / mu_Z_m)
        rem_n = N - n_Z_m
        rem_mu = N - mu_Z_m
        term2 = np.where(
            rem_n > 0,
            rem_n * np.log(np.maximum(rem_n, 1e-12) / np.maximum(rem_mu, 1e-12)),
            0.0,
        )
        log_lr[mask] = term1 + term2

    # Maximum log_lr across all candidate windows for each replicate: shape (R,)
    null_distribution = np.max(log_lr, axis=0)

    # Compute p-values for actual candidate windows
    actual_log_lr = cand_df["log_lr"].values[:, None]  # shape (n_windows, 1)
    # Count how many null replicates had a maximum log_lr >= window's actual log_lr
    greater_equal_counts = np.sum(null_distribution[None, :] >= actual_log_lr, axis=1)
    p_values = (1.0 + greater_equal_counts) / (float(R) + 1.0)

    labeled_df = cand_df.copy()
    labeled_df["p_value"] = p_values

    # Save to /data/labeled_windows.csv
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_csv_path, index=False)

    runtime_sec = time.time() - t_start

    # Summary statistics
    mean_p = float(np.mean(p_values))
    min_p = float(np.min(p_values))
    max_p = float(np.max(p_values))
    sig_count = int(np.sum(p_values < 0.05))

    print("\n" + "=" * 70)
    print("MONTE CARLO LABELING SUMMARY")
    print("=" * 70)
    print(f"Replications (R):                 {R}")
    print(f"Total candidate windows:          {num_windows}")
    print(f"Mean p-value:                     {mean_p:.4f}")
    print(f"Min p-value:                      {min_p:.4f}")
    print(f"Max p-value:                      {max_p:.4f}")
    print(f"Windows with p-value < 0.05:      {sig_count} ({sig_count / num_windows * 100:.2f}%)")
    print(f"Runtime:                          {runtime_sec:.3f} seconds")
    print("-" * 70)
    print(f"Saved labeled windows to: {output_csv_path}")
    print("=" * 70 + "\n")

    return labeled_df


if __name__ == "__main__":
    run_monte_carlo_labeling()
