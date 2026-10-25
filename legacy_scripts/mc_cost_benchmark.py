"""
Monte Carlo vs. Surrogate Runtime Benchmark
Compares the computational cost of traditional Monte Carlo significance testing vs. the ML surrogate.
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

from src.mc_labeling import build_membership_matrix


def benchmark_single_window_mc(
    county_df: pd.DataFrame,
    R: int = 999,
    num_runs: int = 20,
    max_k: int = 20,
    random_seed: int = 42,
) -> tuple[float, float]:
    """
    Measures the runtime of establishing statistical significance for ONE candidate window
    the traditional way: generating the entire R=999 null distribution from scratch and comparing.

    Returns:
        mean_time_ms: mean runtime in milliseconds
        std_time_ms: standard deviation in milliseconds
    """
    N = int(county_df["total_deaths"].sum())
    P_total = float(county_df["avg_population"].sum())
    county_probs = (county_df["avg_population"] / P_total).values

    # Pre-build spatial membership matrix W
    W, _ = build_membership_matrix(county_df, max_k=max_k)
    mu_Z = (W @ (N * county_probs))[:, None]

    # Representative observed test window log_lr
    test_window_log_lr = 10.0

    timings_ms = []
    rng = np.random.default_rng(random_seed)

    for _ in range(num_runs):
        t0 = time.perf_counter()

        # 1. Draw R multinomial replicate datasets under H0
        sim_cases = rng.multinomial(N, county_probs, size=R)

        # 2. Vectorized case counts across all candidate windows
        n_Z = W @ sim_cases.T

        # 3. Compute log-likelihood ratio for all windows
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

        # 4. Record maximum log_lr across all windows per replicate
        null_distribution = np.max(log_lr, axis=0)

        # 5. Compute empirical p-value for the candidate window
        _ = (1.0 + np.sum(null_distribution >= test_window_log_lr)) / (float(R) + 1.0)

        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    return float(np.mean(timings_ms)), float(np.std(timings_ms))


def run_benchmark(
    county_csv_path: Path = None,
    comparison_csv_path: Path = None,
    output_csv_path: Path = None,
    R: int = 999,
    num_runs: int = 20,
) -> pd.DataFrame:
    """
    Runs runtime benchmarking and exports speedup comparison table.
    """
    data_dir = repo_root / "data"
    results_dir = repo_root / "results"

    if county_csv_path is None:
        county_csv_path = data_dir / "county_clean.csv"
    if comparison_csv_path is None:
        comparison_csv_path = results_dir / "comparison.csv"
    if output_csv_path is None:
        output_csv_path = results_dir / "speedup_comparison.csv"

    if not county_csv_path.exists():
        from src.data_prep import prep_mortality_data
        county_df = prep_mortality_data(output_csv_path=county_csv_path)
    else:
        county_df = pd.read_csv(county_csv_path, dtype={"fips": str})

    if not comparison_csv_path.exists():
        from src.train_models import train_and_evaluate
        comp_df = train_and_evaluate(results_dir=results_dir)
    else:
        comp_df = pd.read_csv(comparison_csv_path)

    # Retrieve XGBoost surrogate latency
    surrogate_row = comp_df[comp_df["model_name"].str.contains("XGBoost", case=False)]
    if len(surrogate_row) == 0:
        raise ValueError("XGBoost surrogate not found in results/comparison.csv")
    surrogate_time_ms = float(surrogate_row.iloc[0]["mean_inference_time_ms"])

    print(f"\nBenchmarking single-window Monte Carlo significance testing (R={R}, {num_runs} runs)...")
    mc_mean_ms, mc_std_ms = benchmark_single_window_mc(
        county_df=county_df, R=R, num_runs=num_runs
    )

    # Speedup factor
    speedup_factor = mc_mean_ms / surrogate_time_ms

    speedup_df = pd.DataFrame(
        [
            {
                "method": "Traditional Monte Carlo (R=999)",
                "mean_time_ms": mc_mean_ms,
                "speedup_factor": 1.0,
            },
            {
                "method": "XGBoost Surrogate (1 prediction)",
                "mean_time_ms": surrogate_time_ms,
                "speedup_factor": speedup_factor,
            },
        ]
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    speedup_df.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 75)
    print("RUNTIME & SPEEDUP BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Traditional Monte Carlo (R={R}):  {mc_mean_ms:.3f} ms (+/- {mc_std_ms:.3f} ms)")
    print(f"XGBoost ML Surrogate:             {surrogate_time_ms:.6f} ms per window")
    print("-" * 75)
    print(f"SPEEDUP FACTOR:                   {speedup_factor:,.1f}x FASTER")
    print("=" * 75)
    print(f"\nSaved benchmark results to: {output_csv_path}\n")

    return speedup_df


if __name__ == "__main__":
    run_benchmark()
