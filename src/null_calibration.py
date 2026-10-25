import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution, kstest

from scan import build_neighbors, dataset_features, null_max_distribution, observed_llr, window_limits
from synth import sample_case_counts, sample_region


ROOT = Path(__file__).resolve().parent.parent
_WORKER = {}


def initialize_worker(pop_cap, ridge_path, xgb_path, mc99_count):
    _WORKER["pop_cap"] = float(pop_cap)
    _WORKER["ridge"] = joblib.load(ridge_path)
    _WORKER["xgb"] = joblib.load(xgb_path)
    _WORKER["mc99_count"] = int(mc99_count)


def gumbel_probability(llr, mu, beta):
    if float(llr) == 0:
        return 1.0
    beta = max(float(beta), np.finfo(float).tiny)
    z = float(np.clip((float(llr) - float(mu)) / beta, -700.0, 700.0))
    return float(np.clip(1.0 - np.exp(-np.exp(-z)), 0.001, 1.0))


def evaluate_dataset(index):
    random = np.random.default_rng(500000 + int(index))
    is_national = (int(index) + 1) % 9 == 0
    region, cached_neighbors = sample_region(random, is_national)
    population = region["population"].to_numpy(dtype=float)
    if cached_neighbors is None:
        neighbors, _ = build_neighbors(region["lat"].to_numpy(), region["lon"].to_numpy())
    else:
        neighbors = cached_neighbors
    total_cases, counts, _, _, _, _ = sample_case_counts(population, random, neighbors, pure_null=True)
    limits = window_limits(population, neighbors, _WORKER["pop_cap"])
    scores = observed_llr(counts, population, neighbors, limits)
    observed_max = float(np.max(scores))
    stats = dataset_features(population, counts, int(limits.sum()))
    features = np.asarray([[
        math.log(float(stats["N"])),
        float(stats["n_counties"]),
        math.log(float(stats["n_windows"])),
        math.log(float(stats["total_pop"])),
        float(stats["pop_gini"]),
        float(stats["max_pop_share"]),
        float(stats["frac_counties_mu_below_1"]),
    ]], dtype=float)
    xgb_parameters = np.asarray(_WORKER["xgb"].predict(features), dtype=float)[0]
    ridge_parameters = np.asarray(_WORKER["ridge"].predict(features), dtype=float)[0]
    xgb_beta = float(np.exp(np.clip(xgb_parameters[1], -30.0, 30.0)))
    ridge_beta = float(np.exp(np.clip(ridge_parameters[1], -30.0, 30.0)))
    row = {
        "dataset_id": 500000 + int(index),
        "seed": 500000 + int(index),
        "is_national": is_national,
        "n_counties": int(stats["n_counties"]),
        "N": int(total_cases),
        "n_windows": int(stats["n_windows"]),
        "observed_max_llr": observed_max,
        "gumbel_pred_xgb": gumbel_probability(observed_max, xgb_parameters[0], xgb_beta),
        "gumbel_pred_ridge": gumbel_probability(observed_max, ridge_parameters[0], ridge_beta),
        "mc99": np.nan,
        "gumbel99": np.nan,
    }
    if index < _WORKER["mc99_count"]:
        simulated = null_max_distribution(population, total_cases, neighbors, limits, 99, 1000 + int(index))
        row["mc99"] = float((1 + int(np.sum(simulated >= observed_max))) / 100.0)
        beta = max(float(np.std(simulated)) * math.sqrt(6.0) / math.pi, np.finfo(float).tiny)
        mu = float(np.mean(simulated)) - 0.5772 * beta
        row["gumbel99"] = gumbel_probability(observed_max, mu, beta)
    return row


def clopper_pearson(successes, trials):
    alpha = 0.05
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def build_summary(results):
    methods = ["gumbel_pred_xgb", "gumbel_pred_ridge", "gumbel99", "mc99"]
    rows = []
    for method in methods:
        values = results[method].dropna().to_numpy(dtype=float)
        row = {"method": method, "n": len(values)}
        for alpha in (0.01, 0.05, 0.10):
            rejected = int(np.sum(values < alpha))
            lower, upper = clopper_pearson(rejected, len(values))
            label = f"{alpha:.2f}"
            row[f"reject_count_{label}"] = rejected
            row[f"rejection_fraction_{label}"] = rejected / len(values) if len(values) else np.nan
            row[f"ci95_low_{label}"] = lower
            row[f"ci95_high_{label}"] = upper
        if method.startswith("gumbel"):
            statistic = kstest(values, "uniform") if len(values) else None
            row["ks_statistic"] = float(statistic.statistic) if statistic else np.nan
            row["ks_p_value"] = float(statistic.pvalue) if statistic else np.nan
        else:
            row["ks_statistic"] = np.nan
            row["ks_p_value"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mc99-count", type=int, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.mc99_count < 0 or args.mc99_count > args.count:
        parser.error("count and workers must be positive, and mc99-count must be between zero and count.")
    config = json.loads((ROOT / "config" / "scan.json").read_text(encoding="utf-8"))
    models_dir = ROOT / "results" / "models"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(config["pop_cap"], str(models_dir / "gumbel_pred_ridge.joblib"), str(models_dir / "gumbel_pred_xgb.joblib"), args.mc99_count),
    ) as executor:
        rows = list(executor.map(evaluate_dataset, range(args.count), chunksize=1))
    results = pd.DataFrame(rows)
    tables_dir = ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "null_calibration.csv"
    summary_path = tables_dir / "null_calibration_summary.csv"
    summary = build_summary(results)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Null calibration datasets: {len(results)}; regional={int((~results['is_national']).sum())}; national={int(results['is_national'].sum())}")
    print(f"Saved calibration rows: {results_path} ({len(results)} rows)")
    print(f"Summary rows: {summary_path} ({len(summary)} rows)")
    print(summary.to_string(index=False, na_rep="NA", float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
