import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from build_table import FEATURES, add_model_features
from scan import dataset_features, load_dataset
from synth import sample_case_counts
from scan import build_neighbors, null_max_distribution, window_limits


ROOT = Path(__file__).resolve().parent.parent


def gumbel_moments(values):
    beta = max(float(np.std(values)) * math.sqrt(6.0) / math.pi, np.finfo(float).tiny)
    return float(np.mean(values)) - 0.5772 * beta, beta


def gumbel_p(llr, mu, beta):
    scores = np.asarray(llr, dtype=float)
    z = np.clip((scores - float(mu)) / max(float(beta), np.finfo(float).tiny), -700.0, 700.0)
    values = 1.0 - np.exp(-np.exp(-z))
    values[scores == 0] = 1.0
    return np.clip(values, 0.001, 1.0)


def metrics(reference_p, predicted_p):
    reference_p = np.asarray(reference_p, dtype=float)
    predicted_p = np.clip(np.asarray(predicted_p, dtype=float), 0.001, 1.0)
    hard = (reference_p >= 0.005) & (reference_p <= 0.2)
    result = {"n_all": len(reference_p), "n_hard": int(hard.sum())}
    for label, selection in (("all", np.ones(len(reference_p), dtype=bool)), ("hard", hard)):
        actual = reference_p[selection] < 0.05
        prediction = predicted_p[selection] < 0.05
        tp = int(np.sum(actual & prediction))
        fp = int(np.sum(~actual & prediction))
        fn = int(np.sum(actual & ~prediction))
        f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        union = int(np.sum(actual | prediction))
        result[f"{label}_f1"] = f1
        result[f"{label}_jaccard"] = tp / union if union else 1.0
        result[f"{label}_fp"] = fp
        result[f"{label}_fn"] = fn
        reference_t = -np.log10(reference_p[selection])
        predicted_t = -np.log10(predicted_p[selection])
        if label == "hard":
            result["hard_mae_log10_p"] = float(np.mean(np.abs(reference_t - predicted_t))) if hard.any() else np.nan
        rho = spearmanr(reference_t, predicted_t).statistic if len(reference_t) > 1 else np.nan
        result[f"{label}_spearman"] = float(rho) if np.isfinite(rho) else np.nan
    return result


def evaluate_dataset(name, datasets, models, gumbel_models, threshold):
    counties = load_dataset(name, "primary")
    population = counties["population"].to_numpy(dtype=float)
    counts = counties["deaths"].to_numpy(dtype=float)
    windows = pd.read_csv(ROOT / "data" / "processed" / f"{name}_c10_windows.csv.gz")
    llr = windows["llr"].to_numpy(dtype=float)
    reference_null = np.load(ROOT / "data" / "processed" / f"{name}_c10_ref_null_max.npy")
    sorted_reference = np.sort(reference_null)
    reference_p = (1.0 + len(sorted_reference) - np.searchsorted(sorted_reference, llr, side="left")) / (len(sorted_reference) + 1.0)
    windows["sample_stratum"] = "all"
    stats = dataset_features(population, counts, len(windows))
    features = add_model_features(windows, stats)
    predictions = {}
    xgb_t = np.asarray(models["xgboost"].predict(features[FEATURES].to_numpy(dtype=float)), dtype=float)
    predictions["xgboost_per_window"] = np.clip(np.power(10.0, -np.clip(xgb_t, -20.0, 20.0)), 0.001, 1.0)
    iso_t = np.asarray(models["isotonic"].predict(llr), dtype=float)
    predictions["isotonic"] = np.clip(np.power(10.0, -np.clip(iso_t, -20.0, 20.0)), 0.001, 1.0)
    dataset_features_vector = np.asarray([[
        math.log(float(stats["N"])), float(stats["n_counties"]), math.log(float(stats["n_windows"])),
        math.log(float(stats["total_pop"])), float(stats["pop_gini"]), float(stats["max_pop_share"]),
        float(stats["frac_counties_mu_below_1"]),
    ]], dtype=float)
    for family, model in gumbel_models.items():
        params = np.asarray(model.predict(dataset_features_vector), dtype=float)[0]
        predictions[f"gumbel_pred_{family}"] = gumbel_p(llr, params[0], float(np.exp(np.clip(params[1], -30.0, 30.0))))
    original_null = np.load(ROOT / "data" / "processed" / f"{name}_c10_null_max.npy")
    mu, beta = gumbel_moments(original_null)
    predictions["gumbel999"] = gumbel_p(llr, mu, beta)
    predictions["constant_threshold"] = np.where(llr > threshold, 0.001, 1.0)
    rows = []
    for method, values in predictions.items():
        rows.append({"dataset": name, "method": method, "seed": np.nan, "statistic": "single", **metrics(reference_p, values)})

    neighbors, _ = build_neighbors(counties["lat"].to_numpy(), counties["lon"].to_numpy())
    config = json.loads((ROOT / "config" / "scan.json").read_text(encoding="utf-8"))
    limits = window_limits(population, neighbors, config["pop_cap"])
    N = int(round(float(counts.sum())))
    repeats = {"gumbel99": [], "mc99": []}
    for seed in range(1000, 1020):
        simulated = null_max_distribution(population, N, neighbors, limits, 99, seed)
        sorted_simulated = np.sort(simulated)
        mc = (1.0 + len(sorted_simulated) - np.searchsorted(sorted_simulated, llr, side="left")) / 100.0
        rep_mu, rep_beta = gumbel_moments(simulated)
        repeats["mc99"].append((seed, mc))
        repeats["gumbel99"].append((seed, gumbel_p(llr, rep_mu, rep_beta)))
    for method, entries in repeats.items():
        method_values = []
        for seed, values in entries:
            row = {"dataset": name, "method": method, "seed": seed, "statistic": "repeat", **metrics(reference_p, values)}
            rows.append(row)
            method_values.append(row)
        metric_names = [key for key in method_values[0] if key not in {"dataset", "method", "seed", "statistic"}]
        for statistic in ("mean", "std", "min", "max"):
            row = {"dataset": name, "method": method, "seed": np.nan, "statistic": statistic}
            for metric in metric_names:
                values = np.asarray([item[metric] for item in method_values], dtype=float)
                row[metric] = float(getattr(pd.Series(values), statistic)())
            rows.append(row)
    return rows


def main():
    datasets = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
    models_dir = ROOT / "results" / "models"
    models = {"xgboost": joblib.load(models_dir / "xgboost.joblib"), "isotonic": joblib.load(models_dir / "isotonic.joblib")}
    gumbel_models = {"ridge": joblib.load(models_dir / "gumbel_pred_ridge.joblib"), "xgb": joblib.load(models_dir / "gumbel_pred_xgb.joblib")}
    split = json.loads((ROOT / "results" / "split.json").read_text(encoding="utf-8"))
    train_ids = set(int(value) for value in split["train"])
    metadata = [json.loads(line) for line in (ROOT / "data" / "synth" / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    threshold = float(np.mean([record["null_max_q95"] for record in metadata if int(record["id"]) in train_ids]))
    rows = []
    for dataset in datasets:
        rows.extend(evaluate_dataset(dataset["name"], datasets, models, gumbel_models, threshold))
    output = ROOT / "results" / "tables" / "real_vs_reference.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output} ({len(rows)} rows)")
    print(pd.DataFrame(rows).to_string(index=False, na_rep="NA", float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
