import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from scan import build_neighbors, dataset_features, load_dataset, null_max_distribution, window_limits


ROOT = Path(__file__).resolve().parent.parent


def baseline_runtime(llr, null_values):
    values = np.asarray(llr, dtype=float)
    first = np.asarray(null_values, dtype=float)[:99]
    start = time.perf_counter()
    ordered = np.sort(first)
    mc = (1.0 + len(ordered) - np.searchsorted(ordered, values, side="left")) / 100.0
    mc_seconds = time.perf_counter() - start
    start = time.perf_counter()
    deviation = float(np.std(first))
    beta = deviation * np.sqrt(6.0) / np.pi
    if beta <= 0:
        gumbel = np.ones(len(values), dtype=float)
    else:
        location = float(np.mean(first)) - 0.5772 * beta
        z = np.clip((values - location) / beta, -700.0, 700.0)
        gumbel = 1.0 - np.exp(-np.exp(-z))
    gumbel_seconds = time.perf_counter() - start
    return mc_seconds, gumbel_seconds, mc, gumbel


def gumbel_moments(values):
    beta = max(float(np.std(values)) * math.sqrt(6.0) / math.pi, np.finfo(float).tiny)
    return float(np.mean(values)) - 0.5772 * beta, beta


def dataset_vector(record):
    return [math.log(float(record["N"])), float(record["n_counties"]), math.log(float(record["n_windows"])), math.log(float(record["total_pop"])), float(record["pop_gini"]), float(record["max_pop_share"]), float(record["frac_counties_mu_below_1"])]


def load_or_refit_gumbel_models(metadata, split, tuning):
    model_dir = ROOT / "results" / "models"
    ridge_path = model_dir / "gumbel_pred_ridge.joblib"
    xgb_path = model_dir / "gumbel_pred_xgb.joblib"
    selected = {row["family"]: json.loads(row["parameters"]) for _, row in tuning[tuning["selected"]].iterrows()}
    ridge_alpha = float(selected["ridge"]["alpha"])
    xgb_depth = int(selected["xgb"]["max_depth"])
    ridge = joblib.load(ridge_path) if ridge_path.exists() else None
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    ridge_matches = ridge is not None and float(ridge.named_steps["ridge"].alpha) == ridge_alpha
    xgb_matches = xgb is not None and all(int(estimator.get_params()["max_depth"]) == xgb_depth for estimator in xgb.estimators_)
    if ridge_matches and xgb_matches:
        print(f"Saved gumbel_pred models verified: ridge alpha={ridge_alpha:g}; XGBoost max_depth={xgb_depth}")
        return ridge, xgb
    print(f"Refitting gumbel_pred models from saved split using ridge alpha={ridge_alpha:g} and XGBoost max_depth={xgb_depth}")
    records = {int(record["id"]): record for record in metadata}
    train_ids = [int(value) for value in split["train"]]
    x_train = np.asarray([dataset_vector(records[dataset_id]) for dataset_id in train_ids], dtype=float)
    y_train = []
    for dataset_id in train_ids:
        null_values = np.load(ROOT / "data" / "synth" / f"ds_{dataset_id}_null_max.npy")
        mu, beta = gumbel_moments(null_values)
        y_train.append([mu, math.log(beta)])
    y_train = np.asarray(y_train, dtype=float)
    if not ridge_matches:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
        ridge.fit(x_train, y_train)
        joblib.dump(ridge, ridge_path)
    if not xgb_matches:
        base = XGBRegressor(n_estimators=200, objective="reg:squarederror", tree_method="hist", max_depth=xgb_depth, learning_rate=0.05, n_jobs=1, random_state=42, subsample=0.9, colsample_bytree=1.0, verbosity=0)
        xgb = MultiOutputRegressor(base, n_jobs=1)
        xgb.fit(x_train, y_train)
        joblib.dump(xgb, xgb_path)
    print("Saved gumbel_pred regressors were refit and saved from the saved training split.")
    return ridge, xgb


def parameter_spread(metadata, training_ids):
    records = {int(record["id"]): record for record in metadata}
    mus = []
    betas = []
    for dataset_id in training_ids:
        values = np.load(ROOT / "data" / "synth" / f"ds_{dataset_id}_null_max.npy")
        mu, beta = gumbel_moments(values)
        mus.append(mu)
        betas.append(beta)
    mu_values = np.asarray(mus, dtype=float)
    beta_values = np.asarray(betas, dtype=float)
    print(f"Training Gumbel fits: {len(training_ids)} datasets")
    print(f"mu: mean={mu_values.mean():.6f}, std={mu_values.std():.6f}, min={mu_values.min():.6f}, max={mu_values.max():.6f}")
    print(f"beta: mean={beta_values.mean():.6f}, std={beta_values.std():.6f}, min={beta_values.min():.6f}, max={beta_values.max():.6f}, CV={beta_values.std() / beta_values.mean():.6f}")


def hard_metrics(true_p, predicted_p):
    true_p = np.asarray(true_p, dtype=float)
    predicted_p = np.clip(np.asarray(predicted_p, dtype=float), 0.001, 1.0)
    hard = (true_p >= 0.005) & (true_p <= 0.2)
    actual = true_p[hard] < 0.05
    predicted = predicted_p[hard] < 0.05
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    true_t = -np.log10(true_p[hard])
    pred_t = -np.log10(predicted_p[hard])
    mae = float(np.mean(np.abs(true_t - pred_t))) if hard.any() else np.nan
    return int(hard.sum()), f1, mae


def save_baseline_repeats(rows, output):
    frame = pd.DataFrame(rows)
    summaries = []
    if not frame.empty:
        for (dataset, method), group in frame[frame["row_type"] == "repeat"].groupby(["dataset", "method"], sort=False):
            for statistic in ("mean", "std", "min", "max"):
                summaries.append({"row_type": "summary", "statistic": statistic, "dataset": dataset, "method": method, "seed": np.nan, "hard_window_count": int(group["hard_window_count"].iloc[0]), "hard_f1": float(getattr(group["hard_f1"], statistic)()), "hard_mae_log10_p": float(getattr(group["hard_mae_log10_p"], statistic)())})
    summary_frame = pd.DataFrame(summaries)
    combined = pd.concat([frame, summary_frame], ignore_index=True) if len(summary_frame) else frame
    combined.to_csv(output, index=False)


def gumbel_prediction_time(model, counties, windows):
    population = counties["population"].to_numpy(dtype=float)
    counts = counties["deaths"].to_numpy(dtype=float)
    llr = windows["llr"].to_numpy(dtype=float)
    start = time.perf_counter()
    stats = dataset_features(population, counts, len(windows))
    features = np.asarray([[math.log(float(stats["N"])), float(stats["n_counties"]), math.log(float(stats["n_windows"])), math.log(float(stats["total_pop"])), float(stats["pop_gini"]), float(stats["max_pop_share"]), float(stats["frac_counties_mu_below_1"])]], dtype=float)
    parameters = np.asarray(model.predict(features), dtype=float)[0]
    mu = float(parameters[0])
    beta = max(float(np.exp(np.clip(parameters[1], -30.0, 30.0))), np.finfo(float).tiny)
    z = np.clip((llr - mu) / beta, -700.0, 700.0)
    p = 1.0 - np.exp(-np.exp(-z))
    p[llr == 0] = 1.0
    np.clip(p, 0.001, 1.0, out=p)
    return time.perf_counter() - start


def main():
    config = json.loads((ROOT / "config" / "scan.json").read_text(encoding="utf-8"))
    datasets = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
    metadata = [json.loads(line) for line in (ROOT / "data" / "synth" / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tables_dir = ROOT / "results" / "tables"
    split = json.loads((ROOT / "results" / "split.json").read_text(encoding="utf-8"))
    tuning = pd.read_csv(tables_dir / "gumbel_tuning.csv")
    ridge, xgb = load_or_refit_gumbel_models(metadata, split, tuning)
    train_ids = [int(value) for value in split["train"]]
    parameter_spread(metadata, train_ids)
    old_cost = pd.read_csv(tables_dir / "cost.csv")
    competitor_times = {(row["dataset"], row["competitor"]): float(row["competitor_seconds"]) for _, row in old_cost.iterrows()}
    cost_rows = []
    baseline_rows = []
    for dataset in datasets:
        name = dataset["name"]
        counties = load_dataset(name, "primary")
        population = counties["population"].to_numpy(dtype=float)
        counts = counties["deaths"].to_numpy(dtype=float)
        neighbors, _ = build_neighbors(counties["lat"].to_numpy(), counties["lon"].to_numpy())
        limits = window_limits(population, neighbors, config["pop_cap"])
        cases = int(round(float(counts.sum())))
        windows = pd.read_csv(ROOT / "data" / "processed" / f"{name}_c10_windows.csv.gz")
        ridge_times = [gumbel_prediction_time(ridge, counties, windows) for _ in range(5)]
        xgb_times = [gumbel_prediction_time(xgb, counties, windows) for _ in range(5)]
        ridge_time = float(np.median(ridge_times))
        xgb_time = float(np.median(xgb_times))
        cost_rows.append({"dataset": name, "gumbel_pred_ridge_seconds_median5": ridge_time, "gumbel_pred_ridge_seconds_mean5": float(np.mean(ridge_times)), "gumbel_pred_xgb_seconds_median5": xgb_time, "gumbel_pred_xgb_seconds_mean5": float(np.mean(xgb_times)), "ridge_run_seconds": json.dumps(ridge_times), "xgb_run_seconds": json.dumps(xgb_times), "n_windows": len(windows)})
        print(f"{name}: gumbel_pred_ridge median of 5={ridge_time:.6f}s; gumbel_pred_xgb median of 5={xgb_time:.6f}s")

        true_p = windows["p_value"].to_numpy(dtype=float)
        llr = windows["llr"].to_numpy(dtype=float)
        for seed in range(1000, 1020):
            null_values = null_max_distribution(population, cases, neighbors, limits, 99, seed)
            first = np.sort(null_values)
            mc99 = (1.0 + len(first) - np.searchsorted(first, llr, side="left")) / 100.0
            mu, beta = gumbel_moments(null_values)
            z = np.clip((llr - mu) / beta, -700.0, 700.0)
            gumbel99 = 1.0 - np.exp(-np.exp(-z))
            gumbel99[llr == 0] = 1.0
            gumbel99 = np.clip(gumbel99, 0.001, 1.0)
            for method, p_values in (("mc99", mc99), ("gumbel99", gumbel99)):
                hard_count, f1, mae = hard_metrics(true_p, p_values)
                baseline_rows.append({"row_type": "repeat", "statistic": "", "dataset": name, "method": method, "seed": seed, "hard_window_count": hard_count, "hard_f1": f1, "hard_mae_log10_p": mae})
            if (seed - 999) % 5 == 0:
                print(f"{name}: completed repeat {seed - 999}/20")
        baseline_output = tables_dir / "baseline_repeats.csv"
        save_baseline_repeats(baseline_rows, baseline_output)
        print(f"Saved baseline repeats through {name}: {baseline_output}")

    output = tables_dir / "cost.csv"
    pd.DataFrame(cost_rows).to_csv(output, index=False)
    mean_seconds = float(np.mean([float(record["seconds_total"]) for record in metadata]))
    ratio = 0.165382
    old_competitors = ("mc999", "mc99", "gumbel99")
    cost_by_n_rows = []
    for n in (25, 50, 100, 200, 315, 450):
        contended = n * mean_seconds
        calibrated = contended * ratio
        for dataset in datasets:
            name = dataset["name"]
            measured = next(row for row in cost_rows if row["dataset"] == name)
            ridge_time = measured["gumbel_pred_ridge_seconds_median5"]
            for competitor in old_competitors:
                competitor_seconds = competitor_times[(name, competitor)]
                per_dataset_saving = competitor_seconds - ridge_time
                contended_break_even = contended / per_dataset_saving if per_dataset_saving > 0 else float("inf")
                calibrated_break_even = calibrated / per_dataset_saving if per_dataset_saving > 0 else float("inf")
                cost_by_n_rows.append({"dataset": name, "n_training_datasets": n, "mean_seconds_total_per_dataset": mean_seconds, "competitor": competitor, "competitor_seconds": competitor_seconds, "gumbel_pred_ridge_seconds": ridge_time, "training_seconds_contended": contended, "training_seconds_calibrated": calibrated, "break_even_datasets_contended": contended_break_even, "break_even_datasets_calibrated": calibrated_break_even})
    pd.DataFrame(cost_by_n_rows).to_csv(tables_dir / "cost_by_n.csv", index=False)
    print(f"Saved: {output}")
    print(f"Saved: {tables_dir / 'cost_by_n.csv'}")
    print(f"Baseline repeats saved: {tables_dir / 'baseline_repeats.csv'}")


if __name__ == "__main__":
    main()
