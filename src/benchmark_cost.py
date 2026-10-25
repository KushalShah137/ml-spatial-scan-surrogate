import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_table import FEATURES, add_model_features
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


def main():
    config = json.loads((ROOT / "config" / "scan.json").read_text(encoding="utf-8"))
    datasets = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
    metadata = [json.loads(line) for line in (ROOT / "data" / "synth" / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    training_seconds = float(sum(float(record["seconds_total"]) for record in metadata))
    model = joblib.load(ROOT / "results" / "models" / "xgboost.joblib")
    median_ratio = 0.165382
    calibrated_training_seconds = training_seconds * median_ratio
    print(f"Reusing median contention calibration ratio: {median_ratio:.6f}")
    print(f"Contended training cost: {training_seconds:.3f}s")
    print(f"Calibrated training cost: {calibrated_training_seconds:.3f}s")
    rows = []
    for dataset in datasets:
        name = dataset["name"]
        counties = load_dataset(name, "primary")
        population = counties["population"].to_numpy(dtype=float)
        counts = counties["deaths"].to_numpy(dtype=float)
        neighbors, distances = build_neighbors(counties["lat"].to_numpy(), counties["lon"].to_numpy())
        limits = window_limits(population, neighbors, config["pop_cap"])
        cases = int(round(float(counts.sum())))
        start = time.perf_counter()
        null_max_distribution(population, cases, neighbors, limits, 999, config["seed"])
        mc_seconds = time.perf_counter() - start
        start = time.perf_counter()
        null_99 = null_max_distribution(population, cases, neighbors, limits, 99, config["seed"])
        mc99_generation_seconds = time.perf_counter() - start

        windows = pd.read_csv(ROOT / "data" / "processed" / f"{name}_c10_windows.csv.gz")
        windows["sample_stratum"] = "all"
        mc99_processing_seconds, gumbel99_processing_seconds, _, _ = baseline_runtime(windows["llr"].to_numpy(), null_99)
        mc99_seconds = mc99_generation_seconds + mc99_processing_seconds
        gumbel99_seconds = mc99_generation_seconds + gumbel99_processing_seconds

        start = time.perf_counter()
        stats = dataset_features(population, counts, len(windows))
        features = add_model_features(windows, stats)
        feature_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(features[FEATURES].to_numpy(dtype=float))
        surrogate_seconds = time.perf_counter() - start
        surrogate_total = feature_seconds + surrogate_seconds
        competitor_times = {"mc999": mc_seconds, "mc99": mc99_seconds, "gumbel99": gumbel99_seconds}
        for competitor, competitor_seconds in competitor_times.items():
            saved_per_dataset = competitor_seconds - surrogate_total
            contended_break_even = training_seconds / saved_per_dataset if saved_per_dataset > 0 else float("inf")
            calibrated_break_even = calibrated_training_seconds / saved_per_dataset if saved_per_dataset > 0 else float("inf")
            rows.append({
                "dataset": name,
                "competitor": competitor,
                "competitor_seconds": competitor_seconds,
                "mc999_seconds": mc_seconds,
                "mc99_seconds": mc99_seconds,
                "gumbel99_seconds": gumbel99_seconds,
                "mc99_generation_seconds": mc99_generation_seconds,
                "mc99_postprocessing_seconds": mc99_processing_seconds,
                "gumbel99_generation_seconds": mc99_generation_seconds,
                "gumbel99_postprocessing_seconds": gumbel99_processing_seconds,
                "surrogate_feature_seconds": feature_seconds,
                "surrogate_prediction_seconds": surrogate_seconds,
                "surrogate_total_seconds": surrogate_total,
                "training_seconds_contended": training_seconds,
                "contention_median_ratio": median_ratio,
                "training_seconds_calibrated": calibrated_training_seconds,
                "break_even_datasets_contended": contended_break_even,
                "break_even_datasets_calibrated": calibrated_break_even,
                "n_windows": len(windows),
                "n_predictions": len(prediction),
            })
            print(f"{name} vs {competitor}: competitor={competitor_seconds:.6f}s; surrogate={surrogate_total:.6f}s; break-even contended={contended_break_even:.3f}, calibrated={calibrated_break_even:.3f}")
    output = ROOT / "results" / "tables" / "cost.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
