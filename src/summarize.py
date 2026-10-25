from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"


def print_table(title, frame, columns=None, sort_by=None):
    print(f"\n{title} (display rows={len(frame)})")
    selected = frame[columns] if columns else frame
    if sort_by and sort_by in selected.columns:
        selected = selected.sort_values(sort_by, kind="stable")
    with pd.option_context("display.max_columns", None, "display.width", 300):
        print(selected.to_string(index=False, na_rep="NA", float_format=lambda value: f"{value:.4f}"))


def main():
    print("CSV row counts")
    for path in sorted(TABLES.glob("*.csv")):
        print(f"{path.name}: {len(pd.read_csv(path))}")
    windows = pd.read_csv(TABLES / "window_metrics.csv")
    selected = windows[(windows["scope"] == "synthetic_test") & (windows["window_filter"] == "llr_positive") & (windows["sample_stratum"] == "all")]
    window_summary = selected.groupby("model", as_index=False).agg(
        mae_log10_p=("mae_t", "mean"),
        mae_p_0_01_to_0_10=("mae_p_0_01_to_0_10", "mean"),
        f1_at_0_05=("f1", "mean"),
    )
    print_table("Synthetic test windows (llr > 0; model averages across test sets)", window_summary, ["model", "mae_log10_p", "mae_p_0_01_to_0_10", "f1_at_0_05"], "model")

    dataset = pd.read_csv(TABLES / "dataset_metrics.csv")
    synthetic_dataset = dataset[dataset["scope"] == "synthetic_test"]
    dataset_summary = synthetic_dataset.groupby("model", as_index=False).agg(absolute_log10_p_error=("absolute_log10_p_error", "mean"))
    print_table("Synthetic test most-likely-cluster absolute log10 p error", dataset_summary, ["model", "absolute_log10_p_error"], "model")

    type1 = pd.read_csv(TABLES / "type1_power.csv")
    type1_summary = type1[["group", "method", "significant_count", "n_datasets", "fraction_significant", "ci95_low", "ci95_high"]]
    print_table("Type I error and power (Clopper-Pearson 95% intervals)", type1_summary, sort_by="group")

    quality = pd.read_csv(TABLES / "gumbel_regression_quality.csv")
    print_table("Gumbel parameter regression quality on test datasets", quality)

    learning = pd.read_csv(TABLES / "learning_curve.csv")
    print_table("Learning curve", learning)

    real = pd.read_csv(TABLES / "real_all_windows.csv")
    print_table("Real all-window evaluation", real, sort_by="dataset")

    repeats = pd.read_csv(TABLES / "baseline_repeats.csv")
    baseline_summary = repeats[repeats["row_type"] == "summary"].copy()
    surrogate = real[real["method"] == "xgboost_per_window"][["dataset", "hard_window_count", "hard_f1", "hard_mae_log10_p"]].copy()
    surrogate["row_type"] = "surrogate_reference"
    surrogate["statistic"] = "real_all_windows"
    surrogate["method"] = "xgboost_per_window"
    surrogate["seed"] = float("nan")
    surrogate["hard_window_count"] = surrogate["hard_window_count"].astype(int)
    surrogate = surrogate[["row_type", "statistic", "dataset", "method", "seed", "hard_window_count", "hard_f1", "hard_mae_log10_p"]]
    baseline_summary = baseline_summary[["row_type", "statistic", "dataset", "method", "seed", "hard_window_count", "hard_f1", "hard_mae_log10_p"]]
    baseline_display = pd.concat([baseline_summary, surrogate], ignore_index=True)
    print(f"Baseline repeat records: {int((repeats['row_type'] == 'repeat').sum())}; summary rows: {len(baseline_summary)}")
    print_table("Baseline repeats summary and existing surrogate comparison", baseline_display, sort_by="dataset")

    cost = pd.read_csv(TABLES / "cost.csv")
    print_table("Cost and break-even", cost, sort_by="dataset")

    cost_by_n = pd.read_csv(TABLES / "cost_by_n.csv")
    print_table("Cost by training dataset count", cost_by_n, sort_by="n_training_datasets")


if __name__ == "__main__":
    main()
