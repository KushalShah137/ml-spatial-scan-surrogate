import json
import math
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, ScalarFormatter

from scan import build_neighbors, load_dataset, most_likely_cluster, observed_llr, window_limits
from display_names import DATASET_DISPLAY_NAMES, METHOD_DISPLAY_NAMES


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "figures" / "paper"
REAL_SOURCE = "original"
SINGLE = 3.5
DOUBLE = 7.16
DISPLAY = METHOD_DISPLAY_NAMES
COLORS = {
    "gumbel_pred_xgb": "#0072B2",
    "gumbel_pred_ridge": "#56B4E9",
    "xgboost_per_window": "#D55E00",
    "xgboost": "#D55E00",
    "gumbel99": "#009E73",
    "gumbel999": "#66C2A5",
    "mc99": "#CC79A7",
    "mc999": "#E78AC3",
    "isotonic": "#E69F00",
    "mlr": "#999999",
    "random_forest": "#F0E442",
    "svr": "#332288",
    "mlp": "#AA4499",
    "constant_threshold": "#666666",
}


def setup_plotting():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    })


def save_figure(name, figure, width):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.set_size_inches(width, figure.get_size_inches()[1])
    figure.savefig(OUTPUT / f"{name}.pdf", format="pdf")
    figure.savefig(OUTPUT / f"{name}.png", format="png", dpi=300)
    plt.close(figure)


def label(method):
    return DISPLAY.get(method, method.replace("_", " "))


def color(method):
    return COLORS.get(method, "#777777")


def make_null_calibration():
    frame = pd.read_csv(TABLES / "null_calibration.csv")
    values = frame["gumbel_pred_xgb"].dropna().to_numpy(dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    axes[0].hist(values, bins=np.linspace(0.0, 1.0, 21), color=color("gumbel_pred_xgb"), edgecolor="white", linewidth=0.35)
    axes[0].axhline(len(values) / 20.0, color="#222222", linestyle="--", linewidth=0.9, label="Uniform")
    axes[0].set(xlim=(0, 1), ylim=(0, 135), xlabel="Predicted p-value", ylabel="Null datasets", title="P-value calibration")
    axes[0].legend(frameon=False, loc="upper right")
    ordered = np.sort(values)
    uniform_quantiles = (np.arange(1, len(ordered) + 1) - 0.5) / len(ordered)
    axes[1].scatter(uniform_quantiles, ordered, s=5, alpha=0.45, color=color("gumbel_pred_xgb"), linewidths=0)
    axes[1].plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=0.9)
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Uniform quantile", ylabel="Predicted p-value quantile", title="Uniform QQ plot")
    figure.tight_layout(w_pad=1.1, rect=(0, 0.2, 1, 1))
    save_figure("fig_null_calibration", figure, DOUBLE)


def make_learning_curve():
    curve = pd.read_csv(TABLES / "learning_curve.csv")
    curve["plot_method"] = curve["method"].replace({"xgboost": "xgboost_per_window"})
    figure, axis = plt.subplots(figsize=(SINGLE, 3.25))
    for method in ("gumbel_pred_xgb", "xgboost_per_window"):
        group = curve[curve["plot_method"] == method]
        stats = group.groupby("n_train_datasets")["band_mae_p"].agg(["mean", "min", "max"]).sort_index()
        x = stats.index.to_numpy(dtype=float)
        axis.plot(x, stats["mean"], marker="o", markersize=3, linewidth=1.2, color=color(method), label=label(method))
        axis.fill_between(x, stats["min"].to_numpy(), stats["max"].to_numpy(), color=color(method), alpha=0.18, linewidth=0)
    window = pd.read_csv(TABLES / "window_metrics.csv")
    reference = window[(window["scope"] == "synthetic_test") & (window["window_filter"] == "llr_positive") & (window["sample_stratum"] == "all")]
    for method, style in (("gumbel99", ":"), ("gumbel999", "--")):
        value = float(reference.loc[reference["model"] == method, "mae_p_0_01_to_0_10"].mean())
        axis.axhline(value, color=color(method), linestyle=style, linewidth=1.0, label=f"{label(method)} reference")
    axis.set_xscale("log")
    axis.set_xticks([25, 50, 100, 200, 315])
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set(xlabel="Training datasets", ylabel="Band MAE on p", title="Learning curve")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    save_figure("fig_learning_curve", figure, SINGLE)


def make_pred_vs_true():
    frame = pd.read_csv(ROOT / "results" / "predictions" / "synthetic_test_predictions.csv.gz")
    frame = frame[frame["llr"] > 0].copy()
    true_t = frame["true_t"].to_numpy(dtype=float)
    predicted_t = -np.log10(np.clip(frame["pred_p_gumbel_pred_xgb"].to_numpy(dtype=float), 0.001, 1.0))
    hard = (frame["p_value"].to_numpy(dtype=float) >= 0.005) & (frame["p_value"].to_numpy(dtype=float) <= 0.2)
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.05), constrained_layout=True)
    figure.set_constrained_layout_pads(w_pad=0.02, h_pad=0.01, wspace=0.05, hspace=0.02)
    for axis, selection, title in ((axes[0], np.ones(len(frame), dtype=bool), "All positive-LLR windows"), (axes[1], hard, "True p in [0.005, 0.2]")):
        mesh = axis.hexbin(true_t[selection], predicted_t[selection], gridsize=45, bins="log", mincnt=1, cmap="viridis", linewidths=0)
        bar = figure.colorbar(mesh, ax=axis, pad=0.02)
        bar.set_label("Windows")
        maximum = max(float(np.max(true_t[selection])), float(np.max(predicted_t[selection])), 1.0)
        axis.plot([0, maximum], [0, maximum], color="#D55E00", linewidth=0.9, linestyle="--")
        axis.set(xlabel="True −log10 p", ylabel="Predicted −log10 p", title=title)
    axes[1].set_xlim(0.7, 2.3)
    axes[1].set_ylim(0.6, 3.1)
    figure.suptitle("Synthetic test windows, llr > 0", y=0.99)
    save_figure("fig_pred_vs_true", figure, DOUBLE)


def make_method_comparison():
    windows = pd.read_csv(TABLES / "window_metrics.csv")
    selected = windows[(windows["scope"] == "synthetic_test") & (windows["window_filter"] == "llr_positive") & (windows["sample_stratum"] == "all")]
    band = selected.groupby("model")["mae_p_0_01_to_0_10"].mean()
    dataset = pd.read_csv(TABLES / "dataset_metrics.csv")
    cluster = dataset[dataset["scope"] == "synthetic_test"].groupby("model")["absolute_log10_p_error"].mean()
    methods = [method for method in band.index if method in cluster.index]
    display_names = {
        "xgboost_no_dataset_features": METHOD_DISPLAY_NAMES["xgboost_no_dataset_features"],
        "xgboost_monotone_llr": METHOD_DISPLAY_NAMES["xgboost_monotone_llr"],
    }
    replicates = {"gumbel99": 99, "mc99": 99, "gumbel999": 999}
    rows = []
    for method in methods:
        display_name = display_names.get(method, label(method))
        if method in replicates:
            display_name += f" ({replicates[method]} reps)"
        rows.append({
            "method": method,
            "display_name": display_name,
            "band_mae": float(band[method]),
            "cluster_error": float(cluster[method]),
            "replicates": replicates.get(method, 0),
        })
    plot_data = pd.DataFrame(rows, columns=["method", "display_name", "band_mae", "cluster_error", "replicates"])
    plot_data = plot_data.sort_values("band_mae", ascending=True, kind="stable").reset_index(drop=True)
    band_source = selected.groupby("model")["mae_p_0_01_to_0_10"].mean()
    cluster_source = dataset[dataset["scope"] == "synthetic_test"].groupby("model")["absolute_log10_p_error"].mean()
    for row in plot_data.itertuples(index=False):
        assert np.isclose(row.band_mae, float(band_source[row.method]))
        assert np.isclose(row.cluster_error, float(cluster_source[row.method]))
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE, max(3.15, 0.25 * len(plot_data))), sharey=True)
    positions = np.arange(len(plot_data))
    ordered_methods = plot_data["method"].tolist()
    axes[0].barh(positions, plot_data["band_mae"], color=[color(method) for method in ordered_methods], height=0.72)
    axes[1].barh(positions, plot_data["cluster_error"], color=[color(method) for method in ordered_methods], height=0.72)
    axes[0].set(title="Probability band error", xlabel="MAE on p, true p ∈ [0.01, 0.10]")
    axes[1].set(title="Most likely cluster error", xlabel="Absolute error in log10 p")
    axes[1].set_xscale("log")
    axes[0].set_yticks(positions, plot_data["display_name"])
    axes[1].tick_params(axis="y", labelleft=False, left=False)
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
    axes[0].invert_yaxis()
    figure.tight_layout(w_pad=1.25)
    save_figure("fig_method_comparison", figure, DOUBLE)
    return plot_data


def make_runtime():
    cost = pd.read_csv(TABLES / "cost.csv")
    by_n = pd.read_csv(TABLES / "cost_by_n.csv")
    reference = by_n[by_n["n_training_datasets"] == 450]
    values = {
        "mc999": float(reference.loc[reference["competitor"] == "mc999", "competitor_seconds"].mean()),
        "mc99": float(reference.loc[reference["competitor"] == "mc99", "competitor_seconds"].mean()),
        "gumbel99": float(reference.loc[reference["competitor"] == "gumbel99", "competitor_seconds"].mean()),
        "gumbel_pred_xgb": float(cost["gumbel_pred_xgb_seconds_median5"].mean()),
    }
    methods = list(values)
    figure, axis = plt.subplots(figsize=(SINGLE, 2.65))
    bars = axis.bar(np.arange(len(methods)), [values[method] for method in methods], color=[color(method) for method in methods], width=0.68)
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(methods)), [label(method) for method in methods], rotation=25, ha="right")
    axis.set_ylabel("Seconds per dataset (log scale)")
    axis.set_title("Time to obtain p-values for 1.34M windows")
    axis.grid(axis="y", which="major", alpha=0.18)
    axis.grid(axis="y", which="minor", visible=False)
    axis.bar_label(bars, labels=["139 s", "13.8 s", "13.8 s", "0.037 s"], fontsize=6, padding=2)
    figure.tight_layout()
    save_figure("fig_runtime", figure, SINGLE)


def make_break_even():
    frame = pd.read_csv(TABLES / "cost_by_n.csv")
    frame = frame[frame["competitor"].isin(["mc999", "gumbel99"])]
    figure, axis = plt.subplots(figsize=(DOUBLE, 3.0))
    series = [
        ("mc999", "break_even_datasets_contended", "MC-999, contended", "#CC79A7", "-"),
        ("mc999", "break_even_datasets_calibrated", "MC-999, calibrated", "#CC79A7", "--"),
        ("gumbel99", "break_even_datasets_contended", "Gumbel-99, contended", "#009E73", "-"),
        ("gumbel99", "break_even_datasets_calibrated", "Gumbel-99, calibrated", "#009E73", "--"),
    ]
    for competitor, column, label_text, shade, style in series:
        group = frame[frame["competitor"] == competitor].groupby("n_training_datasets")[column].mean().sort_index()
        axis.plot(group.index, group.values, marker="o", markersize=3, linewidth=1.2, linestyle=style, color=shade, label=label_text)
    axis.set(xlabel="Training datasets", ylabel="Break-even datasets", title="Break-even by training-set size")
    axis.set_xticks([25, 50, 100, 200, 315, 450])
    axis.axvline(315, color="#333333", linestyle=":", linewidth=0.9, label="Evaluated model")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save_figure("fig_break_even", figure, DOUBLE)


def real_source_data():
    if REAL_SOURCE == "original":
        real = pd.read_csv(TABLES / "real_all_windows.csv")
        baseline = pd.read_csv(TABLES / "baseline_repeats.csv")
        baseline = baseline[(baseline["row_type"] == "repeat") & baseline["method"].isin(["gumbel99", "mc99"])]
        return real, baseline, "hard_f1", "hard_mae_log10_p"
    path = TABLES / "real_vs_reference.csv"
    if not path.exists():
        raise FileNotFoundError(f"Reference metrics are unavailable: {path}")
    real = pd.read_csv(path)
    required = {"dataset", "method", "statistic", "hard_f1", "hard_mae_log10_p"}
    missing = required - set(real.columns)
    if missing:
        raise ValueError(f"Reference metrics are missing columns: {sorted(missing)}")
    baseline = real[real["method"].isin(["gumbel99", "mc99"]) & real["seed"].notna()]
    real = real[~real.index.isin(baseline.index)]
    return real, baseline, "hard_f1", "hard_mae_log10_p"


def make_real_hard_f1():
    real, baseline, f1_column, _ = real_source_data()
    datasets = list(real["dataset"].drop_duplicates())
    if REAL_SOURCE == "reference":
        datasets = [name for name in ("mesothelioma", "pneumoconioses", "als") if name in datasets]
    methods = ["xgboost_per_window", "isotonic", "gumbel_pred_xgb", "gumbel_pred_ridge", "gumbel999", "gumbel99", "mc99", "constant_threshold"]
    labels = [label(method) for method in methods]
    figure, axis = plt.subplots(figsize=(DOUBLE, 4.0))
    centers = np.arange(len(datasets), dtype=float)
    width = 0.82 / len(methods)
    for method_index, method in enumerate(methods):
        means = []
        lows = []
        highs = []
        for dataset in datasets:
            if method in ("gumbel99", "mc99"):
                group = baseline[(baseline["dataset"] == dataset) & (baseline["method"] == method)]
                values = group[f1_column].to_numpy(dtype=float)
                means.append(float(np.mean(values)))
                lows.append(float(np.quantile(values, 0.1)))
                highs.append(float(np.quantile(values, 0.9)))
            else:
                row = real[(real["dataset"] == dataset) & (real["method"] == method)].iloc[0]
                means.append(float(row[f1_column]))
                lows.append(float(row[f1_column]))
                highs.append(float(row[f1_column]))
        offsets = centers - 0.41 + width * (method_index + 0.5)
        axis.bar(offsets, means, width=width, color=color(method), label=labels[method_index], linewidth=0)
        if method in ("gumbel99", "mc99"):
            axis.errorbar(offsets, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)], fmt="none", ecolor="#222222", capsize=1.4, linewidth=0.7)
        for x, value in zip(offsets, means):
            digits = 4 if value >= 0.9995 else 3 if value >= 0.99 else 2
            axis.text(x, value + 0.004, f"{value:.{digits}f}", ha="center", va="bottom", fontsize=5, rotation=90 if width < 0.09 else 0, color="#222222", clip_on=False)
    axis.set_xticks(centers, [DATASET_DISPLAY_NAMES.get(dataset, dataset) for dataset in datasets])
    axis.set_ylim(0.6, 1.02)
    axis.set_ylabel("Hard-subset F1")
    axis.set_title(f"Real-dataset hard-subset performance ({REAL_SOURCE})", pad=6)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.015))
    axis.grid(axis="y", alpha=0.18)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure("fig_real_hard_f1", figure, DOUBLE)


def make_clusters():
    datasets = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
    with (ROOT / "config" / "scan.json").open(encoding="utf-8") as handle:
        cap = float(json.load(handle)["pop_cap"])
    figure, axes = plt.subplots(1, len(datasets), figsize=(DOUBLE, 2.8), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        name = dataset["name"]
        frame = load_dataset(name, "primary")
        population = frame["population"].to_numpy(dtype=float)
        counts = frame["deaths"].to_numpy(dtype=float)
        neighbors, _ = build_neighbors(frame["lat"].to_numpy(), frame["lon"].to_numpy())
        limits = window_limits(population, neighbors, cap)
        scores = observed_llr(counts, population, neighbors, limits)
        center, k, _ = most_likely_cluster(scores)
        indices = neighbors[center, :k].astype(int)
        highlighted = np.zeros(len(frame), dtype=bool)
        highlighted[indices] = True
        axis.scatter(frame["lon"], frame["lat"], s=8, color="#D0D0D0", alpha=0.7, linewidths=0)
        axis.scatter(frame.loc[highlighted, "lon"], frame.loc[highlighted, "lat"], s=14, color="#D55E00", alpha=0.85, linewidths=0)
        axis.scatter([frame.iloc[center]["lon"]], [frame.iloc[center]["lat"]], s=36, marker="*", color="#0072B2", edgecolors="white", linewidths=0.4, zorder=3)
        axis.set(title=f"{DATASET_DISPLAY_NAMES.get(name, name)} (k={k})", xlabel="Longitude", ylabel="Latitude")
        axis.set_xlim(-125, -66)
        axis.set_ylim(24, 51)
        axis.set_aspect(1 / math.cos(math.radians(38)), adjustable="box")
        center_lon = float(frame.iloc[center]["lon"])
        center_lat = float(frame.iloc[center]["lat"])
        text_x = min(max(center_lon - 5 if center_lon > -96 else center_lon + 5, -124), -67)
        text_y = min(max(center_lat + 2.0, 25), 49.5)
        axis.annotate(str(frame.iloc[center]["county"]), (center_lon, center_lat), xytext=(text_x, text_y), textcoords="data", ha="right" if center_lon > -96 else "left", fontsize=6, annotation_clip=True, arrowprops={"arrowstyle": "-", "linewidth": 0.4}, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5}, clip_on=True)
    figure.tight_layout(w_pad=0.8)
    save_figure("fig_clusters", figure, DOUBLE)


def main():
    global REAL_SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-source", choices=("original", "reference"), default="original")
    REAL_SOURCE = parser.parse_args().real_source
    setup_plotting()
    make_null_calibration()
    make_learning_curve()
    make_pred_vs_true()
    make_method_comparison()
    make_runtime()
    make_break_even()
    make_real_hard_f1()
    make_clusters()
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
