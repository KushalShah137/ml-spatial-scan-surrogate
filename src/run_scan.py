import argparse
import json
import time
from pathlib import Path

import numpy as np

from data_prep import REPO_ROOT, load_datasets
from scan import build_neighbors, load_dataset, most_likely_cluster, null_max_distribution, observed_llr, window_limits, window_p_values, window_table


def run_one(dataset):
    name = dataset["name"]
    started = time.perf_counter()
    with (REPO_ROOT / "config" / "scan.json").open(encoding="utf-8") as handle:
        scan_config = json.load(handle)
    pop_cap = float(scan_config["pop_cap"])
    replicates = int(scan_config["replicates"])
    seed = int(scan_config["seed"])
    frame = load_dataset(name, "primary")
    population = frame["population"].to_numpy(dtype=float)
    counts = frame["deaths"].to_numpy(dtype=float)
    total_cases = int(round(counts.sum()))
    neighbors, distances = build_neighbors(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    limits = window_limits(population, neighbors, pop_cap)
    null_started = time.perf_counter()
    null_max = null_max_distribution(population, total_cases, neighbors, limits, replicates, seed)
    null_elapsed = time.perf_counter() - null_started
    cap_tag = f"c{int(round(pop_cap * 100))}"
    null_path = REPO_ROOT / "data" / "processed" / f"{name}_{cap_tag}_null_max.npy"
    np.save(null_path, null_max)
    llr = observed_llr(counts, population, neighbors, limits)
    pvalues = window_p_values(llr, null_max)
    table = window_table(llr, pvalues, counts, population, neighbors, distances, limits, frame["fips"].to_numpy())
    windows_path = REPO_ROOT / "data" / "processed" / f"{name}_{cap_tag}_windows.csv.gz"
    table.to_csv(windows_path, index=False, compression="gzip")
    center, k, maximum = most_likely_cluster(llr)
    center_fips = str(frame.iloc[center]["fips"])
    cluster = table.loc[(table["center_fips"] == center_fips) & (table["k"] == k)].iloc[0]
    county = frame.iloc[center]
    elapsed = time.perf_counter() - started
    significant = int((table["p_value"] < 0.05).sum())
    print(f"Dataset: {name}")
    print(f"Window count: {len(table)}")
    print(f"Seconds per replicate: {null_elapsed / replicates:.6f}")
    print(f"Total runtime: {elapsed:.3f} seconds")
    print(
        "Most likely cluster: "
        f"county={county['county']}, state={county['state']}, k={k}, "
        f"n_Z={cluster['n_Z']:g}, mu_Z={cluster['mu_Z']:.6f}, "
        f"llr={maximum:.9f}, p={cluster['p_value']:.6f}"
    )
    print(f"Windows with p < 0.05: {significant}")
    print(f"Saved windows: {windows_path}")
    print(f"Saved null maxima: {null_path}")


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--name")
    args = parser.parse_args()
    with (REPO_ROOT / "config" / "datasets.json").open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    selected = datasets if args.all else [item for item in datasets if item["name"] == args.name]
    if not selected:
        parser.error(f"Unknown dataset name {args.name!r}.")
    for dataset in selected:
        run_one(dataset)


if __name__ == "__main__":
    main()
