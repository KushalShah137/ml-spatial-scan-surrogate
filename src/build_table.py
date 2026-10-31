import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scan import dataset_features, load_dataset
from synth import sample_windows


ROOT = Path(__file__).resolve().parent.parent
FEATURES = [
    "llr", "log1p(llr)", "n_Z", "mu_Z", "log(mu_Z)", "n_Z/mu_Z",
    "radius_km", "pop_share", "k", "log(N)", "n_counties",
    "log(n_windows)", "log(total_pop)", "pop_gini", "max_pop_share",
    "frac_counties_mu_below_1",
]
OUTPUT_COLUMNS = ["id", "is_national", "has_cluster", "sigma", "sample_stratum", "p_value"] + FEATURES
FORBIDDEN = {"sigma", "has_cluster", "cluster_m", "cluster_rr", "observed_max_llr", "seconds_total", "seed", "id", "is_national", "p_value"}
assert not any(feature in FORBIDDEN or feature.startswith("null_max_") for feature in FEATURES)


def add_model_features(windows, stats):
    frame = windows.drop(columns=["n_counties"], errors="ignore").copy()
    required = {"llr", "n_Z", "mu_Z", "radius_km", "pop_share", "k", "p_value", "sample_stratum"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Window table is missing columns: {', '.join(sorted(missing))}")
    frame["log1p(llr)"] = np.log1p(frame["llr"].astype(float))
    frame["log(mu_Z)"] = np.log(frame["mu_Z"].astype(float))
    frame["n_Z/mu_Z"] = frame["n_Z"].astype(float) / frame["mu_Z"].astype(float)
    frame["log(N)"] = np.log(float(stats["N"]))
    frame["n_counties"] = int(stats["n_counties"])
    frame["log(n_windows)"] = np.log(float(stats["n_windows"]))
    frame["log(total_pop)"] = np.log(float(stats["total_pop"]))
    for column in ("pop_gini", "max_pop_share", "frac_counties_mu_below_1"):
        frame[column] = float(stats[column])
    return frame[FEATURES + ["p_value", "sample_stratum"]]


def assemble_table(windows, stats, dataset_id, is_national, has_cluster, sigma):
    features = add_model_features(windows, stats)
    features.insert(0, "sigma", sigma)
    features.insert(0, "has_cluster", has_cluster)
    features.insert(0, "is_national", bool(is_national))
    features.insert(0, "id", dataset_id)
    table = features[OUTPUT_COLUMNS]
    if not set(table.columns).isdisjoint(FORBIDDEN.intersection(FEATURES)):
        raise AssertionError("A prohibited field was included among model features.")
    return table


def read_meta(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {int(record["id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Synthetic metadata contains duplicate ids.")
    return by_id


def prepare_synthetic():
    synth_dir = ROOT / "data" / "synth"
    metadata = read_meta(synth_dir / "meta.jsonl")
    paths = sorted(synth_dir.glob("ds_*_windows.csv.gz"), key=lambda path: int(re.fullmatch(r"ds_(\d+)_windows\.csv\.gz", path.name).group(1)))
    path_ids = [int(re.fullmatch(r"ds_(\d+)_windows\.csv\.gz", path.name).group(1)) for path in paths]
    if set(path_ids) != set(metadata):
        raise ValueError("Synthetic window files and metadata ids do not match.")
    tables = []
    for path, dataset_id in zip(paths, path_ids):
        record = metadata[dataset_id]
        windows = pd.read_csv(path)
        expected_sample_size = min(int(record["n_windows"]), 3000)
        if len(windows) != expected_sample_size:
            raise ValueError(f"Sampled window count disagrees with metadata for id {dataset_id}.")
        stats = {key: record[key] for key in ("n_counties", "N", "n_windows", "total_pop", "pop_gini", "max_pop_share", "frac_counties_mu_below_1")}
        tables.append(assemble_table(windows, stats, dataset_id, record["is_national"], record["has_cluster"], record["sigma"]))
    return pd.concat(tables, ignore_index=True), metadata


def prepare_real():
    with (ROOT / "config" / "datasets.json").open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    tables = {}
    for dataset in datasets:
        name = dataset["name"]
        counties = load_dataset(name, "primary")
        windows_path = ROOT / "data" / "processed" / f"{name}_c10_windows.csv.gz"
        windows = pd.read_csv(windows_path)
        stats = dataset_features(counties["population"].to_numpy(), counties["deaths"].to_numpy(), len(windows))
        sampled = sample_windows(windows, np.random.default_rng(42))
        tables[name] = assemble_table(sampled, stats, name, True, np.nan, np.nan)
    return tables


def print_summary(label, table):
    print(f"{label}: rows={len(table)}")
    print(table[FEATURES].describe().T.to_string(float_format=lambda value: f"{value:.6g}"))


def main():
    output_dir = ROOT / "data" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic, metadata = prepare_synthetic()
    synthetic_path = output_dir / "synthetic_table.csv.gz"
    synthetic.to_csv(synthetic_path, index=False, compression="gzip")
    real_tables = prepare_real()
    print_summary("Synthetic", synthetic)
    synthetic_min = synthetic[FEATURES].min()
    synthetic_max = synthetic[FEATURES].max()
    for name, table in real_tables.items():
        output_path = output_dir / f"real_{name}_table.csv.gz"
        table.to_csv(output_path, index=False, compression="gzip")
        print_summary(f"Real {name}", table)
        outside = []
        for feature in FEATURES:
            low = float(table[feature].min())
            high = float(table[feature].max())
            if low < float(synthetic_min[feature]) or high > float(synthetic_max[feature]):
                outside.append(f"{feature}: real=[{low:g}, {high:g}], synthetic=[{synthetic_min[feature]:g}, {synthetic_max[feature]:g}]")
        print(f"Real {name} feature values outside synthetic range: {outside if outside else 'none'}")
    print(f"Saved synthetic table: {synthetic_path}")
    for name in real_tables:
        print(f"Saved real table: {output_dir / f'real_{name}_table.csv.gz'}")
    print(f"Synthetic datasets: {len(metadata)}")


if __name__ == "__main__":
    main()
