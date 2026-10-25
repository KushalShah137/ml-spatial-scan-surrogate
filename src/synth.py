import argparse
import concurrent.futures
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scan import build_neighbors, load_dataset, most_likely_cluster, null_max_distribution, observed_llr, window_limits, window_p_values, window_table


def repository_root():
    return Path(__file__).resolve().parent.parent


def peak_working_set_bytes():
    if os.name != "nt":
        try:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(value if os.uname().sysname == "Darwin" else value * 1024)
        except Exception:
            return None
    try:
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:
        return None


def gini_coefficient(values):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    count = len(sorted_values)
    total = sorted_values.sum()
    if not count or total <= 0:
        return 0.0
    ranks = np.arange(1, count + 1, dtype=float)
    return float((2.0 * np.dot(ranks, sorted_values) / (count * total)) - (count + 1.0) / count)


def sample_windows(table, random):
    top_count = min(500, len(table))
    ranked = table.sort_values("llr", ascending=False, kind="stable")
    top = ranked.iloc[:top_count].copy()
    top["sample_stratum"] = "top"
    rest = table.drop(index=top.index)
    random_count = min(2500, len(rest))
    if random_count:
        chosen = random.choice(len(rest), size=random_count, replace=False)
        sampled = rest.iloc[np.sort(chosen)].copy()
        sampled["sample_stratum"] = "random"
        return pd.concat([top, sampled], ignore_index=True)
    return top.reset_index(drop=True)


def generate_dataset(dataset_id, pop_cap, replicates, scan_seed, is_national=False):
    started = time.perf_counter()
    root = repository_root()
    output_dir = root / "data" / "synth"
    parts_dir = output_dir / "meta_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / f"ds_{dataset_id}_windows.csv.gz"
    null_path = output_dir / f"ds_{dataset_id}_null_max.npy"
    meta_path = parts_dir / f"ds_{dataset_id}.json"
    if windows_path.exists() and null_path.exists():
        if not meta_path.exists():
            raise FileNotFoundError(f"Existing outputs for id {dataset_id} have no metadata part to merge.")
        return {"meta": json.loads(meta_path.read_text(encoding="utf-8")), "pid": os.getpid(), "peak_bytes": peak_working_set_bytes(), "skipped": True}

    seed = 100000 + int(dataset_id)
    random = np.random.default_rng(seed)
    with (root / "config" / "datasets.json").open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    population_source = load_dataset(datasets[0]["name"], "primary")
    centroid_path = root / "data" / "external" / "census_county_centroids.csv"
    centroids = pd.read_csv(centroid_path, dtype={"fips": str})
    geography = centroids.merge(population_source[["fips", "population"]], on="fips", how="inner", validate="one_to_one")
    geography = geography.reset_index(drop=True)
    all_neighbors, _ = build_neighbors(geography["lat"].to_numpy(), geography["lon"].to_numpy())

    sampled_region_size = int(np.rint(np.exp(random.uniform(np.log(150), np.log(len(geography))))))
    sampled_region_size = min(max(sampled_region_size, 150), len(geography))
    sampled_region_center = int(random.integers(len(geography)))
    if is_national:
        region_size = len(geography)
        region_indices = np.arange(region_size, dtype=int)
    else:
        region_size = sampled_region_size
        region_indices = all_neighbors[sampled_region_center, :region_size].astype(int)
    region = geography.iloc[region_indices].reset_index(drop=True)
    population = region["population"].to_numpy(dtype=float)
    lat = region["lat"].to_numpy(dtype=float)
    lon = region["lon"].to_numpy(dtype=float)
    neighbors, distances = build_neighbors(lat, lon)

    total_cases = int(np.rint(np.exp(random.uniform(np.log(200), np.log(200000)))))
    risk = np.ones(region_size, dtype=float)
    has_heterogeneity = bool(random.random() < 0.3)
    sigma = float(random.uniform(0.1, 0.5)) if has_heterogeneity else 0.0
    if has_heterogeneity:
        risk *= random.lognormal(0.0, sigma, size=region_size)
    has_cluster = bool(random.random() < 0.5)
    cluster_m = 0
    cluster_rr = 0.0
    if has_cluster:
        cluster_center = int(random.integers(region_size))
        cluster_m = int(random.integers(1, 41))
        cluster_rr = float(random.uniform(1.2, 4.0))
        risk[neighbors[cluster_center, :cluster_m].astype(int)] *= cluster_rr
    probabilities = population * risk
    probabilities = probabilities / probabilities.sum()
    counts = random.multinomial(total_cases, probabilities)
    limits = window_limits(population, neighbors, pop_cap)
    null_max = null_max_distribution(population, total_cases, neighbors, limits, replicates, seed + int(scan_seed))
    llr = observed_llr(counts, population, neighbors, limits)
    pvalues = window_p_values(llr, null_max)
    table = window_table(llr, pvalues, counts, population, neighbors, distances, limits, region["fips"].to_numpy())
    selected = sample_windows(table, random)
    observed_max = float(np.max(llr))
    quantiles = np.quantile(null_max, [0.5, 0.9, 0.95, 0.99])
    metadata = {
        "id": int(dataset_id),
        "seed": int(seed),
        "is_national": bool(is_national),
        "n_counties": int(region_size),
        "N": int(total_cases),
        "n_windows": int(len(table)),
        "total_pop": float(population.sum()),
        "pop_gini": gini_coefficient(population),
        "max_pop_share": float(population.max() / population.sum()),
        "frac_counties_mu_below_1": float(np.mean(total_cases * population / population.sum() < 1.0)),
        "sigma": sigma,
        "has_cluster": has_cluster,
        "cluster_m": cluster_m,
        "cluster_rr": cluster_rr,
        "observed_max_llr": observed_max,
        "null_max_mean": float(np.mean(null_max)),
        "null_max_std": float(np.std(null_max)),
        "null_max_q50": float(quantiles[0]),
        "null_max_q90": float(quantiles[1]),
        "null_max_q95": float(quantiles[2]),
        "null_max_q99": float(quantiles[3]),
        "seconds_total": 0.0,
    }
    windows_temp = output_dir / f"ds_{dataset_id}_windows.tmp.csv.gz"
    null_temp = output_dir / f"ds_{dataset_id}_null_max.tmp.npy"
    metadata_temp = parts_dir / f"ds_{dataset_id}.tmp.json"
    selected.to_csv(windows_temp, index=False, compression="gzip")
    with null_temp.open("wb") as handle:
        np.save(handle, null_max)
    metadata["seconds_total"] = float(time.perf_counter() - started)
    metadata_temp.write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(metadata_temp, meta_path)
    os.replace(windows_temp, windows_path)
    os.replace(null_temp, null_path)
    return {"meta": metadata, "pid": os.getpid(), "peak_bytes": peak_working_set_bytes(), "skipped": False}


def merge_metadata(output_dir):
    parts_dir = output_dir / "meta_parts"
    records = [json.loads(path.read_text(encoding="utf-8")) for path in parts_dir.glob("ds_*.json") if ".tmp." not in path.name]
    records.sort(key=lambda record: record["id"])
    target = output_dir / "meta.jsonl"
    temporary = output_dir / "meta.tmp.jsonl"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temporary, target)


def append_progress(output_dir, metadata):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": int(metadata["id"]),
        "n_counties": int(metadata["n_counties"]),
        "N": int(metadata["N"]),
        "seconds": float(metadata["seconds_total"]),
    }
    with (output_dir / "progress.log").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--national", action="store_true")
    args = parser.parse_args()
    if args.count < 0 or args.workers < 1 or args.start < 0:
        parser.error("count and start must be non-negative, and workers must be positive.")
    root = repository_root()
    with (root / "config" / "scan.json").open(encoding="utf-8") as handle:
        scan_config = json.load(handle)
    output_dir = root / "data" / "synth"
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(dataset_id, float(scan_config["pop_cap"]), int(scan_config["replicates"]), int(scan_config["seed"]), args.national) for dataset_id in range(args.start, args.start + args.count)]
    wall_started = time.perf_counter()
    results = []
    if jobs:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_dataset, *job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                append_progress(output_dir, result["meta"])
    results.sort(key=lambda result: result["meta"]["id"])
    for result in results:
        record = result["meta"]
        marker = " (skipped existing outputs)" if result["skipped"] else ""
        print(
            f"id={record['id']}, n_counties={record['n_counties']}, N={record['N']}, "
            f"n_windows={record['n_windows']}, seconds={record['seconds_total']:.3f}{marker}"
        )
    merge_metadata(output_dir)
    seconds = [float(result["meta"]["seconds_total"]) for result in results]
    if seconds:
        print(f"Mean seconds per dataset: {float(np.mean(seconds)):.3f}")
        print(f"Max seconds per dataset: {float(np.max(seconds)):.3f}")
    else:
        print("Mean seconds per dataset: 0.000")
        print("Max seconds per dataset: 0.000")
    wall_elapsed = time.perf_counter() - wall_started
    print(f"Total wall time: {wall_elapsed:.3f} seconds")
    worker_peaks = {}
    for result in results:
        peak = result["peak_bytes"]
        if peak is not None:
            worker_peaks[result["pid"]] = max(worker_peaks.get(result["pid"], 0), peak)
    if worker_peaks:
        for pid, peak in sorted(worker_peaks.items()):
            print(f"Peak memory worker {pid}: {peak / (1024 ** 2):.1f} MiB")
    else:
        print("Peak memory per worker: not measurable")
    total_size = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    print(f"Disk size of data/synth/: {total_size / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
