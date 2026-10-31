import json
from pathlib import Path

import numpy as np

from data_prep import REPO_ROOT
from scan import build_neighbors, load_dataset, null_max_distribution, observed_llr, window_limits, window_p_values


def naive_subset_difference(frame, neighbors, subset, vectorized, pop_cap):
    population = frame["population"].to_numpy(dtype=float)
    counts = frame["deaths"].to_numpy(dtype=float)
    total_cases = float(counts.sum())
    total_population = float(population.sum())
    population_cap = total_population * pop_cap
    differences = []
    for local_index, center in enumerate(subset):
        inside_cases = 0.0
        inside_population = 0.0
        for k, neighbor in enumerate(neighbors[center], start=1):
            inside_cases += float(counts[neighbor])
            inside_population += float(population[neighbor])
            if k > 1 and inside_population > population_cap:
                break
            inside_expected = total_cases * inside_population / total_population
            if inside_cases > inside_expected:
                outside_cases = total_cases - inside_cases
                outside_expected = total_cases - inside_expected
                inside_term = inside_cases * np.log(inside_cases / inside_expected) if inside_cases else 0.0
                outside_term = outside_cases * np.log(outside_cases / outside_expected) if outside_cases else 0.0
                naive = inside_term + outside_term
            else:
                naive = 0.0
            differences.append(abs(float(vectorized[local_index, k - 1]) - float(naive)))
            if inside_population > population_cap:
                break
    return max(differences, default=0.0)


def null_type_one(frame, null_max, seed, pop_cap, replicates=500):
    population = frame["population"].to_numpy(dtype=float)
    counts = frame["deaths"].to_numpy(dtype=float)
    total_cases = int(round(counts.sum()))
    neighbors, _ = build_neighbors(frame["lat"].to_numpy(), frame["lon"].to_numpy())
    limits = window_limits(population, neighbors, pop_cap)
    random = np.random.default_rng(seed)
    probabilities = population / population.sum()
    rejected = 0
    for _ in range(replicates):
        draw = random.multinomial(total_cases, probabilities)
        maximum = float(np.max(observed_llr(draw, population, neighbors, limits)))
        p_value = float(window_p_values(np.asarray([maximum]), null_max)[0])
        rejected += p_value <= 0.05
    return rejected / replicates


def main():
    with (REPO_ROOT / "config" / "datasets.json").open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    with (REPO_ROOT / "config" / "scan.json").open(encoding="utf-8") as handle:
        scan_config = json.load(handle)
    pop_cap = float(scan_config["pop_cap"])
    cap_tag = f"c{int(round(pop_cap * 100))}"
    targets = [datasets[0], datasets[-1]]
    for offset, dataset in enumerate(targets):
        name = dataset["name"]
        frame = load_dataset(name, "primary")
        neighbors, _ = build_neighbors(frame["lat"].to_numpy(), frame["lon"].to_numpy())
        population = frame["population"].to_numpy(dtype=float)
        counts = frame["deaths"].to_numpy(dtype=float)
        limits = window_limits(population, neighbors, pop_cap)
        random = np.random.default_rng(42)
        subset = np.sort(random.choice(len(frame), size=min(40, len(frame)), replace=False))
        vectorized = observed_llr(counts, population, neighbors[subset], limits[subset])
        difference = naive_subset_difference(frame, neighbors, subset, vectorized, pop_cap)
        null_path = REPO_ROOT / "data" / "processed" / f"{name}_{cap_tag}_null_max.npy"
        null_max = np.load(null_path)
        type_one = null_type_one(frame, null_max, 20261025 + offset, pop_cap)
        print(f"Dataset: {name}")
        print(f"Naive/vectorized max absolute difference: {difference:.12g}")
        print(f"Type I error over 500 null datasets: {type_one:.3f}")


if __name__ == "__main__":
    main()
