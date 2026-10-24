from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(name, variant="primary"):
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "processed" / f"{name}_county_{variant}.csv"
    return pd.read_csv(path, dtype={"fips": str})


def build_neighbors(lat, lon):
    latitudes = np.radians(np.asarray(lat, dtype=float).reshape(-1))
    longitudes = np.radians(np.asarray(lon, dtype=float).reshape(-1))
    if latitudes.shape != longitudes.shape or not len(latitudes):
        raise ValueError("Latitude and longitude must be non-empty arrays of equal length.")
    delta_lat = (latitudes[:, None] - latitudes[None, :]) / 2.0
    delta_lon = (longitudes[:, None] - longitudes[None, :]) / 2.0
    haversine = np.sin(delta_lat) ** 2 + np.cos(latitudes[:, None]) * np.cos(latitudes[None, :]) * np.sin(delta_lon) ** 2
    distances = 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    neighbors = np.argsort(distances, axis=1, kind="stable").astype(np.int16)
    ordered_distances = np.take_along_axis(distances, neighbors, axis=1)
    return neighbors, ordered_distances


def window_limits(pop, nb):
    population = np.asarray(pop, dtype=float).reshape(-1)
    neighbors = np.asarray(nb)
    if neighbors.ndim != 2 or neighbors.shape[0] != len(population):
        raise ValueError("Neighbor rows must match the population array.")
    if np.any(population < 0) or not np.isfinite(population).all() or population.sum() <= 0:
        raise ValueError("Population must contain finite non-negative values with a positive sum.")
    cumulative = np.cumsum(population[neighbors], axis=1)
    limits = np.sum(cumulative <= population.sum() * 0.5, axis=1).astype(int)
    return np.maximum(limits, 1)


def observed_llr(counts, pop, nb, K):
    observed = np.asarray(counts, dtype=float).reshape(-1)
    population = np.asarray(pop, dtype=float).reshape(-1)
    neighbors = np.asarray(nb)
    limits = np.asarray(K, dtype=int).reshape(-1)
    if observed.shape != population.shape:
        raise ValueError("Counts and population must have equal lengths.")
    if neighbors.ndim != 2 or neighbors.shape[0] != len(limits):
        raise ValueError("K must contain one limit for each neighbor row.")
    if neighbors.size and int(neighbors.max()) >= len(population):
        raise ValueError("Neighbor index exceeds the county arrays.")
    if np.any(observed < 0) or not np.isfinite(observed).all():
        raise ValueError("Counts must be finite and non-negative.")
    total_cases = float(observed.sum())
    total_population = float(population.sum())
    if total_population <= 0:
        raise ValueError("Total population must be positive.")
    max_k = min(int(limits.max(initial=0)), neighbors.shape[1])
    if max_k <= 0:
        return np.empty((neighbors.shape[0], 0), dtype=float)
    ordered = neighbors[:, :max_k]
    cumulative_counts = np.cumsum(observed[ordered], axis=1)
    cumulative_population = np.cumsum(population[ordered], axis=1)
    expected = total_cases * cumulative_population / total_population
    remainder_cases = total_cases - cumulative_counts
    remainder_expected = total_cases - expected
    scores = np.zeros_like(expected, dtype=float)
    valid_upward = cumulative_counts > expected
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inside = np.where(cumulative_counts > 0, cumulative_counts * np.log(cumulative_counts / expected), 0.0)
        outside = np.where(remainder_cases > 0, remainder_cases * np.log(remainder_cases / remainder_expected), 0.0)
        scores[valid_upward] = (inside + outside)[valid_upward]
    scores[~valid_upward] = 0.0
    invalid = np.arange(max_k)[None, :] >= limits[:, None]
    scores[invalid] = -np.inf
    scores[~np.isfinite(scores) & (scores != -np.inf)] = 0.0
    return scores


def null_max_distribution(pop, N, nb, K, R, seed):
    population = np.asarray(pop, dtype=float).reshape(-1)
    if R < 1:
        raise ValueError("R must be positive.")
    total_population = population.sum()
    if total_population <= 0:
        raise ValueError("Total population must be positive.")
    probabilities = population / total_population
    random = np.random.default_rng(seed)
    result = np.empty(int(R), dtype=float)
    for replicate in range(int(R)):
        draw = random.multinomial(int(N), probabilities)
        scores = observed_llr(draw, population, nb, K)
        result[replicate] = float(np.max(scores)) if scores.size else 0.0
    return result


def window_p_values(llr, null_max):
    scores = np.asarray(llr, dtype=float)
    null_sorted = np.sort(np.asarray(null_max, dtype=float).reshape(-1))
    if not len(null_sorted):
        raise ValueError("The null maximum distribution must not be empty.")
    insertion = np.searchsorted(null_sorted, scores, side="left")
    return (1.0 + len(null_sorted) - insertion) / (len(null_sorted) + 1.0)


def window_table(llr, p_values, counts, pop, nb, distances, K, fips):
    scores = np.asarray(llr, dtype=float)
    pvalues = np.asarray(p_values, dtype=float)
    observed = np.asarray(counts, dtype=float).reshape(-1)
    population = np.asarray(pop, dtype=float).reshape(-1)
    neighbors = np.asarray(nb)
    ordered_distances = np.asarray(distances, dtype=float)
    limits = np.asarray(K, dtype=int).reshape(-1)
    county_fips = np.asarray(fips).astype(str)
    max_k = scores.shape[1]
    center_indices, k_indices = np.where(np.arange(max_k)[None, :] < limits[:, None])
    if not len(center_indices):
        return pd.DataFrame(columns=["center_fips", "k", "n_Z", "mu_Z", "llr", "p_value", "radius_km", "n_counties", "pop_share"])
    cumulative_counts = np.cumsum(observed[neighbors[:, :max_k]], axis=1)
    cumulative_population = np.cumsum(population[neighbors[:, :max_k]], axis=1)
    n_z = cumulative_counts[center_indices, k_indices]
    population_z = cumulative_population[center_indices, k_indices]
    total_population = population.sum()
    total_cases = observed.sum()
    return pd.DataFrame({
        "center_fips": county_fips[center_indices],
        "k": k_indices + 1,
        "n_Z": n_z,
        "mu_Z": total_cases * population_z / total_population,
        "llr": scores[center_indices, k_indices],
        "p_value": pvalues[center_indices, k_indices],
        "radius_km": ordered_distances[center_indices, k_indices],
        "n_counties": k_indices + 1,
        "pop_share": population_z / total_population,
    })


def most_likely_cluster(llr):
    scores = np.asarray(llr, dtype=float)
    if scores.ndim != 2 or not scores.size:
        raise ValueError("LLR scores must be a non-empty two-dimensional array.")
    flat_index = int(np.argmax(scores))
    center, k_index = np.unravel_index(flat_index, scores.shape)
    return int(center), int(k_index + 1), float(scores[center, k_index])
