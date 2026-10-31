from __future__ import annotations

import argparse
import json
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "datasets.json"
CENTROID_PATH = REPO_ROOT / "data" / "external" / "census_county_centroids.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
EXCLUDED_STATE_PREFIXES = {"02", "15", "60", "66", "69", "72", "78"}
FIPS_CROSSWALK = {"46113": "46102", "51515": "51019"}
SUPPRESSED_CROSSWALK = {"51560": "51005"}
OUTPUT_COLUMNS = [
    "fips", "state", "county", "lat", "lon", "population", "deaths",
    "suppressed_flag", "expected",
]


def load_datasets() -> list[dict[str, str]]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    if not isinstance(datasets, list):
        raise ValueError(f"{CONFIG_PATH} must contain a list of datasets.")
    for entry in datasets:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key].strip()
            for key in ("name", "icd10", "raw_file")
        ):
            raise ValueError("Each dataset entry must define non-empty name, icd10, and raw_file values.")
    return datasets


def resolve_raw_path(configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Configured input file not found: {path}")
    return path


def parse_numeric(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null"}:
        return math.nan
    return float(text)


def read_wonder_export(path: Path) -> tuple[pd.DataFrame, float, float, int, list[dict[str, str]], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = handle.read().splitlines()
    if not lines:
        raise ValueError(f"Input file is empty: {path}")

    header_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if header_index is None:
        raise ValueError(f"Input file has no header: {path}")
    stop_index = len(lines)
    for i in range(header_index + 1, len(lines)):
        if lines[i].lstrip().lstrip('"').startswith("---") or not lines[i].strip():
            stop_index = i
            break
    frame = pd.read_csv(
        StringIO("\n".join(lines[header_index:stop_index])),
        sep="\t", dtype=str, keep_default_na=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Notes", "County", "County Code", "Deaths", "Population"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    notes = frame["Notes"].astype(str).str.strip()
    totals = frame.loc[notes.eq("Total")]
    if totals.empty:
        raise ValueError(
            f"No totals row found in {path}; the file must be re-exported with Show Totals ticked."
        )
    total_deaths = parse_numeric(totals.iloc[0]["Deaths"])
    total_population = parse_numeric(totals.iloc[0]["Population"])
    total_line = next(
        line for line in lines[header_index + 1:stop_index]
        if line.split("\t", 1)[0].strip().strip('"') == "Total"
    )
    if not math.isfinite(total_deaths) or not math.isfinite(total_population) or total_population <= 0:
        raise ValueError(f"Totals row in {path} has invalid Deaths or Population values.")

    county_rows = frame.loc[~notes.eq("Total")].copy()
    county_codes = county_rows["County Code"].astype(str).str.strip()
    valid_code = county_codes.str.fullmatch(r"\d{1,5}")
    dropped_examples = county_rows.loc[~valid_code, ["County", "County Code", "Deaths", "Population", "Notes"]].head(10).fillna("").to_dict("records")
    dropped_rows = int((~valid_code).sum())
    county_rows = county_rows.loc[valid_code].copy()
    county_rows["fips"] = county_codes.loc[valid_code].str.zfill(5).to_numpy()

    death_text = county_rows["Deaths"].astype(str).str.strip()
    suppressed = death_text.str.casefold().eq("suppressed")
    parsed_deaths = pd.to_numeric(death_text.where(~suppressed).str.replace(",", "", regex=False), errors="coerce")
    invalid_deaths = parsed_deaths.isna() & ~suppressed
    if invalid_deaths.any():
        bad_codes = county_rows.loc[invalid_deaths, "fips"].tolist()
        raise ValueError(f"Non-numeric, non-suppressed Deaths values for county FIPS: {bad_codes}")
    county_rows["deaths"] = parsed_deaths
    county_rows["suppressed_flag"] = suppressed.astype(int)
    county_rows["population"] = pd.to_numeric(
        county_rows["Population"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    if county_rows["population"].isna().any():
        bad_codes = county_rows.loc[county_rows["population"].isna(), "fips"].tolist()
        raise ValueError(f"Invalid Population values for county FIPS: {bad_codes}")
    county_rows["county_source"] = county_rows["County"].astype(str).str.strip()
    return county_rows[["fips", "county_source", "population", "deaths", "suppressed_flag"]], total_deaths, total_population, dropped_rows, dropped_examples, total_line


def load_centroids() -> pd.DataFrame:
    if not CENTROID_PATH.is_file():
        raise FileNotFoundError(f"County centroid file not found: {CENTROID_PATH}")
    centroids = pd.read_csv(CENTROID_PATH, dtype={"fips": str}, keep_default_na=False)
    required = {"fips", "name", "state", "lat", "lon"}
    missing = required.difference(centroids.columns)
    if missing:
        raise ValueError(f"Centroid file is missing required columns: {', '.join(sorted(missing))}")
    centroids["fips"] = centroids["fips"].astype(str).str.strip().str.zfill(5)
    centroids["lat"] = pd.to_numeric(centroids["lat"], errors="coerce")
    centroids["lon"] = pd.to_numeric(centroids["lon"], errors="coerce")
    centroids["name"] = centroids["name"].astype(str).str.strip()
    centroids["state"] = centroids["state"].astype(str).str.strip()
    centroids = centroids.loc[~centroids["fips"].str[:2].isin(EXCLUDED_STATE_PREFIXES)]
    centroids = centroids.rename(columns={"name": "county"})
    return centroids[["fips", "state", "county", "lat", "lon"]].drop_duplicates("fips")


def aggregate_crosswalked(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    suppressed_special = rows["fips"].isin(SUPPRESSED_CROSSWALK) & rows["suppressed_flag"].eq(1)
    for _, row in rows.loc[suppressed_special].iterrows():
        print(
            f"WARNING: dropping suppressed crosswalk row FIPS {row['fips']} ({row['county_source']}); "
            f"population {row['population']:g} is assigned to {SUPPRESSED_CROSSWALK[row['fips']]}, "
            "and its hidden deaths remain in the residual."
        )
    rows.loc[suppressed_special, "deaths"] = 0
    rows.loc[suppressed_special, "suppressed_flag"] = 0
    rows["fips"] = rows["fips"].replace(FIPS_CROSSWALK)
    rows["fips"] = rows["fips"].replace(SUPPRESSED_CROSSWALK)
    grouped = rows.groupby("fips", as_index=False, sort=True).agg(
        population=("population", "sum"),
        deaths=("deaths", lambda values: values.sum(min_count=1)),
        suppressed_flag=("suppressed_flag", "max"),
    )
    grouped.loc[grouped["suppressed_flag"].eq(1), "deaths"] = np.nan
    return grouped


def allocate_suppressed(residual: float, populations: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    count = len(populations)
    target = int(round(residual))
    if count == 0 or abs(residual - target) > 1e-7 or target < count or target > 9 * count:
        return None, "residual cannot be allocated within 1 to 9 deaths per suppressed county"

    weights = np.asarray(populations, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() == 0:
        weights = np.ones(count, dtype=float)

    allocations = np.zeros(count, dtype=float)
    active = np.ones(count, dtype=bool)
    remaining = float(target)
    while active.any():
        active_indices = np.flatnonzero(active)
        active_weights = weights[active_indices]
        if active_weights.sum() == 0:
            active_weights = np.ones(len(active_indices), dtype=float)
        proposals = remaining * active_weights / active_weights.sum()
        below = proposals < 1.0 - 1e-12
        above = proposals > 9.0 + 1e-12
        if not below.any() and not above.any():
            allocations[active_indices] = proposals
            break
        for index, proposal, is_below, is_above in zip(active_indices, proposals, below, above):
            if is_below:
                allocations[index] = 1.0
                active[index] = False
                remaining -= 1.0
            elif is_above:
                allocations[index] = 9.0
                active[index] = False
                remaining -= 9.0
    floors = np.floor(allocations).astype(int)
    difference = target - int(floors.sum())
    fractions = allocations - floors
    order = np.argsort(-fractions, kind="stable")
    for index in order:
        if difference <= 0:
            break
        if floors[index] < 9:
            floors[index] += 1
            difference -= 1
    if difference != 0 or floors.sum() != target or np.any(floors < 1) or np.any(floors > 9):
        return None, "rounded allocation could not preserve the residual exactly"
    return floors.astype(float), None


def build_report(
    dataset_name: str,
    county_rows: pd.DataFrame,
    total_deaths: float,
    total_population: float,
    unsuppressed_deaths: float,
    residual: float,
    national_rate: float,
    dropped_rows: int,
    unmatched_wonder: list[str],
    unmatched_centroid: list[str],
    primary_sum: float,
    warning: str | None,
    raw_totals_line: str,
) -> str:
    county_count = len(county_rows)
    suppressed_count = int(county_rows["suppressed_flag"].sum())
    suppressed_pct = 100.0 * suppressed_count / county_count if county_count else 0.0
    populations = county_rows["population"].astype(float)
    close = math.isclose(primary_sum, total_deaths, rel_tol=1e-6, abs_tol=1.0)
    lines = [
        f"Dataset: {dataset_name}",
        f"County count: {county_count}",
        f"Rows dropped for invalid county code: {dropped_rows}",
        f"Raw totals row: {raw_totals_line}",
        f"Total deaths (totals row): {total_deaths:g}",
        f"Sum of unsuppressed deaths: {unsuppressed_deaths:g}",
        f"Residual hidden deaths: {residual:g}",
        f"Suppressed county count: {suppressed_count}",
        f"Suppressed county percentage: {suppressed_pct:.2f}%",
        f"National rate: {national_rate:.12g}",
        f"Population min/median/max: {populations.min():g} / {populations.median():g} / {populations.max():g}",
        f"WONDER FIPS without centroid: {unmatched_wonder}",
        f"In-scope centroid FIPS without WONDER row: {unmatched_centroid}",
        f"Primary deaths sum: {primary_sum:g}",
        f"Primary sum close to total deaths: {'yes' if close else 'no'}",
        f"Death buckets: 0={int((county_rows.loc[county_rows['suppressed_flag'].eq(0), 'deaths'] == 0).sum())}, suppressed (1-9)={int(county_rows['suppressed_flag'].sum())}, 10-19={int((county_rows.loc[county_rows['suppressed_flag'].eq(0), 'deaths'].between(10, 19)).sum())}, 20-49={int((county_rows.loc[county_rows['suppressed_flag'].eq(0), 'deaths'].between(20, 49)).sum())}, 50+={int((county_rows.loc[county_rows['suppressed_flag'].eq(0), 'deaths'] >= 50).sum())}",
    ]
    if warning:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def process_dataset(dataset: dict[str, str]) -> str:
    name = dataset["name"].strip()
    raw_path = resolve_raw_path(dataset["raw_file"])
    source_rows, total_deaths, total_population, dropped_rows, dropped_examples, raw_totals_line = read_wonder_export(raw_path)
    print(f"{name}: invalid county-code rows dropped: {dropped_rows}")
    if dropped_examples:
        for example in dropped_examples:
            print(f"{name}: dropped row example: {example}")
    else:
        print(f"{name}: no invalid county-code examples; the valid data section contains no such rows.")
    wonder = aggregate_crosswalked(source_rows)
    wonder = wonder.loc[~wonder["fips"].str[:2].isin(EXCLUDED_STATE_PREFIXES)].copy()
    centroids = load_centroids()

    wonder_fips = set(wonder["fips"])
    centroid_fips = set(centroids["fips"])
    unmatched_wonder = sorted(wonder_fips - centroid_fips)
    unmatched_centroid = sorted(centroid_fips - wonder_fips)
    print(f"{name}: WONDER FIPS without centroid: {unmatched_wonder}")
    print(f"{name}: in-scope centroid FIPS without WONDER row: {unmatched_centroid}")

    joined = wonder.merge(centroids, on="fips", how="inner", validate="one_to_one")
    joined = joined[OUTPUT_COLUMNS[:5] + ["population", "deaths", "suppressed_flag"]].copy()
    national_rate = total_deaths / total_population
    unsuppressed = joined["suppressed_flag"].eq(0)
    unsuppressed_deaths = float(joined.loc[unsuppressed, "deaths"].sum())
    residual = total_deaths - unsuppressed_deaths
    suppressed_mask = joined["suppressed_flag"].eq(1)
    allocation, warning = allocate_suppressed(
        residual, joined.loc[suppressed_mask, "population"].to_numpy(dtype=float)
    )
    if allocation is None:
        print(f"WARNING {name}: {warning}; using the national-rate fallback.")
        fallback = np.rint(
            joined.loc[suppressed_mask, "population"].to_numpy(dtype=float) * national_rate
        )
        allocation = np.clip(fallback, 1, 9).astype(float)
    joined["expected"] = joined["population"] * national_rate

    for variant in ("primary", "zero", "five"):
        output = joined.copy()
        if variant == "primary":
            output.loc[suppressed_mask, "deaths"] = allocation
        elif variant == "zero":
            output.loc[suppressed_mask, "deaths"] = 0
        else:
            output.loc[suppressed_mask, "deaths"] = 5
        output["deaths"] = pd.to_numeric(output["deaths"], errors="coerce")
        output = output[OUTPUT_COLUMNS]
        output_path = OUTPUT_DIR / f"{name}_county_{variant}.csv"
        output.to_csv(output_path, index=False)

    primary_sum = float(joined.loc[unsuppressed, "deaths"].sum() + allocation.sum())
    report = build_report(
        name, joined, total_deaths, total_population, unsuppressed_deaths, residual,
        national_rate, dropped_rows, unmatched_wonder, unmatched_centroid,
        primary_sum, warning, raw_totals_line,
    )
    report_path = OUTPUT_DIR / f"{name}_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(raw_totals_line)
    print(f"Sum of unsuppressed deaths: {unsuppressed_deaths:g}")
    print(report)
    print(f"Saved report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare county data from tab-delimited exports.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Process every configured dataset.")
    selection.add_argument("--name", help="Process one configured dataset by name.")
    args = parser.parse_args()

    datasets = load_datasets()
    if args.all:
        selected = datasets
    else:
        selected = [entry for entry in datasets if entry["name"] == args.name]
        if not selected:
            available = ", ".join(entry["name"] for entry in datasets)
            parser.error(f"Unknown dataset name {args.name!r}. Available names: {available}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in selected:
        process_dataset(dataset)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
