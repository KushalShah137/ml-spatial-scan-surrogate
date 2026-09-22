"""
Data Preparation Pipeline
Preprocesses CDC WONDER Compressed Mortality data and joins US Census county centroids.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def find_mortality_data_file(data_dir: Path) -> Path:
    """Finds the mortality data file supporting standard filenames."""
    candidates = [
        data_dir / "Compressed_Mortality__1999-2016.csv",
        data_dir / "Compressed Mortality, 1999-2016.csv",
        Path("data/Compressed_Mortality__1999-2016.csv"),
        Path("data/Compressed Mortality, 1999-2016.csv"),
        Path("Compressed_Mortality__1999-2016.csv"),
        Path("Compressed Mortality, 1999-2016.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Mortality dataset not found in {data_dir} or current directory.")


def load_census_centroids(reference_path: Path = None) -> pd.DataFrame:
    """Loads US Census county centroid data (fips, lat, lon)."""
    if reference_path is None:
        reference_path = Path(__file__).resolve().parent / "census_county_centroids.csv"

    if reference_path.exists():
        df_cen = pd.read_csv(reference_path, dtype={"fips": str})
    else:
        import urllib.request
        import zipfile
        import io

        url = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open(z.namelist()[0]) as f:
                gaz = pd.read_csv(f, sep="\t", dtype={"GEOID": str})
                gaz.columns = gaz.columns.str.strip()
                df_cen = pd.DataFrame(
                    {
                        "fips": gaz["GEOID"].str.strip().str.zfill(5),
                        "name": gaz["NAME"].str.strip(),
                        "state": gaz["USPS"].str.strip(),
                        "lat": pd.to_numeric(gaz["INTPTLAT"], errors="coerce"),
                        "lon": pd.to_numeric(gaz["INTPTLONG"], errors="coerce"),
                    }
                )

    df_cen["fips"] = df_cen["fips"].astype(str).str.strip().str.zfill(5)
    df_cen["lat"] = pd.to_numeric(df_cen["lat"], errors="coerce")
    df_cen["lon"] = pd.to_numeric(df_cen["lon"], errors="coerce")
    return df_cen[["fips", "lat", "lon"]]


def prep_mortality_data(
    raw_csv_path: Path = None, output_csv_path: Path = None
) -> pd.DataFrame:
    """
    Cleans raw mortality records, aggregates by county across all years,
    joins census centroids, and exports county_clean.csv.
    """
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    if raw_csv_path is None:
        raw_csv_path = find_mortality_data_file(data_dir)

    if output_csv_path is None:
        output_csv_path = data_dir / "county_clean.csv"

    print(f"Loading raw mortality data from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path, dtype={"County Code": str}, low_memory=False)

    # Strip footer/header junk rows — keep only rows where "County Code" is a valid FIPS string and "Deaths" is numeric
    df["Deaths_numeric"] = pd.to_numeric(df["Deaths"], errors="coerce")
    valid_mask = (
        df["County Code"].notna()
        & df["County Code"].astype(str).str.strip().str.match(r"^\d{4,5}$")
        & df["Deaths_numeric"].notna()
    )
    df_valid = df[valid_mask].copy()
    df_valid["fips"] = df_valid["County Code"].astype(str).str.strip().str.zfill(5)
    df_valid["Deaths"] = df_valid["Deaths_numeric"]
    df_valid["Population"] = pd.to_numeric(df_valid["Population"], errors="coerce")

    # Aggregate by county across all years: total_deaths = sum(Deaths), avg_population = mean(Population)
    agg = df_valid.groupby(["fips", "County"], as_index=False).agg(
        total_deaths=("Deaths", "sum"), avg_population=("Population", "mean")
    )
    agg = agg.rename(columns={"County": "county_name"})

    # Join US Census county centroid data by FIPS code for lat/lon
    centroids = load_census_centroids()
    merged = pd.merge(agg, centroids, on="fips", how="inner")

    # Save to /data/county_clean.csv: county_name, fips, total_deaths, avg_population, lat, lon
    cols = ["county_name", "fips", "total_deaths", "avg_population", "lat", "lon"]
    cleaned_df = merged[cols].copy()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_csv_path, index=False)
    print(f"Successfully processed {len(cleaned_df)} counties.")
    print(f"Saved cleaned county data to: {output_csv_path}")

    return cleaned_df


if __name__ == "__main__":
    prep_mortality_data()
