"""
ML Spatial Scan Surrogate - Streamlit Dashboard
Interactive visualization of Kulldorff spatial cluster scans, ML surrogate benchmarks, and speedup analysis.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup page configuration
st.set_page_config(
    page_title="ML Spatial Scan Surrogate",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Repo directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


@st.cache_data
def load_data():
    """Loads all precomputed dataset and results files."""
    county_df = pd.read_csv(DATA_DIR / "county_clean.csv", dtype={"fips": str})
    labeled_df = pd.read_csv(
        DATA_DIR / "labeled_windows.csv", dtype={"center_fips": str}
    )
    comp_df = pd.read_csv(RESULTS_DIR / "comparison.csv")
    speedup_df = pd.read_csv(RESULTS_DIR / "speedup_comparison.csv")
    return county_df, labeled_df, comp_df, speedup_df


def create_circle_points(
    lat_center: float, lon_center: float, radius_km: float, n_points: int = 72
) -> tuple[list[float], list[float]]:
    """
    Computes (lat, lon) coordinates of a spherical circle with a given radius in km.
    """
    R = 6371.0  # Earth radius in km
    d = radius_km / R  # Angular distance in radians
    lat_c = np.radians(lat_center)
    lon_c = np.radians(lon_center)
    angles = np.linspace(0, 2 * np.pi, n_points)

    circle_lats = []
    circle_lons = []
    for a in angles:
        lat = np.arcsin(
            np.sin(lat_c) * np.cos(d) + np.cos(lat_c) * np.sin(d) * np.cos(a)
        )
        lon = lon_c + np.arctan2(
            np.sin(a) * np.sin(d) * np.cos(lat_c),
            np.cos(d) - np.sin(lat_c) * np.sin(lat),
        )
        circle_lats.append(float(np.degrees(lat)))
        circle_lons.append(float(np.degrees(lon)))

    return circle_lats, circle_lons


def main():
    county_df, labeled_df, comp_df, speedup_df = load_data()

    # =========================================================================
    # 1. Header
    # =========================================================================
    st.title("ML Spatial Scan Surrogate")
    st.markdown(
        "**ML surrogate for Kulldorff spatial scan statistic significance testing "
        "— accelerating rare disease cluster detection in spatial epidemiology**"
    )
    st.divider()

    # =========================================================================
    # 5. Sidebar Controls & Dynamic Metrics
    # =========================================================================
    st.sidebar.header("Scan Parameters & Controls")
    p_threshold = st.sidebar.slider(
        "Significance Threshold (p-value)",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Filters windows and counties considered statistically significant clusters.",
    )

    num_overlay_windows = st.sidebar.slider(
        "Top Significant Windows to Overlay",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of top circular candidate windows (by log-likelihood ratio) to draw on the map.",
    )

    # Compute county-level minimum p-value
    best_p_series = labeled_df.groupby("center_fips")["p_value"].min()
    county_map_data = county_df.copy()
    county_map_data["best_p_value"] = county_map_data["fips"].map(best_p_series)
    county_map_data["is_significant"] = county_map_data["best_p_value"] < p_threshold
    county_map_data["Cluster Status"] = county_map_data["is_significant"].map(
        {
            True: f"Significant (p < {p_threshold:.2f})",
            False: f"Non-Significant (p ≥ {p_threshold:.2f})",
        }
    )

    # Global counts
    total_counties = len(county_df)
    total_windows = len(labeled_df)
    sig_windows_df = labeled_df[labeled_df["p_value"] < p_threshold]
    sig_windows_count = len(sig_windows_df)
    sig_counties_count = int(county_map_data["is_significant"].sum())

    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Dataset Summary")
    st.sidebar.metric("Total Counties", f"{total_counties}")
    st.sidebar.metric("Candidate Windows", f"{total_windows:,}")
    st.sidebar.metric(
        "Significant Windows",
        f"{sig_windows_count:,}",
        delta=f"{sig_windows_count / total_windows * 100:.1f}% of total",
    )
    st.sidebar.metric(
        "Counties in Significant Clusters",
        f"{sig_counties_count}",
        delta=f"{sig_counties_count / total_counties * 100:.1f}% of counties",
    )

    # =========================================================================
    # 2. Interactive Cluster Map
    # =========================================================================
    st.subheader("Interactive Disease Cluster Map (Kulldorff Spatial Scan)")
    st.caption(
        f"County centroids sized by total observed mortality deaths and colored by cluster significance "
        f"at threshold p < {p_threshold:.2f}. Outlined circles represent the top {num_overlay_windows} significant candidate windows."
    )

    # Base scatter_geo plot
    color_map = {
        f"Significant (p < {p_threshold:.2f})": "#E63946",
        f"Non-Significant (p ≥ {p_threshold:.2f})": "#457B9D",
    }

    fig_map = px.scatter_geo(
        county_map_data,
        lat="lat",
        lon="lon",
        size="total_deaths",
        color="Cluster Status",
        color_discrete_map=color_map,
        hover_name="county_name",
        hover_data={
            "lat": False,
            "lon": False,
            "total_deaths": ":.0f",
            "avg_population": ":,.0f",
            "best_p_value": ":.4f",
            "Cluster Status": True,
        },
        scope="usa",
        size_max=32,
    )

    # Overlay top N significant candidate windows as circles
    top_sig_windows = (
        sig_windows_df.sort_values(by="log_lr", ascending=False)
        .head(num_overlay_windows)
        .reset_index(drop=True)
    )

    palette = ["#E63946", "#F4A261", "#2A9D8F", "#9B5DE5", "#E76F51", "#00B4D8", "#FFB703", "#588157", "#D62246", "#4A4E69"]

    for idx, win in top_sig_windows.iterrows():
        fips = win["center_fips"]
        c_rows = county_df[county_df["fips"] == fips]
        if len(c_rows) == 0:
            continue
        c_info = c_rows.iloc[0]
        r = float(win["radius"])
        border_color = palette[idx % len(palette)]

        if r > 0:
            c_lats, c_lons = create_circle_points(float(c_info["lat"]), float(c_info["lon"]), r)
            fig_map.add_trace(
                go.Scattergeo(
                    lat=c_lats,
                    lon=c_lons,
                    mode="lines",
                    line=dict(width=2.5, color=border_color, dash="solid"),
                    name=f"Top {idx+1}: {win['center_county']} (k={win['k']}, r={r:.0f}km)",
                    hoverinfo="text",
                    text=(
                        f"<b>Rank #{idx+1} Cluster Window</b><br>"
                        f"Center: {win['center_county']}<br>"
                        f"k: {win['k']} counties<br>"
                        f"Radius: {r:.1f} km<br>"
                        f"Observed Cases (n_Z): {win['n_Z']:.0f}<br>"
                        f"Expected Cases (mu_Z): {win['mu_Z']:.1f}<br>"
                        f"Log-Likelihood Ratio: {win['log_lr']:.2f}<br>"
                        f"p-value: {win['p_value']:.4f}"
                    ),
                    showlegend=True,
                )
            )

    fig_map.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="#F8F9FA",
            subunitcolor="#CED4DA",
            countrycolor="#6C757D",
            showlakes=True,
            lakecolor="#E9ECEF",
            showocean=True,
            oceancolor="#F1F3F5",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=620,
    )

    st.plotly_chart(fig_map, use_container_width=True)

    with st.expander("View Top Significant Candidate Windows Data"):
        display_top = top_sig_windows[
            [
                "center_county",
                "center_fips",
                "k",
                "n_Z",
                "mu_Z",
                "radius",
                "area",
                "log_lr",
                "p_value",
            ]
        ].copy()
        display_top.columns = [
            "Center County",
            "FIPS",
            "k (Counties)",
            "Observed (n_Z)",
            "Expected (mu_Z)",
            "Radius (km)",
            "Area (km²)",
            "Log-LR",
            "p-value",
        ]
        st.dataframe(
            display_top.style.format(
                {
                    "Observed (n_Z)": "{:.0f}",
                    "Expected (mu_Z)": "{:.2f}",
                    "Radius (km)": "{:.1f}",
                    "Area (km²)": "{:,.0f}",
                    "Log-LR": "{:.2f}",
                    "p-value": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

    st.divider()

    # =========================================================================
    # 3. Model Comparison Section
    # =========================================================================
    st.subheader("Model Performance & Evaluation Benchmark")
    st.markdown(
        "Models evaluated on a held-out test split (80/20) **grouped by center county FIPS** "
        "to prevent spatial data leakage across neighboring windows."
    )

    col1, col2 = st.columns(2)

    # MAE Comparison Plot
    with col1:
        comp_sorted_mae = comp_df.sort_values(by="mae", ascending=True).copy()
        comp_sorted_mae["is_surrogate"] = comp_sorted_mae["model_name"].str.contains("XGBoost")
        
        fig_mae = px.bar(
            comp_sorted_mae,
            x="model_name",
            y="mae",
            log_y=True,
            color="is_surrogate",
            color_discrete_map={True: "#2A9D8F", False: "#457B9D"},
            text="mae",
            title="Test Set Mean Absolute Error (MAE) - Log Scale (Lower is Better)",
            labels={"model_name": "Model", "mae": "Mean Absolute Error (MAE)"},
        )
        fig_mae.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            showlegend=False,
        )
        fig_mae.update_layout(xaxis_tickangle=-25, height=420)
        st.plotly_chart(fig_mae, use_container_width=True)

    # Inference Latency Plot
    with col2:
        comp_sorted_time = comp_df.sort_values(by="mean_inference_time_ms", ascending=True).copy()
        comp_sorted_time["is_surrogate"] = comp_sorted_time["model_name"].str.contains("XGBoost")

        fig_time = px.bar(
            comp_sorted_time,
            x="model_name",
            y="mean_inference_time_ms",
            log_y=True,
            color="is_surrogate",
            color_discrete_map={True: "#2A9D8F", False: "#E76F51"},
            text="mean_inference_time_ms",
            title="Inference Latency per Window (ms) - Log Scale (Lower is Better)",
            labels={"model_name": "Model", "mean_inference_time_ms": "Mean Inference Time (ms)"},
        )
        fig_time.update_traces(
            texttemplate="%{text:.5f} ms",
            textposition="outside",
            showlegend=False,
        )
        fig_time.update_layout(xaxis_tickangle=-25, height=420)
        st.plotly_chart(fig_time, use_container_width=True)

    # Benchmark Table
    st.markdown("**Detailed Model Evaluation Metrics Table (`results/comparison.csv`)**")
    st.dataframe(
        comp_df.style.format(
            {
                "mae": "{:.6f}",
                "mean_inference_time_ms": "{:.6f} ms",
            }
        ),
        use_container_width=True,
    )

    st.divider()

    # =========================================================================
    # 4. Computational Speedup Section
    # =========================================================================
    st.subheader("Computational Speedup: Traditional Monte Carlo vs. ML Surrogate")

    # Extract latency metrics
    mc_row = speedup_df[speedup_df["method"].str.contains("Monte Carlo")].iloc[0]
    surrogate_row = speedup_df[speedup_df["method"].str.contains("Surrogate")].iloc[0]

    mc_time_ms = float(mc_row["mean_time_ms"])
    surrogate_time_ms = float(surrogate_row["mean_time_ms"])
    speedup_factor = mc_time_ms / surrogate_time_ms

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(
            label="Traditional Monte Carlo (R=999)",
            value=f"{mc_time_ms:.2f} ms",
            help="Average time to build null distribution and compute p-value for 1 candidate window from scratch.",
        )
    with stat_col2:
        st.metric(
            label="XGBoost Surrogate Model",
            value=f"{surrogate_time_ms:.5f} ms",
            help="Average per-window inference latency using the trained gradient boosted surrogate.",
        )
    with stat_col3:
        st.metric(
            label="Surrogate Acceleration",
            value=f"{speedup_factor:,.0f}x FASTER",
            delta=f"{(mc_time_ms - surrogate_time_ms):.2f} ms saved per window",
        )

    # Speedup Bar Chart
    speedup_plot_df = pd.DataFrame(
        {
            "Method": [
                "Traditional Monte Carlo (R=999)",
                "XGBoost Surrogate Model",
            ],
            "Execution Time (ms)": [mc_time_ms, surrogate_time_ms],
            "Category": ["Monte Carlo", "Surrogate"],
        }
    )

    fig_speedup = px.bar(
        speedup_plot_df,
        x="Method",
        y="Execution Time (ms)",
        color="Category",
        color_discrete_map={"Monte Carlo": "#E63946", "Surrogate": "#2A9D8F"},
        log_y=True,
        text="Execution Time (ms)",
        title="Single-Window Significance Testing Runtime (Log Scale)",
    )
    fig_speedup.update_traces(
        texttemplate="%{y:.4f} ms",
        textposition="outside",
        showlegend=False,
    )
    fig_speedup.update_layout(height=380)

    st.plotly_chart(fig_speedup, use_container_width=True)


if __name__ == "__main__":
    main()
