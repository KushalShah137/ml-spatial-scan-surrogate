"""
Model Training and Benchmarking Pipeline
Trains ML surrogate (XGBoost) and baseline models (MLR, RF, SVR, MLP) using Group-Split evaluation.
"""

from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb

# Add repo root to sys.path if running as a standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def get_models_dict(random_state: int = 42) -> dict[str, tuple[object, str]]:
    """
    Returns dictionary of models to evaluate:
    name -> (model_instance, save_filename)
    """
    return {
        "XGBoost (Surrogate)": (
            xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
            ),
            "xgboost_surrogate.joblib",
        ),
        "Multiple Linear Regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", LinearRegression()),
                ]
            ),
            "multiple_linear_regression.joblib",
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            "random_forest_regressor.joblib",
        ),
        "Support Vector Regression (SVR)": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("reg", SVR(C=1.0, epsilon=0.01)),
                ]
            ),
            "support_vector_regression_svr.joblib",
        ),
        "MLP Regressor": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "reg",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32),
                            max_iter=500,
                            random_state=random_state,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            "mlp_regressor.joblib",
        ),
    }


def train_and_evaluate(
    input_csv_path: Path = None,
    results_dir: Path = None,
    random_state: int = 42,
    num_timing_runs: int = 50,
) -> pd.DataFrame:
    """
    Splits data grouped by center county, trains 5 regression models,
    measures test MAE and per-sample inference latency, and saves models and comparison table.
    """
    if input_csv_path is None:
        input_csv_path = repo_root / "data" / "model_ready.csv"
    if results_dir is None:
        results_dir = repo_root / "results"

    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv_path.exists():
        from src.features import build_model_ready_data
        print(f"{input_csv_path} not found. Constructing features...")
        df = build_model_ready_data(output_csv_path=input_csv_path)
    else:
        df = pd.read_csv(input_csv_path, dtype={"center_fips": str})

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    groups = df["center_fips"]

    # 80/20 Grouped Train/Test Split by center_fips
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_test_samples = len(X_test)
    print("\n" + "=" * 75)
    print("DATASET SPLIT (Grouped by center county FIPS)")
    print("=" * 75)
    print(f"Total samples:       {len(df)}")
    print(f"Training samples:    {len(X_train)} ({len(groups.iloc[train_idx].unique())} unique center counties)")
    print(f"Test samples:        {len(X_test)} ({len(groups.iloc[test_idx].unique())} unique center counties)")
    print("=" * 75)

    models_dict = get_models_dict(random_state=random_state)
    comparison_records = []

    print("\nTraining models and evaluating performance...")
    for model_name, (model, save_name) in models_dict.items():
        # Train model
        model.fit(X_train, y_train)

        # Warmup pass
        _ = model.predict(X_test.head(10))

        # Measure inference latency over test set (excluding training time)
        t_start = time.perf_counter()
        for _ in range(num_timing_runs):
            test_preds = model.predict(X_test)
        t_end = time.perf_counter()

        total_inference_time_ms = (t_end - t_start) * 1000.0
        mean_inference_time_ms = total_inference_time_ms / (num_timing_runs * n_test_samples)

        # Compute test MAE
        mae = float(mean_absolute_error(y_test, test_preds))

        # Save trained model artifact
        model_path = models_dir / save_name
        joblib.dump(model, model_path)

        comparison_records.append(
            {
                "model_name": model_name,
                "mae": mae,
                "mean_inference_time_ms": mean_inference_time_ms,
            }
        )

    # Comparison DataFrame sorted by MAE ascending
    comparison_df = pd.DataFrame(comparison_records).sort_values("mae", ascending=True).reset_index(drop=True)

    # Save comparison table
    comparison_csv_path = results_dir / "comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON TABLE (Sorted by MAE Ascending)")
    print("=" * 75)
    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "mae": "{:.6f}".format,
                "mean_inference_time_ms": "{:.6f}".format,
            },
        )
    )
    print("=" * 75)
    print(f"\nTrained models saved to:    {models_dir}")
    print(f"Comparison table saved to: {comparison_csv_path}\n")

    return comparison_df


if __name__ == "__main__":
    train_and_evaluate()
