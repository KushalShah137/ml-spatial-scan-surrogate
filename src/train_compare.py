import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import beta as beta_distribution

from build_table import add_model_features


ROOT = Path(__file__).resolve().parent.parent
FEATURES = [
    "llr", "log1p(llr)", "n_Z", "mu_Z", "log(mu_Z)", "n_Z/mu_Z",
    "radius_km", "pop_share", "k", "log(N)", "n_counties",
    "log(n_windows)", "log(total_pop)", "pop_gini", "max_pop_share",
    "frac_counties_mu_below_1",
]
WINDOW_FEATURES = FEATURES[:9]
DATASET_FEATURES = FEATURES[9:]
FORBIDDEN = {"sigma", "has_cluster", "cluster_m", "cluster_rr", "observed_max_llr", "seconds_total", "seed", "id", "is_national", "p_value"}
assert not any(feature in FORBIDDEN or feature.startswith("null_max_") for feature in FEATURES)


def stratified_row_sample(frame, maximum, seed):
    if len(frame) <= maximum:
        return frame.copy()
    groups = list(frame.groupby(["id", "sample_stratum"], sort=True).indices.items())
    sizes = np.asarray([len(indices) for _, indices in groups], dtype=int)
    ideal = sizes.astype(float) * maximum / len(frame)
    quotas = np.floor(ideal).astype(int)
    remainder = maximum - int(quotas.sum())
    order = np.argsort(-(ideal - quotas), kind="stable")
    for position in order[:remainder]:
        quotas[position] += 1
    random = np.random.default_rng(seed)
    selected = []
    for (_, indices), quota in zip(groups, quotas):
        if quota:
            selected.extend(random.choice(indices, size=int(quota), replace=False).tolist())
    return frame.iloc[np.sort(selected)].copy()


def split_dataset_ids(metadata, quick):
    ids = np.asarray(sorted(metadata["id"].astype(int).unique()), dtype=int)
    flags = metadata.drop_duplicates("id").set_index("id").loc[ids, "is_national"].astype(int).to_numpy()
    if quick:
        ids, _, flags, _ = train_test_split(ids, flags, train_size=60, random_state=42, stratify=flags)
    train_ids, remaining_ids, train_flags, remaining_flags = train_test_split(
        ids, flags, test_size=0.30, random_state=42, stratify=flags
    )
    val_ids, test_ids = train_test_split(
        remaining_ids, test_size=2.0 / 3.0, random_state=42, stratify=remaining_flags
    )
    return {
        "seed": 42,
        "quick": bool(quick),
        "train": sorted(int(value) for value in train_ids),
        "validation": sorted(int(value) for value in val_ids),
        "test": sorted(int(value) for value in test_ids),
    }


def xgb_model(params, n_estimators, monotone=None):
    arguments = {
        "n_estimators": n_estimators,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "verbosity": 0,
        **params,
    }
    if monotone is not None:
        arguments["monotone_constraints"] = monotone
    return XGBRegressor(**arguments)


def choose_model(model_name, candidates, x_train, y_train, x_val, y_val, tuning_rows):
    best_model = None
    best_score = float("inf")
    best_index = None
    for params, estimator in candidates:
        estimator.fit(x_train, y_train)
        prediction = np.asarray(estimator.predict(x_val), dtype=float)
        score = float(np.mean(np.abs(y_val - prediction)))
        tuning_rows.append({
            "model": model_name,
            "parameters": json.dumps(params, sort_keys=True, separators=(",", ":")),
            "validation_mae_t": score,
            "selected": False,
        })
        if score < best_score:
            best_score = score
            best_model = estimator
            best_index = len(tuning_rows) - 1
    tuning_rows[best_index]["selected"] = True
    return best_model


def fit_models(train, validation, quick, tables_dir, models_dir):
    max_rows = 400000
    train = stratified_row_sample(train, max_rows, 42)
    x_train = train[FEATURES].to_numpy(dtype=float)
    y_train = -np.log10(train["p_value"].to_numpy(dtype=float))
    x_val = validation[FEATURES].to_numpy(dtype=float)
    y_val = -np.log10(validation["p_value"].to_numpy(dtype=float))
    n_trees = 100 if quick else 500
    tuning_rows = []
    models = {}

    grid = [(3, 0.1), (5, 0.05)] if quick else [(3, 0.03), (3, 0.1), (5, 0.03), (5, 0.1)]
    xgb_candidates = [({"max_depth": depth, "learning_rate": rate}, xgb_model({"max_depth": depth, "learning_rate": rate}, n_trees)) for depth, rate in grid]
    models["xgboost"] = (choose_model("xgboost", xgb_candidates, x_train, y_train, x_val, y_val, tuning_rows), FEATURES)

    mlr_candidate = make_pipeline(StandardScaler(), LinearRegression())
    models["mlr"] = (choose_model("mlr", [({"scaler": "standard", "estimator": "linear_regression"}, mlr_candidate)], x_train, y_train, x_val, y_val, tuning_rows), FEATURES)

    forest_trees = 100 if quick else 200
    forest = RandomForestRegressor(n_estimators=forest_trees, min_samples_leaf=20, n_jobs=-1, random_state=42)
    models["random_forest"] = (choose_model("random_forest", [({"n_estimators": forest_trees, "min_samples_leaf": 20}, forest)], x_train, y_train, x_val, y_val, tuning_rows), FEATURES)

    svr_train = stratified_row_sample(train, 20000, 42)
    x_svr = svr_train[FEATURES].to_numpy(dtype=float)
    y_svr = -np.log10(svr_train["p_value"].to_numpy(dtype=float))
    svr_c = [1.0, 10.0] if quick else [0.1, 1.0, 10.0]
    svr_candidates = [
        ({"C": value, "gamma": "scale", "kernel": "rbf", "scaler": "standard"}, make_pipeline(StandardScaler(), SVR(C=value, gamma="scale", kernel="rbf")))
        for value in svr_c
    ]
    models["svr"] = (choose_model("svr", svr_candidates, x_svr, y_svr, x_val, y_val, tuning_rows), FEATURES)

    alphas = [0.0001, 0.001] if quick else [0.0001, 0.001, 0.01]
    mlp_candidates = [
        ({"hidden_layer_sizes": [64, 32], "alpha": alpha, "early_stopping": True}, make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=alpha, early_stopping=True, max_iter=200, n_iter_no_change=12, random_state=42)
        ))
        for alpha in alphas
    ]
    models["mlp"] = (choose_model("mlp", mlp_candidates, x_train, y_train, x_val, y_val, tuning_rows), FEATURES)

    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0)
    isotonic.fit(train["llr"].to_numpy(dtype=float), y_train)
    iso_prediction = isotonic.predict(validation["llr"].to_numpy(dtype=float))
    tuning_rows.append({"model": "isotonic", "parameters": json.dumps({"feature": "llr", "increasing": True}), "validation_mae_t": float(np.mean(np.abs(y_val - iso_prediction))), "selected": True})
    models["isotonic"] = (isotonic, ["llr"])

    x_window_train = train[WINDOW_FEATURES].to_numpy(dtype=float)
    x_window_val = validation[WINDOW_FEATURES].to_numpy(dtype=float)
    no_dataset_candidates = [({"max_depth": depth, "learning_rate": rate, "features": "window_only"}, xgb_model({"max_depth": depth, "learning_rate": rate}, n_trees)) for depth, rate in grid]
    models["xgboost_no_dataset_features"] = (
        choose_model("xgboost_no_dataset_features", no_dataset_candidates, x_window_train, y_train, x_window_val, y_val, tuning_rows), WINDOW_FEATURES
    )

    monotone = tuple(1 if feature == "llr" else 0 for feature in FEATURES)
    monotone_candidates = [({"max_depth": depth, "learning_rate": rate, "llr_constraint": "increasing"}, xgb_model({"max_depth": depth, "learning_rate": rate}, n_trees, monotone)) for depth, rate in grid]
    models["xgboost_monotone_llr"] = (
        choose_model("xgboost_monotone_llr", monotone_candidates, x_train, y_train, x_val, y_val, tuning_rows), FEATURES
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tuning_rows).to_csv(tables_dir / "tuning.csv", index=False)
    for name, (model, _) in models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")
    return models, len(train), len(svr_train), pd.DataFrame(tuning_rows)


def baseline_predictions(frame, dataset_type):
    result = {}
    ids = frame["id"].unique()
    for dataset_id in ids:
        if dataset_type == "synthetic":
            null_path = ROOT / "data" / "synth" / f"ds_{int(dataset_id)}_null_max.npy"
        else:
            null_path = ROOT / "data" / "processed" / f"{dataset_id}_c10_null_max.npy"
        null_values = np.load(null_path)[:99]
        full_null = np.load(null_path)
        sorted_null = np.sort(null_values)
        rows = frame["id"].eq(dataset_id)
        llr = frame.loc[rows, "llr"].to_numpy(dtype=float)
        mc = (1 + len(sorted_null) - np.searchsorted(sorted_null, llr, side="left")) / 100.0
        std = float(np.std(null_values))
        beta = std * math.sqrt(6.0) / math.pi
        if beta <= 0:
            gumbel = np.ones(len(llr), dtype=float)
        else:
            mu = float(np.mean(null_values)) - 0.5772 * beta
            z = np.clip((llr - mu) / beta, -700.0, 700.0)
            gumbel = 1.0 - np.exp(-np.exp(-z))
        full_mu, full_beta = gumbel_moments(full_null)
        full_z = np.clip((llr - full_mu) / full_beta, -700.0, 700.0)
        gumbel_full = 1.0 - np.exp(-np.exp(-full_z))
        gumbel_full[llr == 0] = 1.0
        result[dataset_id] = (np.clip(mc, 0.001, 1.0), np.clip(gumbel, 0.001, 1.0), np.clip(gumbel_full, 0.001, 1.0))
    mc_values = np.empty(len(frame), dtype=float)
    gumbel_values = np.empty(len(frame), dtype=float)
    full_gumbel_values = np.empty(len(frame), dtype=float)
    for dataset_id, (mc, gumbel, full_gumbel) in result.items():
        positions = np.flatnonzero(frame["id"].to_numpy() == dataset_id)
        mc_values[positions] = mc
        gumbel_values[positions] = gumbel
        full_gumbel_values[positions] = full_gumbel
    return mc_values, gumbel_values, full_gumbel_values


def gumbel_moments(null_values):
    values = np.asarray(null_values, dtype=float)
    beta = float(np.std(values)) * math.sqrt(6.0) / math.pi
    beta = max(beta, np.finfo(float).tiny)
    mu = float(np.mean(values)) - 0.5772 * beta
    return mu, beta


def predict_gumbel_parameters(bundle, frame):
    unique = frame.drop_duplicates("id").set_index("id")
    x = unique[DATASET_FEATURES].to_numpy(dtype=float)
    output = {}
    for family, model in bundle["models"].items():
        values = model.predict(x)
        mu = np.asarray(values[:, 0], dtype=float)
        beta = np.exp(np.clip(np.asarray(values[:, 1], dtype=float), -30.0, 30.0))
        output[family] = {dataset_id: (float(mu[i]), float(beta[i])) for i, dataset_id in enumerate(unique.index)}
    return output


def probabilities_from_gumbel(llr, mu, beta):
    scores = np.asarray(llr, dtype=float)
    z = np.clip((scores - float(mu)) / max(float(beta), np.finfo(float).tiny), -700.0, 700.0)
    probabilities = 1.0 - np.exp(-np.exp(-z))
    probabilities[scores == 0] = 1.0
    return np.clip(probabilities, 0.001, 1.0)


def metadata_feature_vector(record):
    return [
        math.log(float(record["N"])),
        float(record["n_counties"]),
        math.log(float(record["n_windows"])),
        math.log(float(record["total_pop"])),
        float(record["pop_gini"]),
        float(record["max_pop_share"]),
        float(record["frac_counties_mu_below_1"]),
    ]


def make_gumbel_estimator(family, parameters, quick):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(parameters["alpha"])))
    estimator = XGBRegressor(
        n_estimators=100 if quick else 200,
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=int(parameters["max_depth"]),
        learning_rate=0.05,
        n_jobs=1,
        random_state=42,
        subsample=0.9,
        colsample_bytree=1.0,
        verbosity=0,
    )
    return MultiOutputRegressor(estimator, n_jobs=1)


def fit_gumbel_regressors(metadata, train_ids, validation_ids, test_ids, quick, tables_dir, models_dir):
    records = {int(record["id"]): record for record in metadata}
    selected_ids = sorted(set(train_ids) | set(validation_ids) | set(test_ids))
    feature_map = {dataset_id: metadata_feature_vector(records[dataset_id]) for dataset_id in selected_ids}
    target_map = {}
    for dataset_id in selected_ids:
        values = np.load(ROOT / "data" / "synth" / f"ds_{dataset_id}_null_max.npy")
        mu, beta = gumbel_moments(values)
        target_map[dataset_id] = (mu, beta)
    x_train = np.asarray([feature_map[dataset_id] for dataset_id in train_ids], dtype=float)
    y_train = np.asarray([[target_map[dataset_id][0], math.log(target_map[dataset_id][1])] for dataset_id in train_ids], dtype=float)
    x_validation = np.asarray([feature_map[dataset_id] for dataset_id in validation_ids], dtype=float)
    validation_mu = np.asarray([target_map[dataset_id][0] for dataset_id in validation_ids], dtype=float)
    validation_beta = np.asarray([target_map[dataset_id][1] for dataset_id in validation_ids], dtype=float)
    candidates = []
    for alpha in (0.1, 1.0, 10.0):
        candidates.append(("ridge", {"alpha": alpha}, make_gumbel_estimator("ridge", {"alpha": alpha}, quick)))
    for depth in (2, 3):
        candidates.append(("xgb", {"max_depth": depth}, make_gumbel_estimator("xgb", {"max_depth": depth}, quick)))
    rows = []
    family_winners = {}
    scale_mu = max(float(np.std(validation_mu)), 1e-12)
    scale_beta = max(float(np.std(validation_beta)), 1e-12)
    for family, parameters, model in candidates:
        model.fit(x_train, y_train)
        prediction = np.asarray(model.predict(x_validation), dtype=float)
        predicted_mu = prediction[:, 0]
        predicted_beta = np.exp(np.clip(prediction[:, 1], -30.0, 30.0))
        mae_mu = float(np.mean(np.abs(validation_mu - predicted_mu)))
        mae_beta = float(np.mean(np.abs(validation_beta - predicted_beta)))
        score = 0.5 * (mae_mu / scale_mu + mae_beta / scale_beta)
        rows.append({"family": family, "parameters": json.dumps(parameters, sort_keys=True), "validation_mu_mae": mae_mu, "validation_beta_mae": mae_beta, "validation_score": score, "selected": False})
        if family not in family_winners or score < family_winners[family][0]:
            family_winners[family] = (score, parameters, model)
    for family, (score, parameters, model) in family_winners.items():
        selected_row = min((row for row in rows if row["family"] == family), key=lambda row: row["validation_score"])
        selected_row["selected"] = True
    best_family = min(family_winners, key=lambda family: family_winners[family][0])
    bundle = {
        "models": {family: value[2] for family, value in family_winners.items()},
        "parameters": {family: value[1] for family, value in family_winners.items()},
        "best_family": best_family,
        "best_parameters": family_winners[best_family][1],
        "feature_map": feature_map,
        "target_map": target_map,
    }
    test_rows = []
    test_x = np.asarray([feature_map[dataset_id] for dataset_id in test_ids], dtype=float)
    for family, model in bundle["models"].items():
        predicted = np.asarray(model.predict(test_x), dtype=float)
        predicted_mu = predicted[:, 0]
        predicted_beta = np.exp(np.clip(predicted[:, 1], -30.0, 30.0))
        actual_mu = np.asarray([target_map[dataset_id][0] for dataset_id in test_ids], dtype=float)
        actual_beta = np.asarray([target_map[dataset_id][1] for dataset_id in test_ids], dtype=float)
        for target, actual, estimate in (("mu", actual_mu, predicted_mu), ("beta", actual_beta, predicted_beta)):
            denominator = float(np.sum((actual - actual.mean()) ** 2))
            r_squared = 1.0 - float(np.sum((actual - estimate) ** 2)) / denominator if denominator > 0 else np.nan
            test_rows.append({"model": f"gumbel_pred_{family}", "target": target, "n_test_datasets": len(test_ids), "mae": float(np.mean(np.abs(actual - estimate))), "r_squared": r_squared})
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables_dir / "gumbel_tuning.csv", index=False)
    pd.DataFrame(test_rows).to_csv(tables_dir / "gumbel_regression_quality.csv", index=False)
    for family, model in bundle["models"].items():
        joblib.dump(model, models_dir / f"gumbel_pred_{family}.joblib")
    return bundle, pd.DataFrame(test_rows)


def learning_curve(train, test, train_ids, metadata, gumbel_bundle, quick, tables_dir):
    metadata_by_id = {int(record["id"]): record for record in metadata}
    targets = gumbel_bundle["target_map"]
    available = len(train_ids)
    specifications = [(25, 3), (50, 3), (100, 3), (200, 3), ("all", 1)]
    rows = []
    seed = 42
    y_test = -np.log10(test["p_value"].to_numpy(dtype=float))
    true_p = test["p_value"].to_numpy(dtype=float)
    hard_band = (true_p >= 0.01) & (true_p <= 0.10)
    true_significant = true_p < 0.05
    for requested, repeats in specifications:
        size = available if requested == "all" else min(int(requested), available)
        for repeat in range(repeats):
            current_seed = seed
            seed += 1
            random = np.random.default_rng(current_seed)
            sampled_ids = sorted(random.choice(np.asarray(train_ids, dtype=int), size=size, replace=False).tolist())
            sampled_rows = stratified_row_sample(train[train["id"].isin(sampled_ids)].copy(), 400000, 42)
            xgb_source = gumbel_bundle["xgboost_reference"]
            xgb_parameters = xgb_source.get_params()
            xgb_fit = xgb_model({"max_depth": xgb_parameters["max_depth"], "learning_rate": xgb_parameters["learning_rate"]}, 100 if quick else 500)
            xgb_fit.fit(sampled_rows[FEATURES].to_numpy(dtype=float), -np.log10(sampled_rows["p_value"].to_numpy(dtype=float)))
            xgb_t = np.asarray(xgb_fit.predict(test[FEATURES].to_numpy(dtype=float)), dtype=float)
            xgb_p = np.clip(np.power(10.0, -np.clip(xgb_t, -20.0, 20.0)), 0.001, 1.0)
            xgb_p[test["llr"].to_numpy(dtype=float) == 0] = 1.0

            family = gumbel_bundle["best_family"]
            parameters = gumbel_bundle["best_parameters"]
            estimator = make_gumbel_estimator(family, parameters, quick)
            g_x = np.asarray([gumbel_bundle["feature_map"][dataset_id] for dataset_id in sampled_ids], dtype=float)
            g_y = np.asarray([[targets[dataset_id][0], math.log(targets[dataset_id][1])] for dataset_id in sampled_ids], dtype=float)
            estimator.fit(g_x, g_y)
            test_unique = test.drop_duplicates("id").set_index("id")
            predicted_params = estimator.predict(test_unique[DATASET_FEATURES].to_numpy(dtype=float))
            parameter_map = {
                dataset_id: (float(value[0]), float(np.exp(np.clip(value[1], -30.0, 30.0))))
                for dataset_id, value in zip(test_unique.index, predicted_params)
            }
            gumbel_p = np.empty(len(test), dtype=float)
            for dataset_id, (mu, beta) in parameter_map.items():
                positions = np.flatnonzero(test["id"].to_numpy() == dataset_id)
                gumbel_p[positions] = probabilities_from_gumbel(test.iloc[positions]["llr"].to_numpy(dtype=float), mu, beta)
            for method, predicted_p in (("xgboost", xgb_p), (f"gumbel_pred_{family}", gumbel_p)):
                mae_band = float(np.mean(np.abs(true_p[hard_band] - predicted_p[hard_band]))) if hard_band.any() else np.nan
                predicted_significant = predicted_p < 0.05
                tp = int(np.sum(true_significant & predicted_significant))
                fp = int(np.sum(~true_significant & predicted_significant))
                fn = int(np.sum(true_significant & ~predicted_significant))
                f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                rows.append({"method": method, "requested_train_datasets": requested, "n_train_datasets": size, "repeat": repeat + 1, "seed": current_seed, "band_mae_p": mae_band, "f1_at_0_05": f1, "n_test_windows": len(test)})
    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "learning_curve.csv", index=False)
    return result


def predict_table(models, frame, gumbel_bundle):
    result = pd.DataFrame({
        "id": frame["id"].to_numpy(),
        "is_national": frame["is_national"].to_numpy(),
        "has_cluster": frame["has_cluster"].to_numpy(),
        "sigma": frame["sigma"].to_numpy(),
        "sample_stratum": frame["sample_stratum"].to_numpy(),
        "llr": frame["llr"].to_numpy(dtype=float),
        "p_value": frame["p_value"].to_numpy(dtype=float),
    })
    result["true_t"] = -np.log10(result["p_value"].to_numpy(dtype=float))
    predictions = {}
    for name, (model, features) in models.items():
        raw_t = np.asarray(model.predict(frame[features].to_numpy(dtype=float)), dtype=float) if name != "isotonic" else np.asarray(model.predict(frame["llr"].to_numpy(dtype=float)), dtype=float)
        p = np.clip(np.power(10.0, -np.clip(raw_t, -20.0, 20.0)), 0.001, 1.0)
        predictions[name] = (raw_t, p)
        result[f"pred_t_{name}"] = raw_t
        result[f"pred_p_{name}"] = p
    mc, gumbel, gumbel_full = baseline_predictions(frame, "synthetic" if frame["id"].dtype.kind in "iu" else "real")
    predictions["mc99"] = (-np.log10(mc), mc)
    predictions["gumbel99"] = (-np.log10(gumbel), gumbel)
    predictions["gumbel999"] = (-np.log10(gumbel_full), gumbel_full)
    result["pred_p_mc99"] = mc
    result["pred_p_gumbel99"] = gumbel
    result["pred_p_gumbel999"] = gumbel_full
    fitted_parameters = predict_gumbel_parameters(gumbel_bundle, frame)
    for family, dataset_values in fitted_parameters.items():
        p_values = np.empty(len(frame), dtype=float)
        for dataset_id, (mu, beta) in dataset_values.items():
            positions = np.flatnonzero(frame["id"].to_numpy() == dataset_id)
            p_values[positions] = probabilities_from_gumbel(frame.iloc[positions]["llr"].to_numpy(dtype=float), mu, beta)
        method = f"gumbel_pred_{family}"
        predictions[method] = (-np.log10(p_values), p_values)
        result[f"pred_p_{method}"] = p_values
    return result, predictions


def calculate_metrics(true_p, true_t, pred_p, pred_t):
    true_p = np.asarray(true_p, dtype=float)
    true_t = np.asarray(true_t, dtype=float)
    pred_p = np.asarray(pred_p, dtype=float)
    pred_t = np.asarray(pred_t, dtype=float)
    actual = true_p < 0.05
    predicted = pred_p < 0.05
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    band = (true_p >= 0.01) & (true_p <= 0.10)
    return {
        "n": len(true_p),
        "mae_t": float(np.mean(np.abs(true_t - pred_t))) if len(true_p) else np.nan,
        "rmse_t": float(np.sqrt(np.mean((true_t - pred_t) ** 2))) if len(true_p) else np.nan,
        "mae_p": float(np.mean(np.abs(true_p - pred_p))) if len(true_p) else np.nan,
        "mae_p_0_01_to_0_10": float(np.mean(np.abs(true_p[band] - pred_p[band]))) if band.any() else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
    }


def binomial_interval(successes, trials, confidence=0.95):
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta_distribution.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


def classification_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    union = int(np.sum(actual | predicted))
    return f1, tp / union if union else 1.0, tp, fp, fn


def evaluate_windows(scope, dataset_name, predictions_frame, predictions):
    rows = []
    for filter_name in ("all", "llr_positive"):
        mask_filter = np.ones(len(predictions_frame), dtype=bool) if filter_name == "all" else predictions_frame["llr"].to_numpy(dtype=float) > 0
        for stratum in ("all", "top", "random"):
            mask_stratum = np.ones(len(predictions_frame), dtype=bool) if stratum == "all" else predictions_frame["sample_stratum"].to_numpy() == stratum
            mask = mask_filter & mask_stratum
            for model_name, (pred_t, pred_p) in predictions.items():
                metrics = calculate_metrics(
                    predictions_frame.loc[mask, "p_value"],
                    predictions_frame.loc[mask, "true_t"],
                    pred_p[mask],
                    pred_t[mask],
                )
                rows.append({"scope": scope, "dataset": dataset_name, "window_filter": filter_name, "sample_stratum": stratum, "model": model_name, **metrics})
    return rows


def dataset_level_metrics(scope, dataset_name, predictions_frame, predictions):
    selected = predictions_frame.groupby("id", sort=False)["llr"].idxmax()
    rows = []
    for index in selected:
        row = predictions_frame.loc[index]
        dataset_id = row["id"]
        true_p = float(row["p_value"])
        true_log = math.log10(max(true_p, 0.001))
        for model_name, (pred_t, pred_p) in predictions.items():
            position = predictions_frame.index.get_loc(index)
            predicted_p = float(pred_p[position])
            rows.append({
                "scope": scope,
                "dataset": dataset_name if scope == "real" else int(dataset_id),
                "id": dataset_id,
                "model": model_name,
                "true_p": true_p,
                "predicted_p": predicted_p,
                "absolute_log10_p_error": abs(math.log10(max(predicted_p, 0.001)) - true_log),
                "true_significant_0_05": bool(true_p < 0.05),
                "predicted_significant_0_05": bool(predicted_p < 0.05),
                "agreement_0_05": bool((true_p < 0.05) == (predicted_p < 0.05)),
                "has_cluster": row["has_cluster"],
                "sigma": row["sigma"],
            })
    return rows


def make_real_all_windows(name, models, threshold, gumbel_bundle):
    from scan import dataset_features
    counties = pd.read_csv(ROOT / "data" / "processed" / f"{name}_county_primary.csv")
    windows = pd.read_csv(ROOT / "data" / "processed" / f"{name}_c10_windows.csv.gz")
    windows["sample_stratum"] = "all"
    stats = dataset_features(counties["population"].to_numpy(), counties["deaths"].to_numpy(), len(windows))
    frame = add_model_features(windows, stats)
    frame.insert(0, "sigma", np.nan)
    frame.insert(0, "has_cluster", np.nan)
    frame.insert(0, "is_national", True)
    frame.insert(0, "id", name)
    true_p = frame["p_value"].to_numpy(dtype=float)
    full_null = np.load(ROOT / "data" / "processed" / f"{name}_c10_null_max.npy")
    null_values = full_null[:99]
    ordered_null = np.sort(null_values)
    mc99 = (1.0 + len(ordered_null) - np.searchsorted(ordered_null, frame["llr"].to_numpy(dtype=float), side="left")) / 100.0
    deviation = float(np.std(null_values))
    beta = deviation * math.sqrt(6.0) / math.pi
    if beta <= 0:
        gumbel99 = np.ones(len(frame), dtype=float)
    else:
        location = float(np.mean(null_values)) - 0.5772 * beta
        z = np.clip((frame["llr"].to_numpy(dtype=float) - location) / beta, -700.0, 700.0)
        gumbel99 = 1.0 - np.exp(-np.exp(-z))
    predictions = {}
    xgb_model_fit, xgb_features = models["xgboost"]
    xgb_t = np.asarray(xgb_model_fit.predict(frame[xgb_features].to_numpy(dtype=float)), dtype=float)
    predictions["xgboost_per_window"] = np.clip(np.power(10.0, -np.clip(xgb_t, -20.0, 20.0)), 0.001, 1.0)
    isotonic = models["isotonic"][0]
    isotonic_t = np.asarray(isotonic.predict(frame["llr"].to_numpy(dtype=float)), dtype=float)
    predictions["isotonic"] = np.clip(np.power(10.0, -np.clip(isotonic_t, -20.0, 20.0)), 0.001, 1.0)
    predictions["gumbel99"] = np.clip(gumbel99, 0.001, 1.0)
    predictions["mc99"] = np.clip(mc99, 0.001, 1.0)
    full_mu, full_beta = gumbel_moments(full_null)
    predictions["gumbel999"] = probabilities_from_gumbel(frame["llr"].to_numpy(dtype=float), full_mu, full_beta)
    fitted_params = predict_gumbel_parameters(gumbel_bundle, frame)
    for family, dataset_values in fitted_params.items():
        mu, beta = dataset_values[name]
        predictions[f"gumbel_pred_{family}"] = probabilities_from_gumbel(frame["llr"].to_numpy(dtype=float), mu, beta)
    predictions["constant_threshold"] = frame["llr"].to_numpy(dtype=float) > threshold
    actual = true_p < 0.05
    hard = (true_p >= 0.005) & (true_p <= 0.2)
    rows = []
    for method, prediction in predictions.items():
        predicted_significant = prediction if method == "constant_threshold" else prediction < 0.05
        f1, jaccard, tp, fp, fn = classification_metrics(actual, predicted_significant)
        hard_f1, hard_jaccard, hard_tp, hard_fp, hard_fn = classification_metrics(actual[hard], np.asarray(predicted_significant)[hard])
        if method == "constant_threshold":
            hard_mae = np.nan
        else:
            hard_mae = float(np.mean(np.abs(-np.log10(true_p[hard]) + np.log10(prediction[hard])))) if hard.any() else np.nan
        rows.append({
            "dataset": name,
            "method": method,
            "n_windows": len(frame),
            "f1": f1,
            "jaccard": jaccard,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "hard_window_count": int(hard.sum()),
            "hard_f1": hard_f1,
            "hard_jaccard": hard_jaccard,
            "hard_mae_log10_p": hard_mae,
            "hard_tp": hard_tp,
            "hard_fp": hard_fp,
            "hard_fn": hard_fn,
            "threshold_llr": threshold if method == "constant_threshold" else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--quick", action="store_true")
    choice.add_argument("--full", action="store_true")
    args = parser.parse_args()
    tables_dir = ROOT / "results" / "tables"
    models_dir = ROOT / "results" / "models"
    predictions_dir = ROOT / "results" / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    synthetic = pd.read_csv(ROOT / "data" / "train" / "synthetic_table.csv.gz")
    split = split_dataset_ids(synthetic[["id", "is_national"]], args.quick)
    (ROOT / "results" / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    train_ids = set(split["train"])
    validation_ids = set(split["validation"])
    test_ids = set(split["test"])
    train = synthetic.loc[synthetic["id"].isin(train_ids)].copy()
    validation = synthetic.loc[synthetic["id"].isin(validation_ids)].copy()
    print(f"Dataset split: train={len(train_ids)}, validation={len(validation_ids)}, test={len(test_ids)} datasets")
    print(f"Window rows before cap: train={len(train)}, validation={len(validation)}")

    models, train_rows, svr_rows, tuning = fit_models(train, validation, args.quick, tables_dir, models_dir)
    print(f"Training rows used: {train_rows}; SVR rows used: {svr_rows}")
    print(f"Tuning rows: {len(tuning)}")
    with (ROOT / "data" / "synth" / "meta.jsonl").open(encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle if line.strip()]
    gumbel_bundle, gumbel_quality = fit_gumbel_regressors(metadata, sorted(train_ids), sorted(validation_ids), sorted(test_ids), args.quick, tables_dir, models_dir)
    gumbel_bundle["xgboost_reference"] = models["xgboost"][0]
    for _, row in gumbel_quality.iterrows():
        print(f"{row['model']} {row['target']}: test MAE={row['mae']:.6f}, R-squared={row['r_squared']:.6f}")
    test = synthetic.loc[synthetic["id"].isin(test_ids)].copy().reset_index(drop=True)
    test_predictions, test_model_predictions = predict_table(models, test, gumbel_bundle)
    window_rows = evaluate_windows("synthetic_test", "synthetic_test", test_predictions, test_model_predictions)
    dataset_rows = dataset_level_metrics("synthetic_test", "synthetic_test", test_predictions, test_model_predictions)
    test_predictions.to_csv(predictions_dir / "synthetic_test_predictions.csv.gz", index=False, compression="gzip")
    print(f"Synthetic test rows: {len(test_predictions)}; test datasets: {len(test_ids)}")

    real_sample_tables = {}
    with (ROOT / "config" / "datasets.json").open(encoding="utf-8") as handle:
        datasets = json.load(handle)
    for dataset in datasets:
        name = dataset["name"]
        frame = pd.read_csv(ROOT / "data" / "train" / f"real_{name}_table.csv.gz").reset_index(drop=True)
        real_sample_tables[name] = frame
        prediction_frame, model_predictions = predict_table(models, frame, gumbel_bundle)
        prediction_frame.to_csv(predictions_dir / f"real_{name}_predictions.csv.gz", index=False, compression="gzip")
        window_rows.extend(evaluate_windows("real", name, prediction_frame, model_predictions))
        dataset_rows.extend(dataset_level_metrics("real", name, prediction_frame, model_predictions))
        print(f"Real {name} sampled rows: {len(frame)}")

    type1_rows = []
    main_xgb = "xgboost"
    for group_name, selector in (
        ("type1", (test_predictions["has_cluster"].eq(False) & test_predictions["sigma"].eq(0.0))),
        ("power", test_predictions["has_cluster"].eq(True)),
    ):
        group = test_predictions.loc[selector]
        if group.empty:
            print(f"Synthetic {group_name} datasets: 0")
            continue
        top_indices = group.groupby("id")["llr"].idxmax()
        top = group.loc[top_indices]
        xgb_column = f"pred_p_{main_xgb}"
        for method, values in (
            ("true_monte_carlo", top["p_value"].to_numpy(dtype=float)),
            ("xgboost_per_window", top[xgb_column].to_numpy(dtype=float)),
            ("gumbel99", top["pred_p_gumbel99"].to_numpy(dtype=float)),
            ("gumbel999", top["pred_p_gumbel999"].to_numpy(dtype=float)),
            ("mc99", top["pred_p_mc99"].to_numpy(dtype=float)),
            ("gumbel_pred_ridge", top["pred_p_gumbel_pred_ridge"].to_numpy(dtype=float)),
            ("gumbel_pred_xgb", top["pred_p_gumbel_pred_xgb"].to_numpy(dtype=float)),
        ):
            successes = int(np.sum(values < 0.05))
            trials = len(values)
            lower, upper = binomial_interval(successes, trials)
            type1_rows.append({"group": group_name, "method": method, "n_datasets": trials, "significant_count": successes, "fraction_significant": successes / trials, "ci95_low": lower, "ci95_high": upper})
        print(f"Synthetic {group_name} datasets: {len(top)}")

    real_all_rows = []
    training_meta = [record for record in metadata if int(record["id"]) in train_ids]
    threshold = float(np.mean([record["null_max_q95"] for record in training_meta]))
    for dataset in datasets:
        name = dataset["name"]
        rows = make_real_all_windows(name, models, threshold, gumbel_bundle)
        real_all_rows.extend(rows)
        for row in rows:
            print(f"Real {name} all windows {row['method']}: F1={row['f1']:.6f}, Jaccard={row['jaccard']:.6f}; hard n={row['hard_window_count']}, F1={row['hard_f1']:.6f}, Jaccard={row['hard_jaccard']:.6f}, MAE log10 p={row['hard_mae_log10_p']}")

    learning = learning_curve(train, test, sorted(train_ids), metadata, gumbel_bundle, args.quick, tables_dir)
    print(f"Saved learning curve: {tables_dir / 'learning_curve.csv'} ({len(learning)} rows)")

    pd.DataFrame(window_rows).to_csv(tables_dir / "window_metrics.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(tables_dir / "dataset_metrics.csv", index=False)
    pd.DataFrame(type1_rows).to_csv(tables_dir / "type1_power.csv", index=False)
    pd.DataFrame(real_all_rows).to_csv(tables_dir / "real_all_windows.csv", index=False)
    print(f"Saved tuning grid: {tables_dir / 'tuning.csv'} ({len(tuning)} rows)")
    print(f"Saved window metrics: {tables_dir / 'window_metrics.csv'} ({len(window_rows)} rows)")
    print(f"Saved dataset metrics: {tables_dir / 'dataset_metrics.csv'} ({len(dataset_rows)} rows)")
    print(f"Saved type-I/power metrics: {tables_dir / 'type1_power.csv'} ({len(type1_rows)} rows)")
    print(f"Saved real all-window metrics: {tables_dir / 'real_all_windows.csv'} ({len(real_all_rows)} rows)")
    print(f"Saved predictions under: {predictions_dir}")


if __name__ == "__main__":
    main()
