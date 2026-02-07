"""
Model Zoo — Ridge, Random Forest, Gradient Boosting, MLP

Shared evaluation helper and hyperparameter configurations for
all 1-step regression baselines.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


def eval_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test,
               target_cols):
    """Fit model, compute overall and per-channel RMSE on val + test.

    Args:
        name: Model name string for display.
        model: sklearn-compatible estimator (unfitted).
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        target_cols: List of target channel names.

    Returns:
        Dict with model name, val/test RMSE (overall and per-channel).
    """
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    mse_val_ch = ((y_val - y_val_pred) ** 2).mean(axis=0)
    mse_test_ch = ((y_test - y_test_pred) ** 2).mean(axis=0)
    rmse_val_ch = np.sqrt(mse_val_ch)
    rmse_test_ch = np.sqrt(mse_test_ch)

    print(f"\n=== {name} ===")
    print(f"Val RMSE : {rmse_val:.6f}")
    print(f"Test RMSE: {rmse_test:.6f}")
    for col, rv, rt in zip(target_cols, rmse_val_ch, rmse_test_ch):
        print(f"  {col:>3}: Val = {rv:.4f}, Test = {rt:.4f}")

    res = {"model": name, "val_rmse": rmse_val, "test_rmse": rmse_test}
    for col, rv, rt in zip(target_cols, rmse_val_ch, rmse_test_ch):
        res[f"val_{col}"] = rv
        res[f"test_{col}"] = rt
    return res


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

RF_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5},
]

GB_PARAM_GRID = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
]

MLP_PARAM_GRID = [
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-4},
    {"hidden_layer_sizes": (64, 64), "alpha": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4},
    {"hidden_layer_sizes": (128, 64, 32), "alpha": 1e-3},
]


def tune_ridge(X_train, y_train, X_val, y_val, X_test, y_test, target_cols,
               alphas=None):
    """Sweep Ridge alpha and return best result dict."""
    if alphas is None:
        alphas = RIDGE_ALPHAS

    results = []
    for a in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=a))
        res = eval_model(f"Ridge(α={a})", model,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         target_cols)
        res["alpha"] = a
        results.append(res)

    return min(results, key=lambda d: d["val_rmse"]), results


def tune_random_forest(X_train, y_train, X_val, y_val, X_test, y_test,
                       target_cols, param_grid=None):
    """Sweep RF hyperparameters and return best result dict."""
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    results = []
    for params in param_grid:
        model = make_pipeline(StandardScaler(), RandomForestRegressor(**params, random_state=42))
        name = f"RF(n={params['n_estimators']}, d={params['max_depth']}, leaf={params['min_samples_leaf']})"
        res = eval_model(name, model,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         target_cols)
        res.update(params)
        results.append(res)

    return min(results, key=lambda d: d["val_rmse"]), results


def tune_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test,
                           target_cols, param_grid=None):
    """Sweep GB hyperparameters (wrapped as MultiOutput) and return best result dict."""
    if param_grid is None:
        param_grid = GB_PARAM_GRID

    results = []
    for params in param_grid:
        base = GradientBoostingRegressor(**params, random_state=42)
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(base))
        name = f"GB(n={params['n_estimators']}, lr={params['learning_rate']}, d={params['max_depth']})"
        res = eval_model(name, model,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         target_cols)
        res.update(params)
        results.append(res)

    return min(results, key=lambda d: d["val_rmse"]), results


def tune_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
             target_cols, param_grid=None):
    """Sweep MLP hyperparameters and return best result dict."""
    if param_grid is None:
        param_grid = MLP_PARAM_GRID

    results = []
    for params in param_grid:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(**params, max_iter=500, early_stopping=True,
                         validation_fraction=0.1, random_state=42)
        )
        name = f"MLP(layers={params['hidden_layer_sizes']}, α={params['alpha']})"
        res = eval_model(name, model,
                         X_train, y_train, X_val, y_val, X_test, y_test,
                         target_cols)
        res.update(params)
        results.append(res)

    return min(results, key=lambda d: d["val_rmse"]), results
