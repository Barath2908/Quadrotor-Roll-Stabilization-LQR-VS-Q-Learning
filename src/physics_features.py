"""
Physics-Guided Feature Engineering for Attitude Dynamics

Constructs torque-aligned features from raw motor commands based on
standard quadrotor dynamics (X-configuration assumed).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-guided torque features to the flight DataFrame.

    For a standard X-configuration quadrotor:
      - u_thrust = u1 + u2 + u3 + u4       (total thrust)
      - u_roll   = u2 - u4                  (roll torque proxy)
      - u_pitch  = u3 - u1                  (pitch torque proxy)
      - u_yaw    = u1 - u2 + u3 - u4        (yaw torque proxy)

    Args:
        df: DataFrame with columns u1, u2, u3, u4.

    Returns:
        DataFrame with added physics feature columns.
    """
    df = df.copy()
    df["u_thrust"] = df["u1"] + df["u2"] + df["u3"] + df["u4"]
    df["u_roll"] = df["u2"] - df["u4"]
    df["u_pitch"] = df["u3"] - df["u1"]
    df["u_yaw"] = df["u1"] - df["u2"] + df["u3"] - df["u4"]
    return df


def build_attitude_datasets(df: pd.DataFrame):
    """Build monolithic and physics-guided feature sets for attitude prediction.

    Monolithic features: [p, q, r, u1, u2, u3, u4]
    Physics features:    [p, q, r, u_thrust, u_roll, u_pitch, u_yaw]

    Target: [p_next, q_next, r_next]

    Args:
        df: DataFrame with all required columns (call add_physics_features first).

    Returns:
        Tuple of (X_mono, X_phys, y_att, t_att, feat_mono_cols, feat_phys_cols, att_state_cols).
    """
    att_state_cols = ["p", "q", "r"]

    feat_mono_cols = ["p", "q", "r", "u1", "u2", "u3", "u4"]
    feat_phys_cols = ["p", "q", "r", "u_thrust", "u_roll", "u_pitch", "u_yaw"]

    data_mono = df[["t"] + feat_mono_cols + att_state_cols].dropna().values
    data_phys = df[["t"] + feat_phys_cols + att_state_cols].dropna().values

    # 1-step pairs
    t_att = data_mono[:-1, 0]
    X_mono = data_mono[:-1, 1:1 + len(feat_mono_cols)]
    X_phys = data_phys[:-1, 1:1 + len(feat_phys_cols)]
    y_att = data_mono[1:, 1 + len(feat_mono_cols):]

    return X_mono, X_phys, y_att, t_att, feat_mono_cols, feat_phys_cols, att_state_cols


def tune_ridge_for_features(feature_set_name, X_tr, y_tr, X_va, y_va, X_te, y_te,
                            att_state_cols, alphas=None):
    """Tune Ridge alpha for a given feature set (monolithic or physics-guided).

    Args:
        feature_set_name: Name string for display.
        X_tr, y_tr: Training data.
        X_va, y_va: Validation data.
        X_te, y_te: Test data.
        att_state_cols: Target column names.
        alphas: List of alpha values to try.

    Returns:
        Best result dict with feature_set, alpha, and RMSE values.
    """
    from src.models import eval_model

    if alphas is None:
        alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

    results = []
    for a in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=a))
        name = f"Ridge(α={a}) [{feature_set_name}]"
        res = eval_model(name, model, X_tr, y_tr, X_va, y_va, X_te, y_te,
                         att_state_cols)
        res["alpha"] = a
        res["feature_set"] = feature_set_name
        results.append(res)

    return min(results, key=lambda d: d["val_rmse"])


def inspect_coefficients(X, y, alpha, feat_cols, target_cols):
    """Fit unscaled Ridge and inspect coefficients for interpretability.

    Args:
        X: Feature array (unscaled).
        y: Target array.
        alpha: Ridge regularization strength.
        feat_cols: Feature names.
        target_cols: Target names.

    Returns:
        DataFrame of coefficients (targets × features).
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    coef_df = pd.DataFrame(
        model.coef_,
        index=target_cols,
        columns=feat_cols,
    )
    print("\nUnscaled Ridge Coefficients:")
    print(coef_df.round(4).to_string())
    return coef_df
