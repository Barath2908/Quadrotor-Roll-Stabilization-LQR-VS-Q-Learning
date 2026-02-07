"""
Roll-Axis Linear Model Extraction

Identifies the hover regime, fits a 1D ARX model for roll dynamics,
and extracts discrete-time (A, B, b) matrices for control design.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def identify_hover_cluster(df, cluster_col="cluster"):
    """Identify the hover cluster as the one with smallest RMS angular rates.

    Args:
        df: DataFrame with cluster labels and angular rate columns (p, q, r).

    Returns:
        Hover cluster ID (int).
    """
    cluster_ids = sorted(df[cluster_col].unique())

    cluster_rms = {}
    for c in cluster_ids:
        sub = df[df[cluster_col] == c]
        rms = np.sqrt((sub[["p", "q", "r"]] ** 2).mean().mean())
        cluster_rms[c] = rms
        print(f"  Cluster {c}: RMS angular rate = {rms:.4f}")

    hover_cluster = min(cluster_rms, key=cluster_rms.get)
    print(f"\n→ Hover cluster: {hover_cluster} (lowest angular activity)")
    return hover_cluster


def fit_roll_arx_model(X_phys_hover, y_att_hover, feat_phys_cols, att_state_cols):
    """Fit a 1D roll-rate ARX model on hover-regime data.

    Model: p_{t+1} = a_p * p_t + b_p * u_roll_t + c_p

    Args:
        X_phys_hover: Physics features for hover subset (N, D).
        y_att_hover: Attitude targets for hover subset (N, 3).
        feat_phys_cols: Physics feature column names.
        att_state_cols: Attitude target column names.

    Returns:
        Tuple of (a_p, b_p, c_p, roll_regression_model, rmse).
    """
    i_p = feat_phys_cols.index("p")
    i_u_roll = feat_phys_cols.index("u_roll")
    i_p_next = att_state_cols.index("p")

    p_t = X_phys_hover[:, i_p]
    u_roll_t = X_phys_hover[:, i_u_roll]
    p_next = y_att_hover[:, i_p_next]

    # Features: [p_t, u_roll_t]
    X_roll = np.column_stack([p_t, u_roll_t])

    reg = LinearRegression()
    reg.fit(X_roll, p_next)

    a_p = reg.coef_[0]
    b_p = reg.coef_[1]
    c_p = reg.intercept_

    print(f"\nRoll ARX model (hover regime):")
    print(f"  p_{{t+1}} = {a_p:.6f} * p_t + {b_p:.6f} * u_roll_t + {c_p:.6f}")

    # Evaluate
    p_pred = reg.predict(X_roll)
    rmse = np.sqrt(mean_squared_error(p_next, p_pred))
    print(f"  Hover-subset RMSE: {rmse:.6f}")

    return a_p, b_p, c_p, reg, rmse


def extract_state_space_matrices(a_p, b_p, c_p, dt_approx=None):
    """Extract discrete-time state-space matrices (A, B, b) for MATLAB/control design.

    System: p_{k+1} = A * p_k + B * u_roll_k + b

    Args:
        a_p: Autoregressive coefficient.
        b_p: Input coefficient.
        c_p: Bias term.
        dt_approx: Approximate sampling period (if known).

    Returns:
        Dict with A_roll, B_roll, b_roll matrices and metadata.
    """
    A_roll = np.array([[a_p]])
    B_roll = np.array([[b_p]])
    b_roll = np.array([[c_p]])

    print(f"\nDiscrete-time state-space matrices:")
    print(f"  A_roll = {A_roll}")
    print(f"  B_roll = {B_roll}")
    print(f"  b_roll = {b_roll}")

    result = {
        "A_roll": A_roll,
        "B_roll": B_roll,
        "b_roll": b_roll,
        "a_p": a_p,
        "b_p": b_p,
        "c_p": c_p,
    }

    if dt_approx is not None:
        print(f"  Approximate Δt = {dt_approx:.6f} s")
        result["dt"] = dt_approx

    return result


def simulate_roll_model(p0, u_seq, A, B, b):
    """Simulate the discrete roll model for a given control sequence.

    Args:
        p0: Initial roll rate.
        u_seq: Array of u_roll commands (length T).
        A: State matrix (scalar or 1×1).
        B: Input matrix (scalar or 1×1).
        b: Bias (scalar or 1×1).

    Returns:
        Array of roll rate trajectory (length T+1, including p0).
    """
    a = float(A) if np.ndim(A) > 0 else A
    bcoef = float(B) if np.ndim(B) > 0 else B
    bias = float(b) if np.ndim(b) > 0 else b

    T = len(u_seq)
    p_traj = np.zeros(T + 1)
    p_traj[0] = p0

    for k in range(T):
        p_traj[k + 1] = a * p_traj[k] + bcoef * u_seq[k] + bias

    return p_traj
