"""
Flight Regime Analysis — Clustering, Per-Regime Evaluation, Bootstrap Uncertainty

Uses K-Means clustering to identify hover vs. aggressive flight regimes,
evaluates model performance per regime, and quantifies prediction uncertainty
via bootstrap ensembles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


def cluster_flight_regimes(df: pd.DataFrame, K: int = 3,
                           cluster_features: list = None,
                           random_state: int = 42):
    """Cluster flight data into K regimes using K-Means.

    Args:
        df: Flight DataFrame with state and motor columns.
        K: Number of clusters.
        cluster_features: Columns to use for clustering.
        random_state: Random seed.

    Returns:
        Modified DataFrame with 'cluster' column added, and the fitted KMeans object.
    """
    if cluster_features is None:
        cluster_features = ["p", "q", "r", "ax", "ay", "az", "u_thrust"]

    df = df.copy()
    if "u_thrust" not in df.columns:
        df["u_thrust"] = df["u1"] + df["u2"] + df["u3"] + df["u4"]

    Z = df[cluster_features].to_numpy()
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)

    kmeans = KMeans(n_clusters=K, n_init=20, random_state=random_state)
    df["cluster"] = kmeans.fit_predict(Z_scaled)

    # Print cluster summary
    summary = df.groupby("cluster")[cluster_features].agg(["mean", "std"])
    print(f"Cluster summary (K={K}):")
    print(summary.round(4))

    return df, kmeans


def per_cluster_rmse(name, model, X_train, y_train, X_test, y_test,
                     cluster_test, target_cols):
    """Evaluate a model's RMSE broken down by flight regime cluster.

    Args:
        name: Model name for display.
        model: sklearn-compatible estimator (will be fitted).
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        cluster_test: Cluster labels for test data.
        target_cols: Target column names.

    Returns:
        Dict mapping cluster_id to per-channel RMSE dict.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {name}: Per-Cluster Test RMSE ===")
    results = {}
    for c in np.unique(cluster_test):
        mask = cluster_test == c
        n_c = mask.sum()
        mse_ch = ((y_test[mask] - y_pred[mask]) ** 2).mean(axis=0)
        rmse_ch = np.sqrt(mse_ch)
        overall = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))

        print(f"  Cluster {c} (n={n_c}): overall RMSE = {overall:.4f}")
        cluster_results = {"overall": overall, "n": n_c}
        for col, r in zip(target_cols, rmse_ch):
            print(f"    {col}: {r:.4f}")
            cluster_results[col] = r
        results[c] = cluster_results

    return results


def bootstrap_ridge_uncertainty(X_train, y_train, X_test,
                                alpha: float = 0.01, B: int = 50,
                                random_state: int = 0):
    """Estimate predictive uncertainty via bootstrap ensemble of Ridge models.

    Args:
        X_train, y_train: Training data.
        X_test: Test features.
        alpha: Ridge alpha.
        B: Number of bootstrap models.
        random_state: Base random seed.

    Returns:
        Tuple of (mean_preds, std_preds, all_boot_preds) each with shape (N_test, D_out).
    """
    rng = np.random.RandomState(random_state)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_targets = y_train.shape[1]

    boot_preds = np.zeros((B, n_test, n_targets))

    for b in range(B):
        idx = rng.choice(n_train, size=n_train, replace=True)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train[idx], y_train[idx])
        boot_preds[b] = model.predict(X_test)

    mean_preds = boot_preds.mean(axis=0)
    std_preds = boot_preds.std(axis=0)

    return mean_preds, std_preds, boot_preds


def plot_uncertainty_band(t_test, y_true, mean_pred, std_pred,
                          axis_idx: int = 0, axis_name: str = "p",
                          start: int = 0, length: int = 500,
                          sigma_mult: float = 2.0):
    """Plot prediction with ±Nσ uncertainty band for one channel.

    Args:
        t_test: Time array for test segment.
        y_true: Ground truth (N_test, D_out).
        mean_pred: Bootstrap mean prediction (N_test, D_out).
        std_pred: Bootstrap std prediction (N_test, D_out).
        axis_idx: Channel index to plot.
        axis_name: Channel name.
        start: Starting index within test set.
        length: Number of points to plot.
        sigma_mult: Number of standard deviations for band width.
    """
    end = start + length
    sl = slice(start, end)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_test[sl], y_true[sl, axis_idx], "k-", lw=1, label="Ground truth")
    ax.plot(t_test[sl], mean_pred[sl, axis_idx], "b-", lw=1, label="Bootstrap mean")
    ax.fill_between(
        t_test[sl],
        mean_pred[sl, axis_idx] - sigma_mult * std_pred[sl, axis_idx],
        mean_pred[sl, axis_idx] + sigma_mult * std_pred[sl, axis_idx],
        alpha=0.3, color="blue",
        label=f"±{sigma_mult}σ band",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{axis_name} [rad/s]")
    ax.set_title(f"Bootstrap uncertainty for {axis_name}(t)")
    ax.legend()
    plt.tight_layout()
    return fig


def mean_uncertainty_per_cluster(std_pred, cluster_test, axis_idx=0, axis_name="p"):
    """Compute mean predictive std per flight regime cluster.

    Args:
        std_pred: Bootstrap std predictions (N_test, D_out).
        cluster_test: Cluster labels for test data.
        axis_idx: Channel index.
        axis_name: Channel name for display.

    Returns:
        Dict mapping cluster_id to mean std.
    """
    print(f"\nMean predictive std for {axis_name} per cluster (test set):")
    results = {}
    for c in np.unique(cluster_test):
        mask = cluster_test == c
        if mask.sum() > 0:
            mean_std = std_pred[mask, axis_idx].mean()
            print(f"  Cluster {c}: mean σ = {mean_std:.6f} (n={mask.sum()})")
            results[c] = mean_std
    return results
