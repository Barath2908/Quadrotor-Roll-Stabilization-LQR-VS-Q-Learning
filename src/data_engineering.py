"""
Data Engineering Module — ULog Parsing, Signal Alignment, Feature Construction

Handles loading PX4 .ulg flight logs and constructing aligned DataFrames
with angular rates, accelerations, and motor commands.
"""

import os
import numpy as np
import pandas as pd
from pyulog import ULog


def ulog_topic_to_df(ulog: ULog, topic_name: str) -> pd.DataFrame:
    """Convert one ULog topic into a pandas DataFrame with time in seconds.

    Args:
        ulog: Loaded ULog object.
        topic_name: Name of the ULog topic (e.g., 'vehicle_angular_velocity').

    Returns:
        DataFrame with a 't' column (seconds from start) and all topic fields.
    """
    matching = [m for m in ulog.data_list if m.name == topic_name]
    if not matching:
        raise ValueError(f"Topic '{topic_name}' not found in log. "
                         f"Available: {[m.name for m in ulog.data_list]}")
    m = matching[0]
    df = pd.DataFrame(m.data)
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column missing in topic '{topic_name}'")
    df = df.sort_values("timestamp")
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]) * 1e-6  # µs → s
    return df


def load_and_align_flight_data(log_path: str) -> pd.DataFrame:
    """Load a PX4 .ulg file and align angular rates, accelerations, and motor commands.

    Performs nearest-timestamp merge of three asynchronous topics:
      - vehicle_angular_velocity → (p, q, r)
      - vehicle_acceleration → (ax, ay, az)
      - actuator_motors → (u1, u2, u3, u4)

    Args:
        log_path: Path to the .ulg file.

    Returns:
        Aligned DataFrame with columns [t, p, q, r, ax, ay, az, u1, u2, u3, u4].
    """
    ulog = ULog(log_path)

    rate_topic = "vehicle_angular_velocity"
    accel_topic = "vehicle_acceleration"
    motors_topic = "actuator_motors"

    df_rate = ulog_topic_to_df(ulog, rate_topic)
    df_accel = ulog_topic_to_df(ulog, accel_topic)
    df_motors = ulog_topic_to_df(ulog, motors_topic)

    # Extract and rename relevant columns
    df_rate_small = df_rate[["t", "xyz[0]", "xyz[1]", "xyz[2]"]].rename(
        columns={"xyz[0]": "p", "xyz[1]": "q", "xyz[2]": "r"}
    )
    df_accel_small = df_accel[["t", "xyz[0]", "xyz[1]", "xyz[2]"]].rename(
        columns={"xyz[0]": "ax", "xyz[1]": "ay", "xyz[2]": "az"}
    )
    df_motors_small = df_motors[
        ["t", "control[0]", "control[1]", "control[2]", "control[3]"]
    ].rename(
        columns={
            "control[0]": "u1", "control[1]": "u2",
            "control[2]": "u3", "control[3]": "u4",
        }
    )

    # Sort by time
    df_rate_small = df_rate_small.sort_values("t")
    df_accel_small = df_accel_small.sort_values("t")
    df_motors_small = df_motors_small.sort_values("t")

    # Merge with nearest-timestamp interpolation
    df = pd.merge_asof(df_rate_small, df_accel_small, on="t", direction="nearest")
    df = pd.merge_asof(df, df_motors_small, on="t", direction="nearest")
    df = df.dropna(subset=["u1", "u2", "u3", "u4"]).reset_index(drop=True)

    return df


def build_markov_dataset(df: pd.DataFrame,
                         feature_cols: list = None,
                         target_cols: list = None):
    """Build 1-step Markov prediction dataset: X[t] → y[t+1].

    Args:
        df: Aligned flight data DataFrame.
        feature_cols: Input feature columns (default: all states + motors).
        target_cols: Target columns to predict at t+1 (default: all states).

    Returns:
        Tuple of (X, y, t_for_X) where X[i] maps to y[i] = state at t+1.
    """
    if feature_cols is None:
        feature_cols = ["p", "q", "r", "ax", "ay", "az", "u1", "u2", "u3", "u4"]
    if target_cols is None:
        target_cols = ["p", "q", "r", "ax", "ay", "az"]

    data = df[["t"] + feature_cols + target_cols].dropna().reset_index(drop=True)
    t_all = data["t"].values
    X_all = data[feature_cols].values
    y_all = data[target_cols].values

    # 1-step pairs: input at t, target at t+1
    X = X_all[:-1]
    y = y_all[1:]
    t_for_X = t_all[:-1]

    return X, y, t_for_X


def time_ordered_split(X, y, t=None, train_frac=0.70, val_frac=0.15):
    """Split data into train/val/test preserving temporal order (no future leakage).

    Args:
        X: Feature array (N, D_in).
        y: Target array (N, D_out).
        t: Optional time array.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.

    Returns:
        Dict with keys 'train', 'val', 'test', each containing (X, y, t) tuples.
    """
    N = len(X)
    n_train = int(train_frac * N)
    n_val = int(val_frac * N)

    splits = {
        "train": (X[:n_train], y[:n_train]),
        "val": (X[n_train:n_train + n_val], y[n_train:n_train + n_val]),
        "test": (X[n_train + n_val:], y[n_train + n_val:]),
    }

    if t is not None:
        splits["train"] = (*splits["train"], t[:n_train])
        splits["val"] = (*splits["val"], t[n_train:n_train + n_val])
        splits["test"] = (*splits["test"], t[n_train + n_val:])

    splits["n_train"] = n_train
    splits["n_val"] = n_val
    splits["n_test"] = N - n_train - n_val

    return splits


def make_sequence_dataset(df: pd.DataFrame,
                          feature_cols: list,
                          target_cols: list,
                          window: int = 10):
    """Build windowed sequence dataset for LSTM/Transformer models.

    Args:
        df: Aligned flight data DataFrame.
        feature_cols: Input feature columns.
        target_cols: Target columns to predict.
        window: Number of past timesteps per sample.

    Returns:
        Tuple of (X_seq, y_seq) with shapes (N-window, window, D_in) and (N-window, D_out).
    """
    data = df[feature_cols + target_cols].dropna().values
    D_in = len(feature_cols)

    X_list, y_list = [], []
    for i in range(window, len(data)):
        X_list.append(data[i - window:i, :D_in])
        y_list.append(data[i, D_in:])

    return np.array(X_list), np.array(y_list)
