"""
Sequence Models — LSTM & Transformer for Quadrotor Dynamics

Keras/TensorFlow implementations for predicting next-state dynamics
from windowed time-series input.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def build_lstm_model(window: int, d_in: int, d_out: int,
                     lstm_units: int = 64, n_layers: int = 2,
                     dropout: float = 0.0) -> keras.Model:
    """Build a stacked LSTM model for next-state prediction.

    Args:
        window: Input sequence length.
        d_in: Number of input features per timestep.
        d_out: Number of output targets.
        lstm_units: Hidden units per LSTM layer.
        n_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between layers.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(window, d_in)))

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        model.add(keras.layers.LSTM(lstm_units, return_sequences=return_seq))
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(d_out))

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(optimizer="adam", loss="mse", metrics=[rmse])
    return model


def build_transformer_model(window: int, d_in: int, d_out: int,
                            d_model: int = 64, num_heads: int = 2,
                            ff_dim: int = 128, dropout: float = 0.1) -> keras.Model:
    """Build a single-layer Transformer encoder for next-state prediction.

    Args:
        window: Input sequence length.
        d_in: Number of input features per timestep.
        d_out: Number of output targets.
        d_model: Internal embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward layer dimension.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model.
    """
    inputs = keras.Input(shape=(window, d_in))

    # Linear projection to model dimension
    x = keras.layers.Dense(d_model)(inputs)

    # Multi-head self-attention
    attn_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads
    )(x, x)
    attn_output = keras.layers.Dropout(dropout)(attn_output)
    x = keras.layers.LayerNormalization()(x + attn_output)

    # Feed-forward block
    ff_output = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff_output = keras.layers.Dense(d_model)(ff_output)
    ff_output = keras.layers.Dropout(dropout)(ff_output)
    x = keras.layers.LayerNormalization()(x + ff_output)

    # Pool over time and predict
    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(d_out)(x)

    model = keras.Model(inputs, outputs)

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(optimizer="adam", loss="mse", metrics=[rmse])
    return model


def train_sequence_model(model, X_train, y_train, X_val, y_val,
                         epochs: int = 200, batch_size: int = 64,
                         patience: int = 10):
    """Train a sequence model with early stopping.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training sequences.
        X_val, y_val: Validation sequences.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        patience: Early stopping patience.

    Returns:
        Training history object.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_rmse",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def per_channel_rmse(y_true, y_pred, channel_names):
    """Compute and print per-channel RMSE.

    Args:
        y_true: Ground truth array (N, D_out).
        y_pred: Predictions array (N, D_out).
        channel_names: List of channel name strings.

    Returns:
        Dict mapping channel name to RMSE value.
    """
    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    results = {}
    for name, r in zip(channel_names, rmse):
        print(f"  {name}: RMSE = {r:.4f}")
        results[name] = r
    return results
