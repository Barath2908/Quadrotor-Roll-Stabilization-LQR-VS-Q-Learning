# Data-Driven System Identification for Quadrotor Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning pipeline for identifying quadrotor dynamics from PX4 flight log data. This project compares classical ML regressors, sequence models (LSTM & Transformer), and physics-guided feature engineering to predict next-state dynamics of a quadrotor UAV.

## Overview

This project addresses the problem of learning discrete-time dynamics models for quadrotor UAVs directly from flight data. Given state measurements (angular rates, accelerations) and motor commands at time $t$, we predict the state at $t+1$.

**Key Research Questions:**
1. Which ML model best captures quadrotor dynamics from flight data?
2. Do sequence models (LSTM, Transformer) outperform 1-step regressors?
3. Does physics-guided feature engineering improve prediction accuracy?
4. How does prediction quality vary across flight regimes (hover vs. aggressive)?
5. Can we extract interpretable linear models for attitude control design?

## Methods

### Data Pipeline
- **Source:** PX4 `.ulg` flight logs from the [ARP Lab dataset](https://github.com/arplaboratory/data-driven-system-identification)
- **Signals:** Angular rates (`p, q, r`), linear accelerations (`ax, ay, az`), motor commands (`u1–u4`)
- **Alignment:** Asynchronous topics merged via nearest-timestamp interpolation
- **Split:** 70/15/15 time-ordered train/val/test (no future leakage)

### Model Zoo (Section 2)
| Model | Description |
|-------|-------------|
| **Ridge Regression** | Linear baseline with L2 regularization (α grid search) |
| **Random Forest** | Ensemble of decision trees (hyperparameter sweep) |
| **Gradient Boosting** | Boosted trees via `MultiOutputRegressor` wrapper |
| **MLP Regressor** | Shallow neural network (sklearn) |

### Sequence Models (Section 3)
| Model | Architecture |
|-------|-------------|
| **LSTM** | 2-layer LSTM (64 units) with sliding window input |
| **Transformer** | Single-layer encoder with multi-head attention (2 heads, d=64) |

### Physics-Guided Features (Section 4)
Instead of raw motor commands (`u1–u4`), we construct torque-aligned features:
- `u_thrust = u1 + u2 + u3 + u4`
- `u_roll = u2 - u4`
- `u_pitch = u3 - u1`
- `u_yaw = u1 - u2 + u3 - u4`

### Flight Regime Analysis (Section 5)
- **K-Means clustering** on state+thrust features to identify hover vs. aggressive maneuver regimes
- **Per-cluster RMSE** evaluation for Ridge and Gradient Boosting
- **Bootstrap ensemble** (50 Ridge models) for predictive uncertainty quantification

### Roll-Axis Linear Model Extraction (Section 6)
- Isolate hover-regime data to fit a 1D discrete-time ARX model for roll dynamics
- Extract `(A, B, b)` matrices suitable for classical control design (e.g., LQR in MATLAB)

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── notebooks/
│   └── quadrotor_sysid_full_pipeline.ipynb   # Full Colab notebook (run end-to-end)
├── src/
│   ├── data_engineering.py      # ULog parsing, signal alignment, feature construction
│   ├── models.py                # Model definitions and evaluation helpers
│   ├── sequence_models.py       # LSTM & Transformer architectures (TensorFlow/Keras)
│   ├── physics_features.py      # Physics-guided feature engineering
│   ├── flight_regimes.py        # Clustering, per-regime evaluation, bootstrap uncertainty
│   └── roll_model.py            # Hover-regime roll-axis linear model extraction
├── figures/                     # Generated plots
└── results/                     # Saved metrics and model outputs
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Colab)
The easiest way to reproduce all results is via the notebook:
```
notebooks/quadrotor_sysid_full_pipeline.ipynb
```
Open in Google Colab — the notebook clones the flight data repo automatically.

### 3. Run Modular Scripts
```python
from src.data_engineering import load_and_align_flight_data
from src.models import eval_model

# Load data from ULog
df = load_and_align_flight_data("path/to/flight.ulg")

# Build features and train
X_train, y_train, X_test, y_test = ...  # see data_engineering.py
```

## Results Summary

### Model Comparison (1-Step Prediction)
| Model | Val RMSE | Test RMSE |
|-------|----------|-----------|
| Ridge (α=0.01) | **Best** | **Best** |
| Random Forest | Good | Good |
| Gradient Boosting | Good | Good |
| MLP | Moderate | Moderate |
| LSTM (window=10) | Moderate | Moderate |
| Transformer | Moderate | Moderate |

> Ridge regression with standardized features is highly competitive for 1-step prediction on smooth flight data, often matching or outperforming deep models.

### Physics-Guided Features
Physics-guided torque features consistently improve attitude prediction RMSE compared to raw motor commands, especially for roll and pitch channels.

### Uncertainty by Flight Regime
Bootstrap uncertainty is significantly higher during aggressive maneuvers vs. hover, confirming that model confidence is regime-dependent.

## Data Source

Flight data is sourced from the [ARP Laboratory](https://github.com/arplaboratory/data-driven-system-identification) at NYU. The dataset contains PX4 `.ulg` logs from real quadrotor flights with diverse maneuver profiles.

## Requirements

- Python ≥ 3.8
- NumPy, Pandas, Matplotlib
- scikit-learn
- TensorFlow ≥ 2.10 (for LSTM & Transformer)
- pyulog (for PX4 log parsing)

## Citation

If you use this work, please cite:
```bibtex
@misc{quadrotor-sysid,
  title={Data-Driven System Identification for Quadrotor Dynamics},
  author={Barath},
  year={2025},
  howpublished={\url{https://github.com/username/quadrotor-sysid}}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- **MECE 4520** — Columbia University, Mechanical Engineering
- [ARP Laboratory, NYU](https://github.com/arplaboratory) — Flight data
- PX4 Autopilot community
