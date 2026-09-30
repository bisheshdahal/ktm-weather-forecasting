# Kathmandu Temperature Forecasting

Comparing three forecasting approaches — naive baseline, XGBoost, and an LSTM — on hourly Kathmandu weather data, rather than just training one model and calling it done.

## Goal

Predict `temperature_2m` one hour ahead, using the previous 24 hours of data across 7 weather variables:

- temperature_2m
- relative_humidity_2m
- apparent_temperature
- wind_speed_10m
- wind_speed_100m
- soil_moisture_0_to_7cm
- soil_moisture_7_to_28cm

## Approach

Three models were trained on the same chronological 80/20 train/test split and evaluated on identical metrics:

| Model | Description |
|---|---|
| **Naive baseline** | Predicts "next hour = this hour" — the bar every real model has to clear |
| **XGBoost** | Gradient-boosted trees on engineered features: lag values (1, 2, 3, 24 hours), rolling mean/std, and cyclical hour/day-of-year encodings |
| **LSTM** | Reads the raw 24-hour sequence directly, learning temporal patterns without hand-crafted features |

## Results

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Naive | 0.871 | 1.233 | 0.956 |
| XGBoost | 0.416 | 0.560 | 0.991 |
| **LSTM** | **0.396** | **0.537** | **0.991** |

Both XGBoost and LSTM dramatically outperform the naive baseline, confirming real short-term structure in the data beyond simple persistence. LSTM edges out XGBoost on MAE and RMSE, though the margin is small given both explain ~99% of variance — in a production setting, XGBoost's much lower training cost (no GPU, seconds not minutes) could make it the better trade-off despite the slightly higher error.

### What actually drives the prediction

![Feature importance](results/feature_importance.png)

`Lag_24` (temperature exactly 24 hours prior) is by far the strongest feature — more important than `Lag_1` (the previous hour), despite being temporally further away. This makes physical sense: temperature follows a strong daily cycle, so "what was it at this hour yesterday" carries more signal than "what was it an hour ago," which is more sensitive to short-term noise like a passing cloud or a wind gust. Rolling statistics, wind, soil moisture, and seasonal encodings contributed comparatively little — the daily cycle alone explains most of the predictable variance in this dataset.

## Repo Structure

```
├── ktmaqi.csv              # raw hourly weather data
├── ktmaqi.ipynb             # full analysis: EDA, all 3 models, evaluation
├── data_prep.py              # sliding-window sequence creation for the LSTM
├── model.py                   # LSTM model class (general-purpose regression)
├── engine.py                   # train / validate loop (model-agnostic)
├── inverse_transform.py         # undo scaling to get predictions back in °C
└── README.md
```

## Usage

```python
# 1. Load and clean data — see ktmaqi.ipynb Section 1

# 2. Build LSTM sequences
from data_prep import create_sequences
X_train, y_train = create_sequences(train_scaled, target_col_index=0, seq_length=24)

# 3. Train
from model import LSTM
from engine import fit

model = LSTM(input_size=7, hidden_size=128, num_stacked_layers=4).to(device)
train_losses, val_losses = fit(
    model, train_loader, test_loader,
    optimizer, loss_function, device,
    epochs=25
)

# 4. Convert predictions back to real-world units
from inverse_transform import inverse_target
y_pred_celsius = inverse_target(y_pred, scaler, n_features=7, target_idx=0)
```

Full walkthrough — EDA, feature engineering, all three models, and the comparison above — is in `ktmaqi.ipynb`.

## Status

✅ Complete — naive baseline, XGBoost, and LSTM all trained, evaluated, and compared on the same test set.

**Possible next steps:** multi-step forecasting (predict several hours ahead instead of one), walk-forward validation instead of a single train/test split, or dropping the near-zero-importance features to see if performance holds with a simpler model.