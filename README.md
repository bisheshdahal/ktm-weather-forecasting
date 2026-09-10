# Kathmandu Time-Series Forecasting

Time-series forecasting on Kathmandu weather data (KTM AQI dataset, Kaggle) — comparing forecasting approaches rather than just training one model.

## Goal

Predict `temperature_2m` one hour ahead, using the previous 24 hours across 7 features:

- temperature_2m
- relative_humidity_2m
- apparent_temperature
- wind_speed_10m
- wind_speed_100m
- soil_moisture_0_to_7cm
- soil_moisture_7_to_28cm

The project is being built in stages, expanding in scope and complexity as it progresses (more targets / longer forecast horizons planned).

## Project Stages

1. **Understand the data** — EDA, temporal feature engineering, chronological split, scaling
2. **Basic LSTM** — first working baseline model
3. **Improve** — feature engineering, regularization, hyperparameter tuning
4. **Advanced** — multi-step forecasting, Attention-LSTM, walk-forward validation, error analysis
5. **Model comparison** — Naive vs XGBoost vs LSTM vs Attention-LSTM
6. **Final presentation** — README write-up, plots, comparison tables

## Repo Structure

```
├── data/
│   └── ktmaqi.csv
├── src/
│   ├── data_prep.py        # sequence creation, scaling, splitting
│   ├── models/
│   │   └── lstm.py          # LSTM model class
│   └── engine.py             # train / validate loops (model-agnostic)
├── notebooks/
│   └── 01_eda.ipynb
├── results/
│   └── (saved metrics / plots)
└── README.md
```

## Usage

```python
from src.data_prep import create_sequences
from src.models.lstm import LSTM
from src.engine import fit

X_train, y_train = create_sequences(train_scaled, target_col_index=0, seq_length=24)

model = LSTM(input_size=7, hidden_size=64, num_stacked_layers=2).to(device)

train_losses, val_losses = fit(
    model, train_loader, test_loader,
    optimizer, loss_function, device,
    epochs=20
)
```

## Status

🚧 In progress — currently on Stage 2 (basic LSTM).
