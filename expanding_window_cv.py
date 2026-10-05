"""
Expanding-Window Cross-Validation (2020-2022)
=============================================
Fold 1: Train up to end-2019, Test on 2020
Fold 2: Train up to end-2020, Test on 2021
Fold 3: Train up to end-2021, Test on 2022

Models: ARIMA, LSTM (pure), Hybrid ARIMA-LSTM
Metric: MAPE per fold, then Mean MAPE across 3 folds
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Reproducibility
SEED = 2024
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters (same as main notebook) ──
WINDOW = 30
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
ES_PATIENCE = 15

# ── Load data ──
df = pd.read_csv("harga_saham_batubara.csv", index_col=0, parse_dates=True)
ticker_list = ["ADRO", "PTBA", "INDY", "ITMG"]

# ── Best ARIMA orders (from main notebook) ──
best_orders = {"ADRO": (3,1,3), "PTBA": (1,1,3), "INDY": (1,1,3), "ITMG": (3,1,3)}

# ── Folds ──
folds = [
    {"name": "Fold1", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"name": "Fold2", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "Fold3", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]

def build_lstm(window, units, dropout, lr):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def calc_mape(actual, pred):
    return np.mean(np.abs((actual - pred) / actual)) * 100

callbacks_def = [
    EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
]

# ── Main loop ──
all_results = []

for fold in folds:
    print(f"\n{'='*60}")
    print(f"  {fold['name']}: Train <= {fold['train_end']}, Test = {fold['test_start']} to {fold['test_end']}")
    print(f"{'='*60}")
    
    for ticker in ticker_list:
        series = df[ticker].dropna()
        
        train_data = series[series.index <= fold["train_end"]]
        test_data  = series[(series.index >= fold["test_start"]) & (series.index <= fold["test_end"])]
        
        if len(test_data) == 0:
            print(f"  {ticker}: No test data, skipping")
            continue
        
        n_test = len(test_data)
        aktual = test_data.values
        
        # ────────────────── ARIMA ──────────────────
        try:
            arima_model = SARIMAX(
                train_data, order=best_orders[ticker], trend="n",
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False, method="lbfgs", maxiter=300)
            
            arima_fc = arima_model.get_forecast(steps=n_test).predicted_mean.values
            mape_arima = calc_mape(aktual, arima_fc)
        except Exception as e:
            print(f"  {ticker} ARIMA error: {e}")
            mape_arima = np.nan
        
        # ────────────────── LSTM (pure) ──────────────────
        try:
            scaler_pure = MinMaxScaler()
            train_scaled = scaler_pure.fit_transform(train_data.values.reshape(-1, 1)).flatten()
            
            # Create full scaled series for test sequences
            full_for_test = pd.concat([train_data, test_data])
            full_scaled = scaler_pure.transform(full_for_test.values.reshape(-1, 1)).flatten()
            
            # Training sequences from train data
            X_train, y_train = create_sequences(train_scaled, WINDOW)
            X_train = X_train.reshape(-1, WINDOW, 1)
            
            # Test sequences: need WINDOW points before each test point
            n_train = len(train_data)
            X_test_list = []
            for i in range(n_test):
                idx = n_train + i
                X_test_list.append(full_scaled[idx-WINDOW:idx])
            X_test = np.array(X_test_list).reshape(-1, WINDOW, 1)
            
            mdl_pure = build_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
            mdl_pure.fit(X_train, y_train, validation_split=0.15, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, callbacks=callbacks_def, verbose=0)
            
            pred_scaled = mdl_pure.predict(X_test, verbose=0).flatten()
            lstm_pred = scaler_pure.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            mape_lstm = calc_mape(aktual, lstm_pred)
        except Exception as e:
            print(f"  {ticker} LSTM error: {e}")
            mape_lstm = np.nan
        
        # ────────────────── Hybrid ARIMA-LSTM ──────────────────
        try:
            # Residuals from ARIMA on training data
            arima_fitted_vals = arima_model.fittedvalues.values
            resid_train = train_data.values - arima_fitted_vals
            # Remove any NaN at the start
            resid_train = resid_train[~np.isnan(resid_train)]
            
            scaler_resid = MinMaxScaler()
            resid_scaled = scaler_resid.fit_transform(resid_train.reshape(-1, 1)).flatten()
            
            X_res_train, y_res_train = create_sequences(resid_scaled, WINDOW)
            X_res_train = X_res_train.reshape(-1, WINDOW, 1)
            
            mdl_resid = build_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
            mdl_resid.fit(X_res_train, y_res_train, validation_split=0.15, epochs=EPOCHS,
                         batch_size=BATCH_SIZE, callbacks=callbacks_def, verbose=0)
            
            # Iterative residual forecast
            last_window = resid_scaled[-WINDOW:].tolist()
            resid_preds = []
            for _ in range(n_test):
                inp = np.array(last_window[-WINDOW:]).reshape(1, WINDOW, 1)
                p = mdl_resid.predict(inp, verbose=0).flatten()[0]
                resid_preds.append(p)
                last_window.append(p)
            
            resid_pred = scaler_resid.inverse_transform(
                np.array(resid_preds).reshape(-1, 1)
            ).flatten()
            
            hybrid_pred = arima_fc + resid_pred
            mape_hybrid = calc_mape(aktual, hybrid_pred)
        except Exception as e:
            print(f"  {ticker} Hybrid error: {e}")
            mape_hybrid = np.nan
        
        print(f"  {ticker}: ARIMA={mape_arima:.2f}%  LSTM={mape_lstm:.2f}%  Hybrid={mape_hybrid:.2f}%")
        
        all_results.append({
            "Fold": fold["name"],
            "Emiten": ticker,
            "ARIMA_MAPE": mape_arima,
            "LSTM_MAPE": mape_lstm,
            "Hybrid_MAPE": mape_hybrid,
        })

# ── Aggregate: Mean MAPE across folds ──
df_results = pd.DataFrame(all_results)
print("\n\n" + "="*60)
print("  DETAIL PER FOLD")
print("="*60)
print(df_results.to_string(index=False))

# Mean across folds per emiten
df_mean = df_results.groupby("Emiten")[["ARIMA_MAPE", "LSTM_MAPE", "Hybrid_MAPE"]].mean()
df_mean = df_mean.round(2)
df_mean.columns = ["ARIMA", "LSTM", "Hybrid"]

print("\n\n" + "="*60)
print("  Table X. Mean MAPE from expanding-window validation (2020-2022)")
print("="*60)
print(df_mean.to_string())

# Save
df_mean.to_csv("expanding_window_validation.csv")
df_results.to_csv("expanding_window_detail.csv", index=False)
print("\nSaved to expanding_window_validation.csv and expanding_window_detail.csv")
