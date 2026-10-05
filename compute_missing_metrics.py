"""
Compute missing metrics for BAB 3 tables from saved data files.
NO model retraining — uses only saved pickle files and CSV data.

Outputs:
  - ljungbox_results.csv   (Table 7)
  - full_comparison.csv    (Table 8 - complete with NRW, MASE, RMSSE, DA)
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox

# ── Paths ──
BASE = r"C:\Users\dayinta agustina\Downloads\Documents\PENELITIAN FMIPA FORECAST FINAL"
DIR_DATA = os.path.join(BASE, "data")
DIR_MODELS = os.path.join(BASE, "models")

# ── Load raw data ──
df = pd.read_csv(os.path.join(BASE, "harga_saham_batubara.csv"),
                 index_col=0, parse_dates=True)
tickers = ["ADRO", "PTBA", "INDY", "ITMG"]

# ── Load saved ARIMA artifacts ──
with open(os.path.join(DIR_DATA, "arima_residuals.pkl"), "rb") as f:
    arima_residuals = pickle.load(f)
with open(os.path.join(DIR_DATA, "arima_fitted.pkl"), "rb") as f:
    arima_fitted = pickle.load(f)
with open(os.path.join(DIR_DATA, "arima_forecast.pkl"), "rb") as f:
    arima_forecast = pickle.load(f)
with open(os.path.join(DIR_DATA, "arima_best_orders.pkl"), "rb") as f:
    arima_best_orders = pickle.load(f)

# ── Load scalers ──
with open(os.path.join(BASE, "scalers_lstm.pkl"), "rb") as f:
    scalers_lstm = pickle.load(f)
with open(os.path.join(BASE, "scalers_resid.pkl"), "rb") as f:
    scalers_resid = pickle.load(f)

# ── Constants ──
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
WINDOW = 30
N_TEST = 476
N_EVAL = 446  # N_TEST - WINDOW

print("="*70)
print("  PART 1: LJUNG-BOX RESIDUAL DIAGNOSTICS (Table 7)")
print("="*70)

ljungbox_rows = []
for ticker in tickers:
    resid = arima_residuals[ticker]
    # Drop NaN
    resid_clean = resid.dropna().values if hasattr(resid, 'dropna') else resid[~np.isnan(resid)]
    
    lags_test = [5, 10, 20]
    lb = acorr_ljungbox(resid_clean, lags=lags_test, return_df=True)
    
    for lag in lags_test:
        row = lb.loc[lag]
        ljungbox_rows.append({
            "Stock": ticker,
            "Lag": lag,
            "Q-statistic": round(row["lb_stat"], 4),
            "p-value": round(row["lb_pvalue"], 6),
        })
        print(f"  {ticker} Lag={lag}: Q={row['lb_stat']:.4f}, p={row['lb_pvalue']:.6f}")

df_lb = pd.DataFrame(ljungbox_rows)
df_lb.to_csv(os.path.join(BASE, "ljungbox_results.csv"), index=False)
print(f"\n  Saved: ljungbox_results.csv\n")

# ══════════════════════════════════════════════════════════════
print("="*70)
print("  PART 2: FULL COMPARISON TABLE (Table 8)")
print("="*70)

# ── Load LSTM models and generate predictions ──
import tensorflow as tf
tf.random.set_seed(2024)

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

comparison_rows = []

for ticker in tickers:
    series = df[ticker].dropna()
    train_data = series[series.index <= TRAIN_END]
    test_data  = series[series.index >= TEST_START]
    
    n_test = min(len(test_data), N_TEST)
    test_data = test_data.iloc[:n_test]
    aktual_full = test_data.values
    
    # ── ARIMA forecast (already saved) ──
    arima_fc = arima_forecast[ticker].values[:n_test]
    
    # ── LSTM Pure predictions ──
    scaler_pure = scalers_lstm[ticker]
    full_for_test = pd.concat([train_data, test_data])
    full_scaled = scaler_pure.transform(full_for_test.values.reshape(-1, 1)).flatten()
    n_train = len(train_data)
    
    mdl_pure = tf.keras.models.load_model(
        os.path.join(DIR_MODELS, f"lstm_pure_{ticker}.keras"))
    
    X_test_list = []
    for i in range(n_test):
        idx = n_train + i
        X_test_list.append(full_scaled[idx-WINDOW:idx])
    X_test = np.array(X_test_list).reshape(-1, WINDOW, 1)
    
    pred_pure_scaled = mdl_pure.predict(X_test, verbose=0).flatten()
    lstm_pred = scaler_pure.inverse_transform(
        pred_pure_scaled.reshape(-1, 1)).flatten()
    
    # ── Hybrid predictions ──
    resid_train = arima_residuals[ticker]
    resid_clean = resid_train.dropna().values if hasattr(resid_train, 'dropna') else resid_train[~np.isnan(resid_train)]
    
    scaler_resid = scalers_resid[ticker]
    resid_scaled = scaler_resid.transform(resid_clean.reshape(-1, 1)).flatten()
    
    mdl_resid = tf.keras.models.load_model(
        os.path.join(DIR_MODELS, f"lstm_resid_{ticker}.keras"))
    
    # Rolling residual prediction for test set
    last_window = list(resid_scaled[-WINDOW:])
    resid_preds = []
    for _ in range(n_test):
        inp = np.array(last_window[-WINDOW:]).reshape(1, WINDOW, 1)
        p = mdl_resid.predict(inp, verbose=0).flatten()[0]
        resid_preds.append(p)
        last_window.append(p)
    
    resid_pred = scaler_resid.inverse_transform(
        np.array(resid_preds).reshape(-1, 1)).flatten()
    hybrid_pred = arima_fc + resid_pred
    
    # ── Align to N_EVAL (skip first WINDOW points) ──
    actual = aktual_full[WINDOW:]
    arima_eval = arima_fc[WINDOW:]
    lstm_eval = lstm_pred[WINDOW:]
    hybrid_eval = hybrid_pred[WINDOW:]
    nrw_eval = aktual_full[WINDOW-1:-1]  # Naive: y_hat(t) = y(t-1)
    
    n = len(actual)
    
    # ── Metric functions ──
    def calc_mae(a, p): return np.mean(np.abs(a - p))
    def calc_rmse(a, p): return np.sqrt(np.mean((a - p)**2))
    def calc_mape(a, p): return np.mean(np.abs((a - p) / a)) * 100
    def calc_r2(a, p):
        ss_res = np.sum((a - p)**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        return 1 - ss_res / ss_tot
    
    # Naive MAE for scaling
    naive_mae = np.mean(np.abs(actual[1:] - actual[:-1]))
    naive_mse = np.mean((actual[1:] - actual[:-1])**2)
    
    def calc_mase(a, p):
        return calc_mae(a, p) / naive_mae
    def calc_rmsse(a, p):
        return np.sqrt(calc_rmse(a, p)**2 / naive_mse)
    def calc_da(a, p):
        actual_dir = np.sign(a[1:] - a[:-1])
        pred_dir = np.sign(p[1:] - p[:-1])
        return np.mean(actual_dir == pred_dir) * 100
    
    # ── Compute all metrics for each model ──
    models = {
        "NRW": nrw_eval,
        f"ARIMA": arima_eval,
        "LSTM": lstm_eval,
        "Hybrid": hybrid_eval,
    }
    
    order = arima_best_orders[ticker]
    
    for model_name, preds in models.items():
        mae = calc_mae(actual, preds)
        rmse = calc_rmse(actual, preds)
        mape = calc_mape(actual, preds)
        r2 = calc_r2(actual, preds)
        mase = calc_mase(actual, preds)
        rmsse = calc_rmsse(actual, preds)
        da = calc_da(actual, preds)
        
        label = model_name
        if model_name == "ARIMA":
            label = f"ARIMA({order[0]},{order[1]},{order[2]})"
        
        comparison_rows.append({
            "Stock": ticker,
            "Model": label,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 6),
            "MASE": round(mase, 4),
            "RMSSE": round(rmsse, 4),
            "DA": round(da, 2),
        })
        
        print(f"  {ticker:4s} {label:18s}  MAE={mae:10.2f}  RMSE={rmse:10.2f}  "
              f"MAPE={mape:6.2f}%  R²={r2:8.4f}  MASE={mase:.4f}  RMSSE={rmsse:.4f}  DA={da:.2f}%")
    print()

df_comp = pd.DataFrame(comparison_rows)
df_comp.to_csv(os.path.join(BASE, "full_comparison.csv"), index=False)
print(f"  Saved: full_comparison.csv\n")

print("="*70)
print("  DONE — All metrics computed from saved model files.")
print("="*70)
