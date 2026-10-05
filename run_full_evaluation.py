"""
=======================================================================
run_full_evaluation.py  --  Complete Reviewer Evaluation Pipeline (FIXED)
=======================================================================
Produces ALL outputs required by reviewer:

  1. ljungbox_results.csv           -- Ljung-Box residual diagnostics
  2. full_comparison.csv            -- NRW + ARIMA + LSTM + Hybrid metrics
                                      (MAE, RMSE, MAPE, R2, MASE, RMSSE, DA)
  3. dm_test_results.csv            -- Diebold-Mariano test (all model pairs)
  4. expanding_window_full.csv      -- Expanding-window per-fold detail
                                      (MAE, RMSE, MAPE, R2, MASE, RMSSE, DA)
  5. expanding_window_summary.csv   -- Mean metrics across folds
  6. expanding_window_orders.csv    -- ARIMA orders re-selected per fold
  7. raw_predictions.csv            -- Actual vs Predicted for all models

KEY FIX: Uses 1-step-ahead rolling ARIMA forecast (not multi-step).
=======================================================================
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as LSTM_Layer, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# -- Reproducibility --
SEED = 2024
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -- Paths --
BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(BASE, "data")
DIR_MODELS = os.path.join(BASE, "models")

# -- Constants --
TICKERS = ["ADRO", "PTBA", "INDY", "ITMG"]
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
WINDOW = 30  # LSTM lookback

# LSTM hyperparameters (same as main notebook)
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
ES_PATIENCE = 15

# -- Load main data --
print("Loading data...")
df = pd.read_csv(os.path.join(BASE, "harga_saham_batubara.csv"),
                 index_col=0, parse_dates=True)
print(f"  Data shape: {df.shape}")
print(f"  Date range: {df.index.min()} to {df.index.max()}")

# -- Load saved ARIMA artifacts --
print("Loading ARIMA artifacts...")
with open(os.path.join(DIR_DATA, "arima_residuals.pkl"), "rb") as f:
    arima_residuals = pickle.load(f)
with open(os.path.join(DIR_DATA, "arima_best_orders.pkl"), "rb") as f:
    arima_best_orders_pkl = pickle.load(f)

# -- Load scalers --
with open(os.path.join(BASE, "scalers_lstm.pkl"), "rb") as f:
    scalers_lstm = pickle.load(f)
with open(os.path.join(BASE, "scalers_resid.pkl"), "rb") as f:
    scalers_resid = pickle.load(f)


# ====================================================================
#  PHASE 0: VERIFY ARIMA ORDERS VIA GRID SEARCH
# ====================================================================
print("\n" + "="*70)
print("  PHASE 0: ARIMA ORDER VERIFICATION (Grid Search)")
print("="*70)

def arima_grid_search(train_data, p_range=range(5), q_range=range(5), d=1):
    """Grid search for best ARIMA(p,d,q) by AIC."""
    train_vals = train_data.values if hasattr(train_data, 'values') else np.array(train_data)
    best_aic = np.inf
    best_order = (0, d, 0)
    results = []
    for p in p_range:
        for q in q_range:
            try:
                model = SARIMAX(train_vals, order=(p, d, q), trend="n",
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                result = model.fit(disp=False, method="lbfgs", maxiter=300)
                results.append(((p, d, q), result.aic))
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue
    return best_order, best_aic, results

# Verify orders for each ticker
verified_orders = {}
for ticker in TICKERS:
    series = df[ticker].dropna()
    train_data = series[series.index <= TRAIN_END]
    print(f"\n  {ticker}: Grid search on {len(train_data)} training points...", end="", flush=True)
    t0 = time.time()
    best_order, best_aic, all_results = arima_grid_search(train_data)
    elapsed = time.time() - t0
    verified_orders[ticker] = best_order
    print(f" DONE ({elapsed:.1f}s)")
    print(f"    Best order: {best_order}, AIC: {best_aic:.2f}")
    print(f"    PKL order:  {arima_best_orders_pkl[ticker]}")
    # Show top 3 by AIC
    all_results.sort(key=lambda x: x[1])
    for rank, (order, aic) in enumerate(all_results[:3], 1):
        print(f"    Top {rank}: {order} AIC={aic:.2f}")

# Use the verified orders from grid search
ARIMA_ORDERS = verified_orders
print(f"\n  FINAL ARIMA ORDERS: {ARIMA_ORDERS}")


# ====================================================================
#  HELPER FUNCTIONS
# ====================================================================
def calc_mae(a, p):
    return np.mean(np.abs(a - p))

def calc_rmse(a, p):
    return np.sqrt(np.mean((a - p)**2))

def calc_mape(a, p):
    mask = a != 0
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100

def calc_r2(a, p):
    ss_res = np.sum((a - p)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def calc_mase(a, p, naive_mae):
    """MASE = MAE(model) / MAE(naive random walk on test set)."""
    if naive_mae == 0:
        return np.nan
    return calc_mae(a, p) / naive_mae

def calc_rmsse(a, p, naive_mse):
    """RMSSE = sqrt(MSE(model) / MSE(naive random walk on test set))."""
    if naive_mse == 0:
        return np.nan
    mse_model = np.mean((a - p)**2)
    return np.sqrt(mse_model / naive_mse)

def calc_da(a, p):
    """Directional Accuracy: % of times predicted direction matches actual."""
    if len(a) < 2:
        return np.nan
    actual_dir = np.sign(a[1:] - a[:-1])
    pred_dir = np.sign(p[1:] - p[:-1])
    return np.mean(actual_dir == pred_dir) * 100

def compute_all_metrics(actual, pred, naive_mae, naive_mse):
    """Compute all 7 metrics in a dict."""
    return {
        "MAE":  round(calc_mae(actual, pred), 4),
        "RMSE": round(calc_rmse(actual, pred), 4),
        "MAPE": round(calc_mape(actual, pred), 4),
        "R2":   round(calc_r2(actual, pred), 6),
        "MASE": round(calc_mase(actual, pred, naive_mae), 4),
        "RMSSE":round(calc_rmsse(actual, pred, naive_mse), 4),
        "DA":   round(calc_da(actual, pred), 4),
    }

def diebold_mariano_test(actual, pred1, pred2, h=1, power=2):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Both models have the same predictive accuracy.
    H1: Model 1 and Model 2 have different predictive accuracy.
    """
    e1 = actual - pred1
    e2 = actual - pred2
    d = np.abs(e1)**power - np.abs(e2)**power
    n = len(d)

    d_mean = np.mean(d)

    # Autocovariance-weighted variance (Newey-West HAC)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1] if len(d) > k else 0
        gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_lstm(window, units, dropout, lr):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM_Layer(units, return_sequences=True),
        Dropout(dropout),
        LSTM_Layer(units),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


def arima_rolling_forecast(train_data, test_data, order):
    """
    1-step-ahead rolling ARIMA forecast.
    
    Fits ARIMA on training data, then for each test point:
    - Forecast 1 step ahead
    - Append actual test value to the model (without refitting)
    - Repeat
    
    Uses plain numpy arrays to avoid DatetimeIndex frequency issues.
    """
    # Use plain arrays to avoid statsmodels index issues
    train_vals = train_data.values if hasattr(train_data, 'values') else np.array(train_data)
    test_vals = test_data.values if hasattr(test_data, 'values') else np.array(test_data)
    
    model = SARIMAX(train_vals, order=order, trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, method="lbfgs", maxiter=300)
    
    predictions = []
    residuals_train = fitted.resid
    
    n_test = len(test_vals)
    current_model = fitted
    
    for i in range(n_test):
        # Forecast 1 step ahead
        fc = current_model.forecast(steps=1)[0]
        predictions.append(fc)
        
        # Append actual observation (no refit for speed)
        current_model = current_model.append([test_vals[i]], refit=False)
        
        if (i + 1) % 50 == 0:
            print(f".", end="", flush=True)
    
    return np.array(predictions), fitted, residuals_train


# ====================================================================
#  PART 1: LJUNG-BOX RESIDUAL DIAGNOSTICS
# ====================================================================
print("\n" + "="*70)
print("  PART 1: LJUNG-BOX RESIDUAL DIAGNOSTICS")
print("="*70)

ljungbox_rows = []
for ticker in TICKERS:
    resid = arima_residuals[ticker]
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
            "Significant_5pct": "Yes" if row["lb_pvalue"] < 0.05 else "No",
        })
        status = "SIGNIFICANT" if row["lb_pvalue"] < 0.05 else "not significant"
        print(f"  {ticker} Lag={lag}: Q={row['lb_stat']:.4f}, p={row['lb_pvalue']:.6f} ({status})")

df_lb = pd.DataFrame(ljungbox_rows)
df_lb.to_csv(os.path.join(BASE, "ljungbox_results.csv"), index=False)
print(f"\n  [OK] Saved: ljungbox_results.csv")


# ====================================================================
#  PART 2: FULL COMPARISON TABLE (NRW + ARIMA + LSTM + Hybrid)
#  Using 1-step rolling ARIMA + saved LSTM models
# ====================================================================
print("\n" + "="*70)
print("  PART 2: FULL COMPARISON TABLE (1-step rolling ARIMA)")
print("="*70)

comparison_rows = []
raw_pred_rows = []
arima_preds_cache = {}   # Cache ARIMA rolling predictions for DM test

for ticker in TICKERS:
    series = df[ticker].dropna()
    train_data = series[series.index <= TRAIN_END]
    test_data  = series[series.index >= TEST_START]

    n_test = len(test_data)
    aktual_full = test_data.values
    order = ARIMA_ORDERS[ticker]

    # -- ARIMA: 1-step-ahead rolling forecast --
    print(f"\n  {ticker}: ARIMA{order} rolling forecast ({n_test} steps)", end="", flush=True)
    t0 = time.time()
    arima_fc, arima_fitted_model, arima_resid = arima_rolling_forecast(
        train_data, test_data, order
    )
    print(f" ({time.time()-t0:.1f}s)")
    arima_preds_cache[ticker] = arima_fc

    # -- LSTM Pure predictions (load saved model) --
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

    # -- Hybrid predictions: ARIMA rolling + LSTM residual --
    resid_train = arima_residuals[ticker]
    resid_clean = resid_train.dropna().values if hasattr(resid_train, 'dropna') else resid_train[~np.isnan(resid_train)]

    scaler_resid = scalers_resid[ticker]
    resid_scaled = scaler_resid.transform(resid_clean.reshape(-1, 1)).flatten()

    mdl_resid = tf.keras.models.load_model(
        os.path.join(DIR_MODELS, f"lstm_resid_{ticker}.keras"))

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

    # -- Naive Random Walk: y_hat(t) = y(t-1) --
    nrw_pred = np.empty(n_test)
    nrw_pred[0] = train_data.values[-1]
    nrw_pred[1:] = aktual_full[:-1]

    # -- Compute naive MAE/MSE on test set for MASE/RMSSE scaling --
    naive_errors = aktual_full[1:] - aktual_full[:-1]
    naive_mae = np.mean(np.abs(naive_errors))
    naive_mse = np.mean(naive_errors**2)

    actual = aktual_full

    models = {
        "NRW": nrw_pred,
        f"ARIMA({order[0]},{order[1]},{order[2]})": arima_fc,
        "LSTM": lstm_pred,
        "Hybrid": hybrid_pred,
    }

    # Store predictions for DM test
    arima_preds_cache[ticker + "_all"] = {
        "NRW": nrw_pred,
        "ARIMA": arima_fc,
        "LSTM": lstm_pred,
        "Hybrid": hybrid_pred,
        "actual": actual,
    }

    for model_name, preds in models.items():
        m = compute_all_metrics(actual, preds, naive_mae, naive_mse)
        m["Stock"] = ticker
        m["Model"] = model_name
        comparison_rows.append(m)

        print(f"  {ticker:4s} {model_name:18s}  MAE={m['MAE']:10.2f}  RMSE={m['RMSE']:10.2f}  "
              f"MAPE={m['MAPE']:6.2f}%  R2={m['R2']:8.4f}  MASE={m['MASE']:.4f}  "
              f"RMSSE={m['RMSSE']:.4f}  DA={m['DA']:.2f}%")
    print()

    # -- Raw predictions for export --
    dates = test_data.index
    for i in range(n_test):
        raw_pred_rows.append({
            "Date": dates[i],
            "Stock": ticker,
            "Actual": round(aktual_full[i], 4),
            "NRW": round(nrw_pred[i], 4),
            "ARIMA": round(arima_fc[i], 4),
            "LSTM": round(lstm_pred[i], 4),
            "Hybrid": round(hybrid_pred[i], 4),
        })

# Save full comparison
df_comp = pd.DataFrame(comparison_rows)
cols_order = ["Stock", "Model", "MAE", "RMSE", "MAPE", "R2", "MASE", "RMSSE", "DA"]
df_comp = df_comp[cols_order]
df_comp.to_csv(os.path.join(BASE, "full_comparison.csv"), index=False)
print(f"  [OK] Saved: full_comparison.csv")

# Save raw predictions
df_raw = pd.DataFrame(raw_pred_rows)
df_raw.to_csv(os.path.join(BASE, "raw_predictions.csv"), index=False)
print(f"  [OK] Saved: raw_predictions.csv")


# ====================================================================
#  PART 3: DIEBOLD-MARIANO TEST
# ====================================================================
print("\n" + "="*70)
print("  PART 3: DIEBOLD-MARIANO TEST")
print("="*70)

dm_rows = []

for ticker in TICKERS:
    cache = arima_preds_cache[ticker + "_all"]
    actual = cache["actual"]

    preds_dict = {
        "NRW": cache["NRW"],
        "ARIMA": cache["ARIMA"],
        "LSTM": cache["LSTM"],
        "Hybrid": cache["Hybrid"],
    }

    # Pairwise DM tests
    model_names = list(preds_dict.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            # Use MSE loss (power=2)
            dm_stat, p_val = diebold_mariano_test(
                actual, preds_dict[m1], preds_dict[m2], h=1, power=2)
            sig = "Yes" if p_val < 0.05 else "No"
            better = ""
            if p_val < 0.05:
                mae1 = calc_mae(actual, preds_dict[m1])
                mae2 = calc_mae(actual, preds_dict[m2])
                better = m2 if mae2 < mae1 else m1

            dm_rows.append({
                "Stock": ticker,
                "Model_1": m1,
                "Model_2": m2,
                "DM_statistic": round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
                "p_value": round(p_val, 6) if not np.isnan(p_val) else np.nan,
                "Significant_5pct": sig,
                "Better_Model": better,
            })
            print(f"  {ticker} {m1:8s} vs {m2:8s}: DM={dm_stat:8.4f}, p={p_val:.6f} "
                  f"{'*** ' + better + ' better' if sig == 'Yes' else '(not significant)'}")
    print()

df_dm = pd.DataFrame(dm_rows)
df_dm.to_csv(os.path.join(BASE, "dm_test_results.csv"), index=False)
print(f"  [OK] Saved: dm_test_results.csv")


# ====================================================================
#  PART 4: EXPANDING-WINDOW CROSS-VALIDATION WITH ARIMA RE-SELECTION
# ====================================================================
print("\n" + "="*70)
print("  PART 4: EXPANDING-WINDOW CROSS-VALIDATION (ARIMA RE-SELECTION)")
print("="*70)
print("  Fold 1: Train <= 2019, Test = 2020")
print("  Fold 2: Train <= 2020, Test = 2021")
print("  Fold 3: Train <= 2021, Test = 2022")
print("="*70)

folds = [
    {"name": "Fold1", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"name": "Fold2", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "Fold3", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]

callbacks_def = [
    EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
]

ew_detail_rows = []
ew_order_rows = []

for fold in folds:
    print(f"\n  {'-'*60}")
    print(f"  {fold['name']}: Train <= {fold['train_end']}, Test = {fold['test_start']} to {fold['test_end']}")
    print(f"  {'-'*60}")

    for ticker in TICKERS:
        series = df[ticker].dropna()

        train_data = series[series.index <= fold["train_end"]]
        test_data  = series[(series.index >= fold["test_start"]) & (series.index <= fold["test_end"])]

        if len(test_data) == 0:
            print(f"    {ticker}: No test data, skipping")
            continue

        n_test = len(test_data)
        aktual = test_data.values

        # -- ARIMA with Grid Search re-selection --
        print(f"    {ticker}: ARIMA grid search...", end="", flush=True)
        t0 = time.time()
        best_order, best_aic, _ = arima_grid_search(train_data)
        print(f" order={best_order} AIC={best_aic:.2f} ({time.time()-t0:.1f}s)")

        ew_order_rows.append({
            "Fold": fold["name"],
            "Stock": ticker,
            "Best_Order": f"({best_order[0]},{best_order[1]},{best_order[2]})",
            "AIC": round(best_aic, 2),
        })

        # 1-step rolling ARIMA forecast for this fold
        try:
            print(f"    {ticker}: ARIMA rolling forecast", end="", flush=True)
            arima_fc, arima_fitted_ew, _ = arima_rolling_forecast(
                train_data, test_data, best_order
            )
            print(f" OK")
        except Exception as e:
            print(f"    {ticker} ARIMA forecast error: {e}")
            arima_fc = np.full(n_test, np.nan)

        # -- LSTM (pure) --
        try:
            scaler_pure = MinMaxScaler()
            train_scaled = scaler_pure.fit_transform(train_data.values.reshape(-1, 1)).flatten()

            full_for_test_ew = pd.concat([train_data, test_data])
            full_scaled_ew = scaler_pure.transform(full_for_test_ew.values.reshape(-1, 1)).flatten()

            X_train, y_train = create_sequences(train_scaled, WINDOW)
            X_train = X_train.reshape(-1, WINDOW, 1)

            n_train_ew = len(train_data)
            X_test_list = []
            for i in range(n_test):
                idx = n_train_ew + i
                X_test_list.append(full_scaled_ew[idx-WINDOW:idx])
            X_test = np.array(X_test_list).reshape(-1, WINDOW, 1)

            mdl_pure_ew = build_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
            mdl_pure_ew.fit(X_train, y_train, validation_split=0.15, epochs=EPOCHS,
                           batch_size=BATCH_SIZE, callbacks=callbacks_def, verbose=0)

            pred_scaled = mdl_pure_ew.predict(X_test, verbose=0).flatten()
            lstm_pred_ew = scaler_pure.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"    {ticker} LSTM error: {e}")
            lstm_pred_ew = np.full(n_test, np.nan)

        # -- Hybrid ARIMA-LSTM --
        try:
            arima_fitted_vals = arima_fitted_ew.fittedvalues
            if hasattr(arima_fitted_vals, 'values'):
                arima_fitted_vals = arima_fitted_vals.values
            resid_train_ew = train_data.values - arima_fitted_vals
            resid_train_ew = resid_train_ew[~np.isnan(resid_train_ew)]

            scaler_resid_ew = MinMaxScaler()
            resid_scaled_ew = scaler_resid_ew.fit_transform(resid_train_ew.reshape(-1, 1)).flatten()

            X_res_train, y_res_train = create_sequences(resid_scaled_ew, WINDOW)
            X_res_train = X_res_train.reshape(-1, WINDOW, 1)

            mdl_resid_ew = build_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
            mdl_resid_ew.fit(X_res_train, y_res_train, validation_split=0.15, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, callbacks=callbacks_def, verbose=0)

            last_win = resid_scaled_ew[-WINDOW:].tolist()
            resid_preds_ew = []
            for _ in range(n_test):
                inp = np.array(last_win[-WINDOW:]).reshape(1, WINDOW, 1)
                p = mdl_resid_ew.predict(inp, verbose=0).flatten()[0]
                resid_preds_ew.append(p)
                last_win.append(p)

            resid_pred_ew = scaler_resid_ew.inverse_transform(
                np.array(resid_preds_ew).reshape(-1, 1)).flatten()
            hybrid_pred_ew = arima_fc + resid_pred_ew
        except Exception as e:
            print(f"    {ticker} Hybrid error: {e}")
            hybrid_pred_ew = np.full(n_test, np.nan)

        # -- NRW --
        nrw_pred_ew = np.empty(n_test)
        nrw_pred_ew[0] = train_data.values[-1]
        nrw_pred_ew[1:] = aktual[:-1]

        # -- Compute metrics --
        naive_errors = aktual[1:] - aktual[:-1]
        naive_mae = np.mean(np.abs(naive_errors))
        naive_mse = np.mean(naive_errors**2)

        for model_name, preds in [("NRW", nrw_pred_ew), ("ARIMA", arima_fc),
                                   ("LSTM", lstm_pred_ew), ("Hybrid", hybrid_pred_ew)]:
            if np.any(np.isnan(preds)):
                m = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                     "R2": np.nan, "MASE": np.nan, "RMSSE": np.nan, "DA": np.nan}
            else:
                m = compute_all_metrics(aktual, preds, naive_mae, naive_mse)

            m["Fold"] = fold["name"]
            m["Stock"] = ticker
            m["Model"] = model_name
            ew_detail_rows.append(m)

        print(f"    {ticker}: NRW={calc_mape(aktual, nrw_pred_ew):.2f}%  "
              f"ARIMA={calc_mape(aktual, arima_fc):.2f}%  "
              f"LSTM={calc_mape(aktual, lstm_pred_ew):.2f}%  "
              f"Hybrid={calc_mape(aktual, hybrid_pred_ew):.2f}%")

# Create structured output directories
DIR_OUTPUTS = os.path.join(BASE, "outputs")
DIR_RESID = os.path.join(DIR_OUTPUTS, "residual_diagnostics")
DIR_EVAL = os.path.join(DIR_OUTPUTS, "evaluation")
DIR_STAT = os.path.join(DIR_OUTPUTS, "statistical_tests")
DIR_VAL = os.path.join(DIR_OUTPUTS, "validation")
DIR_LONG = os.path.join(DIR_OUTPUTS, "long_term")
DIR_REPRO = os.path.join(DIR_OUTPUTS, "reproducibility")

for d in [DIR_OUTPUTS, DIR_RESID, DIR_EVAL, DIR_STAT, DIR_VAL, DIR_LONG, DIR_REPRO]:
    os.makedirs(d, exist_ok=True)

# 1. Ljung-Box
df_lb.to_csv(os.path.join(DIR_RESID, "ljung_box_results.csv"), index=False)
df_lb.to_csv(os.path.join(BASE, "ljungbox_results.csv"), index=False)

# 2. Final predictions & metrics
df_raw.to_csv(os.path.join(DIR_EVAL, "final_test_predictions.csv"), index=False)
df_raw.to_csv(os.path.join(BASE, "raw_predictions.csv"), index=False)

df_comp.to_csv(os.path.join(DIR_EVAL, "final_metrics.csv"), index=False)
df_comp.to_csv(os.path.join(BASE, "full_comparison.csv"), index=False)

df_nrw = df_comp[df_comp["Model"] == "NRW"]
df_nrw.to_csv(os.path.join(DIR_EVAL, "random_walk_results.csv"), index=False)

# 3. Diebold-Mariano
df_dm.to_csv(os.path.join(DIR_STAT, "diebold_mariano_results.csv"), index=False)
df_dm.to_csv(os.path.join(BASE, "dm_test_results.csv"), index=False)

# 4. Expanding-window
df_ew_detail = pd.DataFrame(ew_detail_rows)
cols_ew = ["Fold", "Stock", "Model", "MAE", "RMSE", "MAPE", "R2", "MASE", "RMSSE", "DA"]
df_ew_detail = df_ew_detail[cols_ew]
df_ew_detail.to_csv(os.path.join(DIR_VAL, "expanding_window_results.csv"), index=False)
df_ew_detail.to_csv(os.path.join(BASE, "expanding_window_full.csv"), index=False)

df_ew_orders = pd.DataFrame(ew_order_rows)
df_ew_orders.to_csv(os.path.join(DIR_VAL, "expanding_window_orders.csv"), index=False)
df_ew_orders.to_csv(os.path.join(BASE, "expanding_window_orders.csv"), index=False)

df_ew_summary = df_ew_detail.groupby(["Stock", "Model"])[
    ["MAE", "RMSE", "MAPE", "R2", "MASE", "RMSSE", "DA"]
].mean().round(4).reset_index()
df_ew_summary.to_csv(os.path.join(DIR_VAL, "expanding_window_summary.csv"), index=False)
df_ew_summary.to_csv(os.path.join(BASE, "expanding_window_summary.csv"), index=False)

# 5. Long-term scenarios (copy from tabel06 if present or generate)
src_milestone = os.path.join(BASE, "tabel06_forecast_milestone_2045.csv")
if os.path.exists(src_milestone):
    df_long = pd.read_csv(src_milestone)
    df_long.to_csv(os.path.join(DIR_LONG, "long_term_scenarios_2025_2045.csv"), index=False)

# 6. Reproducibility info
import sys
import platform
env_info = f"""=======================================================================
Antigravity Research Pipeline Environment Information
=======================================================================
Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
OS: {platform.platform()}
Python: {sys.version}
Numpy: {np.__version__}
Pandas: {pd.__version__}
TensorFlow: {tf.__version__}
Seed: {SEED}
Evaluation Window: Test Set 2023-2024 (1-step-ahead rolling)
Expanding Window Folds: Fold 1 (2020), Fold 2 (2021), Fold 3 (2022)
=======================================================================
"""
with open(os.path.join(DIR_REPRO, "environment_info.txt"), "w") as f:
    f.write(env_info)

manifest = f"""=======================================================================
RUN MANIFEST -- GENERATED ARTIFACTS
=======================================================================
1. outputs/residual_diagnostics/ljung_box_results.csv
2. outputs/evaluation/final_test_predictions.csv
3. outputs/evaluation/final_metrics.csv
4. outputs/evaluation/random_walk_results.csv
5. outputs/statistical_tests/diebold_mariano_results.csv
6. outputs/validation/expanding_window_results.csv
7. outputs/validation/expanding_window_orders.csv
8. outputs/validation/expanding_window_summary.csv
9. outputs/long_term/long_term_scenarios_2025_2045.csv
10. outputs/reproducibility/environment_info.txt
11. outputs/reproducibility/run_manifest.txt
=======================================================================
"""
with open(os.path.join(DIR_REPRO, "run_manifest.txt"), "w") as f:
    f.write(manifest)

print("\n" + "="*70)
print("  PIPELINE COMPLETE -- ALL OUTPUT FILES GENERATED")
print("="*70)
print(f"  Structured directory: {DIR_OUTPUTS}")
print(manifest)
print("="*70)
print("  VERIFIED ARIMA ORDERS:")
for ticker, order in ARIMA_ORDERS.items():
    print(f"    {ticker}: {order}")
print("="*70)
print("  DONE")
print("="*70)
