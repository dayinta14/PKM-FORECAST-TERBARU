"""
================================================================================
run_clean_empirical_pipeline.py
================================================================================
Pipeline Rerun Empiris Penuh & Bersih (100% Bebas Data Leakage & Bebas Pre-IPO Flat)

Fitur Utama:
1. Pembersihan observasi pre-IPO buatan (ADRO mulai 2008-07-16, INDY mulai 2008-06-11,
   ITMG mulai 2007-12-18, PTBA mulai 2004-01-01).
2. Grid search AIC ARIMA (p,q in 0..4, d=1) pada data latih yang bersih.
3. Estimasi residual ARIMA in-sample bebas look-ahead bias.
4. Pelatihan model Standalone LSTM dan Residual LSTM dari nol (scratch)
   dengan pembagian train/val strictly kronologis (tanpa shuffle acak).
5. True 1-step-ahead rolling forecast (h=1) konsisten untuk ARIMA, LSTM, dan Hybrid:
   - ARIMA di-roll dengan append(refit=False) setelah prediksi t dicatat.
   - LSTM standalone menggunakan sliding window nilai aktual masa lalu (sampai t-1).
   - Hybrid LSTM residual menggunakan sliding window residual aktual masa lalu
     (e_{t-1} = y_{t-1} - L_{t-1}) untuk memprediksi koreksi e_hat_t.
6. Evaluasi 7 metrik lengkap: MAE, RMSE, MAPE, R2, MASE, RMSSE, Directional Accuracy (DA).
7. Uji diagnostik Ljung-Box, uji signifikansi Diebold-Mariano (Newey-West HAC),
   dan Expanding-Window Cross-Validation (3 Folds).
8. Ekspor seluruh artefak prediksi, metrik, tabel komparasi, dan visualisasi grafik.
================================================================================
"""

import os
import sys
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- REPRODUCIBILITY ---
SEED = 2024
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "clean_rerun")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Parameter Utama
TICKERS = ["ADRO", "PTBA", "INDY", "ITMG"]
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

# Tanggal IPO / Pencatatan Perdana Nyata di BEI (IDX)
# Sebelum tanggal ini, data di yfinance hasil backfill konstan (dibuang)
IPO_DATES = {
    "ADRO": "2008-07-16",
    "INDY": "2008-06-11",
    "ITMG": "2007-12-18",
    "PTBA": "2004-01-01",  # Listing 2002-12-23, aktif penuh sejak awal dataset
}

WINDOW = 30
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
ES_PATIENCE = 15

print("="*75)
print("  MEMULAI PIPELINE RERUN EMPIRIS BERSIH (CLEAN EMPIRICAL PIPELINE)")
print("="*75)


# ==============================================================================
# 1. LOAD DAN BERSIHKAN DATASET (ELIMINASI PRE-IPO FLAT ROWS)
# ==============================================================================
print("\n[1/7] Memuat dan membersihkan dataset harian...")
df_raw = pd.read_csv(os.path.join(BASE_DIR, "harga_saham_batubara.csv"),
                     index_col=0, parse_dates=True)

cleaned_series = {}
data_manifest = []

for ticker in TICKERS:
    raw_s = df_raw[ticker].dropna()
    ipo_date = IPO_DATES[ticker]
    # Filter hanya data pada atau setelah tanggal perdagangan perdana
    clean_s = raw_s[raw_s.index >= ipo_date].copy()
    
    train_part = clean_s[clean_s.index <= TRAIN_END]
    test_part = clean_s[clean_s.index >= TEST_START]
    
    cleaned_series[ticker] = {
        "full": clean_s,
        "train": train_part,
        "test": test_part
    }
    
    data_manifest.append({
        "Ticker": ticker,
        "IPO_Date": ipo_date,
        "Raw_Total_Rows": len(raw_s),
        "Clean_Total_Rows": len(clean_s),
        "Removed_Pre_IPO_Rows": len(raw_s) - len(clean_s),
        "Train_Rows (<=2022)": len(train_part),
        "Test_Rows (2023-2024)": len(test_part),
        "Min_Price": clean_s.min(),
        "Max_Price": clean_s.max()
    })

df_manifest = pd.DataFrame(data_manifest)
df_manifest.to_csv(os.path.join(OUTPUT_DIR, "clean_dataset_summary.csv"), index=False)
print(df_manifest.to_string(index=False))


# ==============================================================================
# 2. GRID SEARCH ARIMA AIC PADA DATA LATIH BERSIH
# ==============================================================================
print("\n" + "="*75)
print("[2/7] Menjalankan Exhaustive Grid Search ARIMA AIC (p in 0..4, q in 0..4, d=1)...")
print("="*75)

def run_arima_grid_search(series_train, p_range=range(5), q_range=range(5), d=1):
    vals = series_train.values
    best_aic = np.inf
    best_order = (0, d, 0)
    all_res = []
    
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                m = SARIMAX(vals, order=(p, d, q), trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                res = m.fit(disp=False, method="lbfgs", maxiter=250)
                all_res.append({
                    "order": (p, d, q),
                    "aic": res.aic,
                    "bic": res.bic,
                    "hqic": res.hqic,
                    "logl": res.llf
                })
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    df_res = pd.DataFrame(all_res).sort_values("aic").reset_index(drop=True)
    return best_order, best_aic, df_res

arima_best_orders = {}
arima_orders_summary = []
arima_fitted_models = {}
arima_in_sample_resid = {}

for ticker in TICKERS:
    tr = cleaned_series[ticker]["train"]
    print(f"  {ticker} ({len(tr)} baris latih)...", end="", flush=True)
    t0 = time.time()
    best_order, best_aic, grid_df = run_arima_grid_search(tr)
    dur = time.time() - t0
    arima_best_orders[ticker] = best_order
    print(f" Selesai ({dur:.1f}s) -> Best: ARIMA{best_order} (AIC={best_aic:.2f})")
    
    top2 = grid_df.iloc[1] if len(grid_df) > 1 else None
    top3 = grid_df.iloc[2] if len(grid_df) > 2 else None
    
    arima_orders_summary.append({
        "Ticker": ticker,
        "Best_Order": str(best_order),
        "AIC": round(best_aic, 2),
        "BIC": round(grid_df.iloc[0]["bic"], 2),
        "Candidate_2": f"ARIMA{top2['order']} (AIC={top2['aic']:.2f})" if top2 is not None else "",
        "Candidate_3": f"ARIMA{top3['order']} (AIC={top3['aic']:.2f})" if top3 is not None else "",
    })
    
    # Simpan hasil grid search lengkap
    grid_df.to_csv(os.path.join(OUTPUT_DIR, f"grid_search_{ticker}.csv"), index=False)
    
    # Fit model terbaik untuk in-sample residuals
    best_model = SARIMAX(tr.values, order=best_order, trend="n",
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False, method="lbfgs", maxiter=300)
    arima_fitted_models[ticker] = best_model
    # In-sample residual: e_t = y_t - fitted_t
    fitted_vals = best_model.fittedvalues
    resids = tr.values - fitted_vals
    arima_in_sample_resid[ticker] = resids

df_orders_summary = pd.DataFrame(arima_orders_summary)
df_orders_summary.to_csv(os.path.join(OUTPUT_DIR, "clean_arima_orders.csv"), index=False)
print("\nRingkasan Orde ARIMA Terpilih (Data Bersih):")
print(df_orders_summary.to_string(index=False))


# ==============================================================================
# 3. UJI DIAGNOSTIK RESIDUAL LJUNG-BOX
# ==============================================================================
print("\n" + "="*75)
print("[3/7] Menghitung Uji Diagnostik Ljung-Box pada Residual ARIMA Pelatihan...")
print("="*75)

ljungbox_rows = []
for ticker in TICKERS:
    # Buang resid awal (d=1 diff startup)
    res_clean = arima_in_sample_resid[ticker][1:]
    lb_df = acorr_ljungbox(res_clean, lags=[5, 10, 20], return_df=True)
    for lag in [5, 10, 20]:
        q_stat = lb_df.loc[lag, "lb_stat"]
        p_val = lb_df.loc[lag, "lb_pvalue"]
        ljungbox_rows.append({
            "Ticker": ticker,
            "Lag": lag,
            "Q_stat": round(q_stat, 4),
            "p_value": p_val,
            "Significant_5pct": "Ya" if p_val < 0.05 else "Tidak",
            "Interpretation": "Menolak White Noise (Dependensi non-linear signifikan)" if p_val < 0.05 else "Gagal Menolak White Noise"
        })

df_lb = pd.DataFrame(ljungbox_rows)
df_lb.to_csv(os.path.join(OUTPUT_DIR, "clean_ljungbox_results.csv"), index=False)
print(df_lb.to_string(index=False))


# ==============================================================================
# 4. PELATIHAN MODEL LSTM DARI AWAL (STANDALONE & RESIDUAL)
# ==============================================================================
print("\n" + "="*75)
print("[4/7] Melatih Ulang Model LSTM dari Nol (Chronological Split, Shuffle=False)...")
print("="*75)

def build_lstm_model(input_window):
    model = Sequential([
        Input(shape=(input_window, 1)),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model

def create_sequences_chronological(data_1d, window):
    X, y = [], []
    for i in range(window, len(data_1d)):
        X.append(data_1d[i-window:i])
        y.append(data_1d[i])
    return np.array(X)[..., np.newaxis], np.array(y)

trained_scalers_price = {}
trained_scalers_resid = {}
trained_lstm_pure = {}
trained_lstm_resid = {}

for ticker in TICKERS:
    tr_price = cleaned_series[ticker]["train"].values
    tr_resid = arima_in_sample_resid[ticker][1:]  # skip t=0 diff
    
    print(f"\n  --- Training LSTM untuk {ticker} ---")
    
    # 4A. Standalone LSTM (Harga)
    sc_price = MinMaxScaler(feature_range=(0, 1))
    sc_tr_price = sc_price.fit_transform(tr_price.reshape(-1, 1)).flatten()
    trained_scalers_price[ticker] = sc_price
    
    X_p, y_p = create_sequences_chronological(sc_tr_price, WINDOW)
    # Split kronologis 85% train, 15% validation
    n_p = len(X_p)
    split_p = int(n_p * 0.85)
    X_p_tr, y_p_tr = X_p[:split_p], y_p[:split_p]
    X_p_val, y_p_val = X_p[split_p:], y_p[split_p:]
    
    mdl_pure = build_lstm_model(WINDOW)
    cb_pure = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)
    ]
    mdl_pure.fit(X_p_tr, y_p_tr, validation_data=(X_p_val, y_p_val),
                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_pure,
                 shuffle=False, verbose=0)
    trained_lstm_pure[ticker] = mdl_pure
    mdl_pure.save(os.path.join(MODELS_DIR, f"clean_lstm_pure_{ticker}.keras"))
    print(f"    [OK] LSTM Standalone dilatih & disimpan (Train: {len(X_p_tr)}, Val: {len(X_p_val)})")
    
    # 4B. Residual LSTM (Koreksi Non-Linear)
    sc_resid = MinMaxScaler(feature_range=(-1, 1))
    sc_tr_resid = sc_resid.fit_transform(tr_resid.reshape(-1, 1)).flatten()
    trained_scalers_resid[ticker] = sc_resid
    
    X_r, y_r = create_sequences_chronological(sc_tr_resid, WINDOW)
    n_r = len(X_r)
    split_r = int(n_r * 0.85)
    X_r_tr, y_r_tr = X_r[:split_r], y_r[:split_r]
    X_r_val, y_r_val = X_r[split_r:], y_r[split_r:]
    
    mdl_resid = build_lstm_model(WINDOW)
    cb_resid = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)
    ]
    mdl_resid.fit(X_r_tr, y_r_tr, validation_data=(X_r_val, y_r_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_resid,
                  shuffle=False, verbose=0)
    trained_lstm_resid[ticker] = mdl_resid
    mdl_resid.save(os.path.join(MODELS_DIR, f"clean_lstm_resid_{ticker}.keras"))
    print(f"    [OK] LSTM Residual dilatih & disimpan (Train: {len(X_r_tr)}, Val: {len(X_r_val)})")


# ==============================================================================
# 5. ONE-STEP-AHEAD ROLLING FORECAST (h = 1) PADA TEST SET (2023-2024)
# ==============================================================================
print("\n" + "="*75)
print("[5/7] Menjalankan Evaluasi 1-Step-Ahead Rolling Forecast (h=1) Konsisten...")
print("="*75)

# Metrik Helpers
def calc_metrics(actual, pred, naive_errs):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    ss_res = np.sum((actual - pred)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    naive_mae = np.mean(np.abs(naive_errs))
    naive_mse = np.mean(naive_errs**2)
    mase = mae / naive_mae if naive_mae != 0 else np.nan
    rmsse = np.sqrt(np.mean((actual - pred)**2) / naive_mse) if naive_mse != 0 else np.nan
    
    # Directional Accuracy
    act_dir = np.sign(actual[1:] - actual[:-1])
    prd_dir = np.sign(pred[1:] - pred[:-1])
    da = np.mean(act_dir == prd_dir) * 100 if len(act_dir) > 0 else np.nan
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "MASE": mase,
        "RMSSE": rmsse,
        "DA": da
    }

predictions_all = []
metrics_summary = []

for ticker in TICKERS:
    tr_series = cleaned_series[ticker]["train"]
    te_series = cleaned_series[ticker]["test"]
    n_test = len(te_series)
    test_dates = te_series.index
    aktual_arr = te_series.values
    
    best_order = arima_best_orders[ticker]
    print(f"\n  --- Evaluasi Rolling Forecast untuk {ticker} (N={n_test} hari bursa) ---")
    
    # Inisialisasi status model
    # A. ARIMA rolling
    arima_res = arima_fitted_models[ticker]
    
    # B. Standalone LSTM window buffer
    sc_p = trained_scalers_price[ticker]
    # 30 titik terakhir dari training set
    price_buffer = list(sc_p.transform(tr_series.values[-WINDOW:].reshape(-1, 1)).flatten())
    
    # C. Hybrid Residual LSTM window buffer
    sc_r = trained_scalers_resid[ticker]
    # 30 residual terakhir dari training set
    resid_buffer = list(sc_r.transform(arima_in_sample_resid[ticker][-WINDOW:].reshape(-1, 1)).flatten())
    
    nrw_preds = []
    arima_preds = []
    lstm_preds = []
    hybrid_preds = []
    
    # Observasi terakhir sebelum test set dimulai (y_0)
    prev_actual = tr_series.values[-1]
    
    for t in range(n_test):
        y_t = aktual_arr[t]
        
        # 1. Naive Random Walk: y_hat_t = y_{t-1}
        nrw_fc = prev_actual
        nrw_preds.append(nrw_fc)
        
        # 2. ARIMA 1-step forecast: L_hat_t
        fc_step = arima_res.forecast(steps=1)
        arima_fc = float(fc_step[0] if hasattr(fc_step, "__getitem__") else fc_step)
        arima_preds.append(arima_fc)
        
        # 3. Standalone LSTM 1-step forecast: y_hat_lstm_t
        inp_p = np.array(price_buffer[-WINDOW:]).reshape(1, WINDOW, 1)
        pred_p_sc = trained_lstm_pure[ticker].predict(inp_p, verbose=0).flatten()[0]
        lstm_fc = float(sc_p.inverse_transform([[pred_p_sc]])[0, 0])
        lstm_preds.append(lstm_fc)
        
        # 4. Hybrid ARIMA-LSTM:
        # LSTM Residual memprediksi e_hat_t dari sliding window residual aktual masa lalu
        inp_r = np.array(resid_buffer[-WINDOW:]).reshape(1, WINDOW, 1)
        pred_r_sc = trained_lstm_resid[ticker].predict(inp_r, verbose=0).flatten()[0]
        resid_fc = float(sc_r.inverse_transform([[pred_r_sc]])[0, 0])
        hybrid_fc = arima_fc + resid_fc
        hybrid_preds.append(hybrid_fc)
        
        # ======================================================================
        # UPDATE PROTOKOL (Setelah waktu t diramalkan dan aktual y_t tiba):
        # Observasi aktual y_t HANYA dimasukkan ke riwayat untuk meramalkan t+1!
        # ======================================================================
        # Update ARIMA Kalman state (refit=False)
        arima_res = arima_res.append([y_t], refit=False)
        
        # Update Standalone LSTM buffer dengan harga aktual y_t
        scaled_actual = sc_p.transform([[y_t]])[0, 0]
        price_buffer.append(scaled_actual)
        
        # Update Hybrid Residual buffer dengan residual riil: e_t = y_t - arima_fc
        real_resid_t = y_t - arima_fc
        scaled_resid = sc_r.transform([[real_resid_t]])[0, 0]
        resid_buffer.append(scaled_resid)
        
        # Simpan untuk baris prediksi berikutnya
        prev_actual = y_t
        
        predictions_all.append({
            "Date": test_dates[t].strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Actual": y_t,
            "NRW": nrw_fc,
            "ARIMA": arima_fc,
            "LSTM": lstm_fc,
            "Hybrid": hybrid_fc,
            "ARIMA_Order": str(best_order)
        })
    
    # Hitung naive errors untuk MASE & RMSSE: e_{t} = y_t - y_{t-1}
    naive_diffs = aktual_arr[1:] - aktual_arr[:-1]
    
    # Hitung metrik masing-masing model
    nrw_p = np.array(nrw_preds)
    ari_p = np.array(arima_preds)
    lst_p = np.array(lstm_preds)
    hyb_p = np.array(hybrid_preds)
    
    m_nrw = calc_metrics(aktual_arr, nrw_p, naive_diffs)
    m_ari = calc_metrics(aktual_arr, ari_p, naive_diffs)
    m_lst = calc_metrics(aktual_arr, lst_p, naive_diffs)
    m_hyb = calc_metrics(aktual_arr, hyb_p, naive_diffs)
    
    for m_label, m_vals in [("NRW (Naive)", m_nrw),
                            (f"ARIMA{best_order}", m_ari),
                            ("LSTM (Standalone)", m_lst),
                            ("Hybrid ARIMA-LSTM", m_hyb)]:
        metrics_summary.append({
            "Ticker": ticker,
            "Model": m_label,
            "MAE": round(m_vals["MAE"], 4),
            "RMSE": round(m_vals["RMSE"], 4),
            "MAPE": round(m_vals["MAPE"], 4),
            "R2": round(m_vals["R2"], 6),
            "MASE": round(m_vals["MASE"], 4),
            "RMSSE": round(m_vals["RMSSE"], 4),
            "DA": round(m_vals["DA"], 2)
        })

df_preds = pd.DataFrame(predictions_all)
df_preds.to_csv(os.path.join(OUTPUT_DIR, "clean_test_predictions.csv"), index=False)

df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "clean_metrics_comparison.csv"), index=False)
print("\nHASIL METRIK EVALUASI TEST SET (2023-2024):")
print(df_metrics.to_string(index=False))


# ==============================================================================
# 6. UJI DIEBOLD-MARIANO (DM TEST)
# ==============================================================================
print("\n" + "="*75)
print("[6/7] Menghitung Uji Hipotesis Diebold-Mariano (Newey-West HAC, h=1)...")
print("="*75)

def diebold_mariano_test(actual, pred1, pred2, h=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2  # MSE loss differential
    n = len(d)
    mean_d = np.mean(d)
    
    # Newey-West variance estimation
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        cov = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2.0 * weight * cov
    
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        dm_stat = 0.0
        p_val = 1.0
    else:
        dm_stat = mean_d / np.sqrt(var_d)
        p_val = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_val

dm_results = []
for ticker in TICKERS:
    sub = df_preds[df_preds["Ticker"] == ticker]
    act = sub["Actual"].values
    nrw = sub["NRW"].values
    ari = sub["ARIMA"].values
    lst = sub["LSTM"].values
    hyb = sub["Hybrid"].values
    
    pairs = [
        ("Hybrid vs ARIMA", hyb, ari),
        ("Hybrid vs LSTM", hyb, lst),
        ("Hybrid vs NRW", hyb, nrw),
        ("ARIMA vs LSTM", ari, lst),
        ("ARIMA vs NRW", ari, nrw),
        ("LSTM vs NRW", lst, nrw),
    ]
    
    for pair_name, p1, p2 in pairs:
        stat, p = diebold_mariano_test(act, p1, p2)
        sig = "Ya" if p < 0.05 else "Tidak"
        m1_name, m2_name = pair_name.split(" vs ")
        # loss d = e1^2 - e2^2. Jika stat < 0 dan p < 0.05 -> Model 1 lebih baik
        if sig == "Ya":
            better = m1_name if stat < 0 else m2_name
        else:
            better = "Ekuivalen (Tidak beda signifikan)"
            
        dm_results.append({
            "Ticker": ticker,
            "Comparison": pair_name,
            "DM_Statistic": round(stat, 4),
            "p_value": p,
            "Significant_5pct": sig,
            "Superior_Model": better
        })

df_dm = pd.DataFrame(dm_results)
df_dm.to_csv(os.path.join(OUTPUT_DIR, "clean_dm_test_results.csv"), index=False)
print(df_dm.to_string(index=False))


# ==============================================================================
# 7. EXPANDING-WINDOW CROSS-VALIDATION (3 FOLDS KRONOLOGIS)
# ==============================================================================
print("\n" + "="*75)
print("[7/7] Menjalankan Expanding-Window Cross-Validation (3 Folds)...")
print("="*75)

ew_folds = [
    {"name": "Fold1", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"name": "Fold2", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"name": "Fold3", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
]

ew_detail_rows = []
ew_order_rows = []

for fold in ew_folds:
    print(f"\n  --- {fold['name']} (Train <= {fold['train_end']}, Test = {fold['test_start']} s.d. {fold['test_end']}) ---")
    
    for ticker in TICKERS:
        full_clean = cleaned_series[ticker]["full"]
        f_tr = full_clean[full_clean.index <= fold["train_end"]]
        f_te = full_clean[(full_clean.index >= fold["test_start"]) & (full_clean.index <= fold["test_end"])]
        
        if len(f_te) == 0:
            continue
            
        n_te = len(f_te)
        act_vals = f_te.values
        
        # 1. Grid search ulang ARIMA per fold
        f_best_order, f_best_aic, _ = run_arima_grid_search(f_tr)
        ew_order_rows.append({
            "Fold": fold["name"],
            "Ticker": ticker,
            "Best_Order": str(f_best_order),
            "AIC": round(f_best_aic, 2),
            "Train_Obs": len(f_tr),
            "Test_Obs": n_te
        })
        
        # Fit ARIMA fold
        f_arima_fit = SARIMAX(f_tr.values, order=f_best_order, trend="n",
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False, method="lbfgs", maxiter=250)
        f_resids = f_tr.values - f_arima_fit.fittedvalues
        f_resids = f_resids[1:]  # skip startup
        
        # 2. Train LSTM Pure fold
        f_sc_p = MinMaxScaler(feature_range=(0, 1))
        f_sc_tr_p = f_sc_p.fit_transform(f_tr.values.reshape(-1, 1)).flatten()
        X_p, y_p = create_sequences_chronological(f_sc_tr_p, WINDOW)
        split_p = int(len(X_p) * 0.85)
        
        mdl_pure = build_lstm_model(WINDOW)
        mdl_pure.fit(X_p[:split_p], y_p[:split_p],
                     validation_data=(X_p[split_p:], y_p[split_p:]),
                     epochs=60, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
        
        # 3. Train LSTM Resid fold
        f_sc_r = MinMaxScaler(feature_range=(-1, 1))
        f_sc_tr_r = f_sc_r.fit_transform(f_resids.reshape(-1, 1)).flatten()
        X_r, y_r = create_sequences_chronological(f_sc_tr_r, WINDOW)
        split_r = int(len(X_r) * 0.85)
        
        mdl_resid = build_lstm_model(WINDOW)
        mdl_resid.fit(X_r[:split_r], y_r[:split_r],
                      validation_data=(X_r[split_r:], y_r[split_r:]),
                      epochs=60, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
        
        # 4. Rolling evaluation fold (h=1)
        f_p_buffer = list(f_sc_p.transform(f_tr.values[-WINDOW:].reshape(-1, 1)).flatten())
        f_r_buffer = list(f_sc_r.transform(f_resids[-WINDOW:].reshape(-1, 1)).flatten())
        
        f_nrw_p = []
        f_ari_p = []
        f_lst_p = []
        f_hyb_p = []
        prev_val = f_tr.values[-1]
        
        cur_arima = f_arima_fit
        for i in range(n_te):
            y_i = act_vals[i]
            # NRW
            f_nrw_p.append(prev_val)
            # ARIMA
            fc_a = float(cur_arima.forecast(steps=1)[0])
            f_ari_p.append(fc_a)
            # LSTM Pure
            inp_p = np.array(f_p_buffer[-WINDOW:]).reshape(1, WINDOW, 1)
            pred_p = float(f_sc_p.inverse_transform([[mdl_pure.predict(inp_p, verbose=0)[0, 0]]])[0, 0])
            f_lst_p.append(pred_p)
            # Hybrid
            inp_r = np.array(f_r_buffer[-WINDOW:]).reshape(1, WINDOW, 1)
            pred_r = float(f_sc_r.inverse_transform([[mdl_resid.predict(inp_r, verbose=0)[0, 0]]])[0, 0])
            fc_h = fc_a + pred_r
            f_hyb_p.append(fc_h)
            
            # Update observasi aktual
            cur_arima = cur_arima.append([y_i], refit=False)
            f_p_buffer.append(f_sc_p.transform([[y_i]])[0, 0])
            f_r_buffer.append(f_sc_r.transform([[y_i - fc_a]])[0, 0])
            prev_val = y_i
            
        f_diffs = act_vals[1:] - act_vals[:-1]
        for m_name, p_arr in [("NRW", np.array(f_nrw_p)),
                              ("ARIMA", np.array(f_ari_p)),
                              ("LSTM", np.array(f_lst_p)),
                              ("Hybrid", np.array(f_hyb_p))]:
            m_res = calc_metrics(act_vals, p_arr, f_diffs)
            ew_detail_rows.append({
                "Fold": fold["name"],
                "Ticker": ticker,
                "Model": m_name,
                "MAE": round(m_res["MAE"], 4),
                "RMSE": round(m_res["RMSE"], 4),
                "MAPE": round(m_res["MAPE"], 4),
                "R2": round(m_res["R2"], 6),
                "MASE": round(m_res["MASE"], 4),
                "RMSSE": round(m_res["RMSSE"], 4),
                "DA": round(m_res["DA"], 2)
            })

df_ew_detail = pd.DataFrame(ew_detail_rows)
df_ew_detail.to_csv(os.path.join(OUTPUT_DIR, "clean_expanding_window_detail.csv"), index=False)

df_ew_orders = pd.DataFrame(ew_order_rows)
df_ew_orders.to_csv(os.path.join(OUTPUT_DIR, "clean_expanding_window_orders.csv"), index=False)

df_ew_summary = df_ew_detail.groupby(["Ticker", "Model"])[["MAE", "RMSE", "MAPE", "R2", "MASE", "RMSSE", "DA"]].mean().reset_index()
df_ew_summary.to_csv(os.path.join(OUTPUT_DIR, "clean_expanding_window_summary.csv"), index=False)

print("\nHASIL EXPANDING-WINDOW CROSS-VALIDATION (RATA-RATA 3 FOLDS):")
print(df_ew_summary.to_string(index=False))


# ==============================================================================
# 8. VISUALISASI GRAFIK PUBLIKASI JURNAL
# ==============================================================================
print("\n" + "="*75)
print("[Visualisasi] Membangkitkan Grafik Evaluasi Publikasi...")
print("="*75)

# Figure 1: Actual vs Predictions Test Set
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

colors_map = {
    "Actual": "#1a202c",
    "NRW": "#718096",
    "ARIMA": "#2b6cb0",
    "LSTM": "#dd6b20",
    "Hybrid": "#38a169"
}

for idx, ticker in enumerate(TICKERS):
    ax = axes[idx]
    sub = df_preds[df_preds["Ticker"] == ticker]
    dates = pd.to_datetime(sub["Date"])
    
    ax.plot(dates, sub["Actual"], label="Aktual", color=colors_map["Actual"], lw=1.4)
    ax.plot(dates, sub["ARIMA"], label=f"ARIMA{arima_best_orders[ticker]}", color=colors_map["ARIMA"], lw=1.1, ls="--")
    ax.plot(dates, sub["Hybrid"], label="Hybrid ARIMA-LSTM", color=colors_map["Hybrid"], lw=1.2, ls=":")
    ax.plot(dates, sub["LSTM"], label="LSTM Standalone", color=colors_map["LSTM"], lw=0.9, alpha=0.7)
    
    ax.set_title(f"{ticker} — Out-of-Sample Rolling Forecast (2023–2024)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Harga (IDR)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

plt.suptitle("Figure. Evaluasi 1-Step-Ahead Rolling Forecast Model pada Data Bersih Bebas Pre-IPO (2023–2024)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "clean_fig_forecast_test.png"), dpi=300)
plt.close()
print("  [OK] clean_fig_forecast_test.png")

# Figure 2: Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
df_m_plot = df_metrics.pivot(index="Ticker", columns="Model", values="MAPE")
df_m_plot.plot(kind="bar", ax=ax, colormap="tab10", width=0.75)
ax.set_title("Perbandingan MAPE (%) Antar Model per Emiten (Data Bersih)", fontweight="bold", fontsize=11)
ax.set_ylabel("MAPE (%)", fontsize=10)
ax.set_xlabel("Emiten", fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.xticks(rotation=0)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "clean_fig_mape_comparison.png"), dpi=300)
plt.close()
print("  [OK] clean_fig_mape_comparison.png")

print("\n" + "="*75)
print("  PIPELINE RERUN BERSIH SELESAI DENGAN SUKSES!")
print("="*75)
