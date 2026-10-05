"""
Hitung R² LSTM murni secara matematis.

Kita sudah punya dari hasil_evaluasi_LSTM.csv (alignment-corrected, WINDOW=30):
  ADRO: MAE=344.80, RMSE=404.24, MAPE=24.48%
  PTBA: MAE=254.20, RMSE=285.26, MAPE=11.74%
  INDY: MAE=70.20,  RMSE=91.79,  MAPE=4.32%
  ITMG: MAE=4649.92, RMSE=4784.36, MAPE=22.40%

R² = 1 - SS_res / SS_tot
SS_res = n * RMSE^2  (n = jumlah observasi evaluasi = 476 - 30 = 446)
SS_tot = sum((aktual - mean(aktual))^2)  => dihitung dari data CSV

"""
import pandas as pd
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("harga_saham_batubara.csv", index_col=0, parse_dates=True)

ticker_list = ["ADRO", "PTBA", "INDY", "ITMG"]

# RMSE dari hasil_evaluasi_LSTM.csv (alignment-corrected final)
rmse_final = {
    "ADRO": 404.24,
    "PTBA": 285.26,
    "INDY": 91.79,
    "ITMG": 4784.36,
}

# Split: 4723 train + 476 test  (total = 5199 setelah fillna)
N_TEST  = 476
WINDOW  = 30
N_EVAL  = N_TEST - WINDOW   # = 446 observasi yang dievaluasi

print(f"N_EVAL (observasi per emiten): {N_EVAL}\n")

results = []
for ticker in ticker_list:
    series = df[ticker].dropna()

    # Test set aktual (476 terakhir), lalu potong WINDOW pertama = 446 observasi
    aktual = series.values[-N_TEST:][WINDOW:]

    n      = len(aktual)
    mean_y = aktual.mean()
    SS_tot = np.sum((aktual - mean_y) ** 2)
    SS_res = n * (rmse_final[ticker] ** 2)
    r2     = 1 - SS_res / SS_tot

    print(f"{ticker}:")
    print(f"  n        = {n}")
    print(f"  mean_y   = {mean_y:.2f}")
    print(f"  SS_tot   = {SS_tot:,.2f}")
    print(f"  SS_res   = {SS_res:,.2f}  (n * RMSE²)")
    print(f"  R²       = {r2:.6f}\n")

    results.append({
        "Emiten":  ticker,
        "RMSE":    rmse_final[ticker],
        "n":       n,
        "SS_tot":  round(SS_tot, 2),
        "SS_res":  round(SS_res, 2),
        "R²":      round(r2, 6),
    })

df_r2 = pd.DataFrame(results)
print("=== TABEL AKHIR R² LSTM MURNI ===")
print(df_r2[["Emiten", "RMSE", "R²"]].to_string(index=False))
df_r2.to_csv("hasil_r2_lstm_final.csv", index=False)
print("\nDisimpan ke hasil_r2_lstm_final.csv")
