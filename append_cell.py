import nbformat

notebook_path = "model_ann_lstm.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cell_content = """# --- EVALUASI LSTM STANDALONE KHUSUS SESUAI PERMINTAAN ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

evaluasi_lstm_standalone = []
prediksi_lstm_standalone = {}

print("=== Memulai Ulang Training LSTM Standalone ===")
for ticker in ticker_list:
    print(f"\\nTraining ulang {ticker}...")
    
    # 1. Gunakan pipeline & data yang sama dari dictionary data_lstm
    d = data_lstm[ticker]
    X_tr = d["X_tr"]
    y_tr = d["y_tr"]
    X_te = d["X_te"]
    scaler = d["scaler"]
    
    # 2. Bangun arsitektur LSTM dengan hyperparameter yang sama
    mdl = bangun_lstm(WINDOW, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)
    
    # 3. Fit model (seed sudah diset di awal notebook)
    hist = mdl.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_def,
        verbose=0
    )
    
    # 4. Prediksi pada test set
    pred_sc = mdl.predict(X_te, verbose=0).flatten()
    
    # 5. Inverse transform
    pred = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
    prediksi_lstm_standalone[ticker] = pred
    
    # Aktual: menyesuaikan dengan window size agar sejajar dengan output LSTM murni
    # (karena data_lstm X_te kehilangan WINDOW hari pertama dari test set)
    aktual = df_test[ticker].dropna().values[WINDOW:]
    
    # 6. Hitung metrik evaluasi
    mae = mean_absolute_error(aktual, pred)
    rmse = np.sqrt(mean_squared_error(aktual, pred))
    mape = np.mean(np.abs((aktual - pred) / aktual)) * 100
    
    evaluasi_lstm_standalone.append({
        "Emiten": ticker,
        "Model": "LSTM",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })
    print(f"Selesai {ticker} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# Buat DataFrame dan tampilkan
df_eval_lstm = pd.DataFrame(evaluasi_lstm_standalone)
print("\\n=== Tabel Hasil Evaluasi LSTM Standalone ===")
try:
    display(df_eval_lstm)
except:
    print(df_eval_lstm)

# Simpan ke CSV
output_csv_path = "hasil_evaluasi_LSTM.csv"
df_eval_lstm.to_csv(output_csv_path, index=False)
print(f"\\nBerhasil menyimpan hasil evaluasi ke: {output_csv_path}")

# Plot grafik perbandingan aktual vs prediksi LSTM Standalone
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for ax, ticker, warna in zip(axes, ticker_list, warna_saham):
    pred = prediksi_lstm_standalone[ticker]
    
    # Index test setelah dipotong window
    idx = df_test[ticker].dropna().index[WINDOW:]
    aktual = df_test[ticker].dropna().values[WINDOW:]
    
    ax.plot(idx, aktual, label="Aktual", color="navy", lw=1.5)
    ax.plot(idx, pred, label="Prediksi LSTM Standalone", color="green", ls="--", lw=1.5)
    
    ax.set_title(f"{ticker} - Aktual vs Prediksi (LSTM Standalone)", fontweight="bold")
    ax.set_ylabel("Harga (IDR)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.suptitle("Forecast LSTM Standalone pada Test Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
"""

# Cek apakah cell sudah ada untuk menghindari duplikasi
if nb.cells[-1].cell_type == "code" and "EVALUASI LSTM STANDALONE KHUSUS SESUAI PERMINTAAN" in nb.cells[-1].source:
    print("Cell sudah ada.")
else:
    new_cell = nbformat.v4.new_code_cell(source=new_cell_content)
    nb.cells.append(new_cell)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Successfully appended new cell to model_ann_lstm.ipynb")
