import nbformat

notebook_path = "model_ann_lstm.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

new_cell_content = """# Menghitung R2 untuk LSTM Murni (Standalone)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

hasil_lstm_final = []

for ticker in ticker_list:
    # Actual harus dimulai setelah WINDOW=30
    aktual = df_test[ticker].dropna().values[WINDOW:]

    # Prediksi LSTM yang sudah dibuat sebelumnya
    pred = lstm_pred_pure[ticker]

    # Samakan panjang untuk keamanan
    n = min(len(aktual), len(pred))
    aktual = aktual[:n]
    pred = pred[:n]

    mae = mean_absolute_error(aktual, pred)
    rmse = np.sqrt(mean_squared_error(aktual, pred))
    mape = np.mean(np.abs((aktual - pred) / aktual)) * 100
    r2 = r2_score(aktual, pred)

    hasil_lstm_final.append({
        "Emiten": ticker,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R²": r2
    })

df_lstm_final = pd.DataFrame(hasil_lstm_final)
display(df_lstm_final)
"""

new_cell = nbformat.v4.new_code_cell(source=new_cell_content)
nb.cells.append(new_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully appended R2 calculation cell to model_ann_lstm.ipynb")
