import nbformat

notebook_path = "model_ann_lstm.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "hasil_evaluasi_LSTM.csv" in cell.source:
            cell.source = cell.source.replace("hasil_evaluasi_LSTM.csv", "hasil_evaluasi_LSTM_v2.csv")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Updated notebook to avoid PermissionError on hasil_evaluasi_LSTM.csv")
