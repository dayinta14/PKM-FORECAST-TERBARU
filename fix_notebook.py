import nbformat

notebook_path = "model_ann_lstm.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "tabel05_perbandingan_model.csv" in cell.source:
            cell.source = cell.source.replace("tabel05_perbandingan_model.csv", "tabel05_perbandingan_model_v2.csv")

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Updated notebook to avoid PermissionError")
