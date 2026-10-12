import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "clean_rerun")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. MAPE Comparison Bar Chart
df_metrics = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_metrics_comparison.csv"))
fig, ax = plt.subplots(figsize=(10, 5))
df_m_plot = df_metrics.pivot(index="Ticker", columns="Model", values="MAPE")
# Order columns logically
cols_order = [c for c in ["NRW (Naive)", "ARIMA(2, 1, 4)", "ARIMA(3, 1, 4)", "ARIMA(4, 1, 4)", "Hybrid ARIMA-LSTM", "LSTM (Standalone)"] if c in df_m_plot.columns]
df_m_plot = df_m_plot[[c for c in df_m_plot.columns if c in cols_order]]
df_m_plot.plot(kind="bar", ax=ax, width=0.75, colormap="viridis")
ax.set_title("Perbandingan MAPE (%) Out-of-Sample (2023-2024) - Data Bersih Bebas Pre-IPO", fontweight="bold", fontsize=11)
ax.set_ylabel("MAPE (%)", fontsize=10)
ax.set_xlabel("Emiten Saham", fontsize=10)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.xticks(rotation=0)
plt.legend(frameon=True, fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "clean_fig_mape_comparison.png"), dpi=300)
plt.close()
print("[OK] clean_fig_mape_comparison.png")

# 2. Directional Accuracy Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
df_da_plot = df_metrics.pivot(index="Ticker", columns="Model", values="DA")
df_da_plot = df_da_plot[[c for c in df_da_plot.columns if c in cols_order]]
df_da_plot.plot(kind="bar", ax=ax, width=0.75, colormap="Spectral")
ax.axhline(50, color="red", linestyle="--", alpha=0.7, label="Ambang Acak (50%)")
ax.set_title("Perbandingan Directional Accuracy / DA (%) Out-of-Sample (2023-2024)", fontweight="bold", fontsize=11)
ax.set_ylabel("Directional Accuracy (%)", fontsize=10)
ax.set_xlabel("Emiten Saham", fontsize=10)
ax.set_ylim(35, 55)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.xticks(rotation=0)
plt.legend(frameon=True, fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "clean_fig_da_comparison.png"), dpi=300)
plt.close()
print("[OK] clean_fig_da_comparison.png")

# 3. Expanding Window CV Summary Bar Chart
if os.path.exists(os.path.join(OUTPUT_DIR, "clean_expanding_window_summary.csv")):
    df_ew = pd.read_csv(os.path.join(OUTPUT_DIR, "clean_expanding_window_summary.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    df_ew_plot = df_ew.pivot(index="Ticker", columns="Model", values="MAPE")
    df_ew_plot.plot(kind="bar", ax=ax, width=0.75, colormap="magma")
    ax.set_title("Expanding-Window Validation: Rata-Rata MAPE (%) 3 Folds (2020, 2021, 2022)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Rata-rata MAPE (%)", fontsize=10)
    ax.set_xlabel("Emiten Saham", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=0)
    plt.legend(frameon=True, fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "clean_fig_expanding_window_mape.png"), dpi=300)
    plt.close()
    print("[OK] clean_fig_expanding_window_mape.png")

print("Seluruh visualisasi selesai disimpan.")
