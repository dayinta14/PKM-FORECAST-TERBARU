import json
import datetime
import pandas as pd
import numpy as np

# 1. Load CSV outputs from outputs/clean_rerun
arima_orders_df = pd.read_csv("outputs/clean_rerun/clean_arima_orders.csv")
metrics_df = pd.read_csv("outputs/clean_rerun/clean_metrics_comparison.csv")
test_preds_df = pd.read_csv("outputs/clean_rerun/clean_test_predictions.csv")
ew_summary_df = pd.read_csv("outputs/clean_rerun/clean_expanding_window_summary.csv")
dm_df = pd.read_csv("outputs/clean_rerun/clean_dm_test_results.csv")

# 2. Load clean historical series from harga_saham_batubara.csv
raw_hist = pd.read_csv("harga_saham_batubara.csv")
raw_hist["Date"] = pd.to_datetime(raw_hist["Date"])

IPO_DATES = {
    "ADRO": "2008-07-16",
    "PTBA": "2004-01-01",
    "INDY": "2008-06-11",
    "ITMG": "2007-12-18",
}

# 3. Load original data_saham.json to preserve forecast data in archive
with open("data_saham_backup.json", "r") as f:
    old_data = json.load(f)

new_stocks = {}

for ticker in ["ADRO", "PTBA", "INDY", "ITMG"]:
    old_stock = old_data.get("stocks", {}).get(ticker, {})
    
    # ARIMA order
    order_str = arima_orders_df.loc[arima_orders_df["Ticker"] == ticker, "Best_Order"].values[0]
    # parse "(p, d, q)"
    order_tuple = eval(order_str)
    
    # Filter historical train data (Date >= IPO and Date <= 2022-12-31)
    ipo_d = pd.to_datetime(IPO_DATES[ticker])
    stock_hist = raw_hist[(raw_hist["Date"] >= ipo_d) & (raw_hist["Date"] <= "2022-12-31")].sort_values("Date")
    hist_points = [
        {"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
        for d, v in zip(stock_hist["Date"], stock_hist[ticker])
    ]
    
    # Filter test predictions (476 trading days)
    st_preds = test_preds_df[test_preds_df["Ticker"] == ticker].sort_values("Date")
    test_actual = [
        {"date": d, "value": round(float(v), 2)}
        for d, v in zip(st_preds["Date"], st_preds["Actual"])
    ]
    test_predicted = [
        {"date": d, "value": round(float(v), 2)}
        for d, v in zip(st_preds["Date"], st_preds["Hybrid"])
    ]
    
    # All models test predictions
    test_models = []
    for _, row in st_preds.iterrows():
        test_models.append({
            "date": row["Date"],
            "actual": round(float(row["Actual"]), 2),
            "hybrid": round(float(row["Hybrid"]), 2),
            "arima": round(float(row["ARIMA"]), 2),
            "nrw": round(float(row["NRW"]), 2),
            "lstm": round(float(row["LSTM"]), 2)
        })
        
    # Model evaluation metrics
    stock_metrics = metrics_df[metrics_df["Ticker"] == ticker]
    models_dict = {}
    for _, row in stock_metrics.iterrows():
        m_name = row["Model"]
        key = "Hybrid" if "Hybrid" in m_name else ("ARIMA" if "ARIMA" in m_name else ("NRW" if "NRW" in m_name else "LSTM"))
        models_dict[key] = {
            "model_name": m_name,
            "MAE": round(float(row["MAE"]), 4),
            "RMSE": round(float(row["RMSE"]), 4),
            "MAPE": round(float(row["MAPE"]), 4),
            "R2": round(float(row["R2"]), 4),
            "MASE": round(float(row["MASE"]), 4),
            "RMSSE": round(float(row["RMSSE"]), 4),
            "DA": round(float(row["DA"]), 2)
        }
        
    # Expanding window metrics for this ticker
    ew_ticker = ew_summary_df[ew_summary_df["Ticker"] == ticker]
    ew_dict = {}
    for _, row in ew_ticker.iterrows():
        ew_dict[row["Model"]] = {
            "MAE": round(float(row["MAE"]), 4),
            "RMSE": round(float(row["RMSE"]), 4),
            "MAPE": round(float(row["MAPE"]), 4),
            "R2": round(float(row["R2"]), 4),
            "DA": round(float(row["DA"]), 2)
        }

    # Diebold Mariano comparisons for this ticker
    dm_ticker = dm_df[dm_df["Ticker"] == ticker]
    dm_dict = {}
    for _, row in dm_ticker.iterrows():
        dm_dict[row["Comparison"]] = {
            "stat": round(float(row["DM_Statistic"]), 4),
            "p_val": float(row["p_value"]),
            "significant": row["Significant_5pct"],
            "superior": row["Superior_Model"]
        }

    # Top-level metrics object for Hybrid
    hyb_m = models_dict["Hybrid"]
    primary_metrics = {
        "MAE": hyb_m["MAE"],
        "RMSE": hyb_m["RMSE"],
        "MAPE": hyb_m["MAPE"],
        "R2": hyb_m["R2"],
        "MASE": hyb_m["MASE"],
        "RMSSE": hyb_m["RMSSE"],
        "DA": hyb_m["DA"]
    }

    new_stocks[ticker] = {
        "ticker": f"{ticker}.JK",
        "order": list(order_tuple),
        "metrics": primary_metrics,
        "models": models_dict,
        "expanding_window_cv": ew_dict,
        "diebold_mariano": dm_dict,
        "historical": hist_points,
        "test_actual": test_actual,
        "test_predicted": test_predicted,
        "test_models": test_models,
        # Archive of previous forecast
        "forecast_archive": old_stock.get("forecast", []),
        # Note regarding 2045 projection
        "forecast_status": "Diarsipkan sementara (unvalidated univariate long-term forecast)",
        "forecast": []  # Disabled from main active forecast display per instruction 6
    }

new_data = {
    "meta": {
        "generated_at": datetime.datetime.now().isoformat(),
        "pipeline": "Clean Empirical Pipeline (Post-Audit SINTA 2 JISI Revision)",
        "data_source": "Yahoo Finance / IDX (Data Harian Bersih Bebas Pre-IPO)",
        "historical_period": "2004-01-01 to 2022-12-30 (Train Active Period per Stock)",
        "test_period": "2023-01-02 to 2024-12-30 (Test Period, N=476 trading days)",
        "evaluation_protocol": "One-step-ahead rolling forecast (h=1) strictly preventing look-ahead bias",
        "longterm_projection_status": "Proyeksi 2025-2045 dinonaktifkan sementara dan diarsipkan sesuai review reviewer JISI",
        "tickers": ["ADRO", "PTBA", "INDY", "ITMG"],
        "description": "Hasil peramalan harga saham batubara harian menggunakan model Hybrid ARIMA-LSTM, Standalone ARIMA, Naive Random Walk (NRW), dan Standalone LSTM."
    },
    "stocks": new_stocks
}

with open("data_saham.json", "w") as f:
    json.dump(new_data, f, indent=2)

print("Berhasil memperbarui data_saham.json!")
for t in ["ADRO", "PTBA", "INDY", "ITMG"]:
    s = new_stocks[t]
    print(f"\n{t}:")
    print(f"  Order: {s['order']}")
    print(f"  Hybrid Metrics: MAPE={s['metrics']['MAPE']}%, MAE={s['metrics']['MAE']}, RMSE={s['metrics']['RMSE']}, R2={s['metrics']['R2']}, DA={s['metrics']['DA']}%")
    print(f"  Train obs: {len(s['historical'])}, Test obs: {len(s['test_actual'])}")
