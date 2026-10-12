import json

with open("data_saham.json") as f:
    d = json.load(f)

print("Root keys:", list(d.keys()))
print("Meta:", d.get("meta"))
stocks = d.get("stocks", {})
print("Stocks:", list(stocks.keys()))
for ticker in stocks:
    s = stocks[ticker]
    print(f"\nStock {ticker}:")
    for k, v in s.items():
        if isinstance(v, list):
            print(f"  {k}: list of len {len(v)} (first: {v[0] if v else 'empty'})")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {v}")
