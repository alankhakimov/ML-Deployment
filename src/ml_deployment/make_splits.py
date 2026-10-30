from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
BUSBRA_DIR = DATA_DIR / "BUSBRA" / "BUSBRA"

# placeholder for the output directory
OUT = ROOT_DIR / "splits"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BUSBRA_DIR / "bus_data.csv")
test = df[df["HOP"] == 1]
pool = df[df["HOP"] == 2]

assert set(test["Case"]).isdisjoint(pool["Case"]) # verify no patient leakage
assert len(test) + len(pool) == len(df)

pool[["ID"]].to_csv(OUT / "pool_ids.csv", index=False)
test[["ID"]].to_csv(OUT / "test_ids.csv", index=False)

for name, part in [("pool", pool), ("test", test)]:
    print(name, len(part), part["Pathology"].value_counts(normalize=True).round(3).to_dict())