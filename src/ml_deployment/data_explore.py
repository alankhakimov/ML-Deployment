from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
BUSBRA_DIR = DATA_DIR / "BUSBRA" / "BUSBRA"

df = pd.read_csv(BUSBRA_DIR / "bus_data.csv")

# Label consistency within a case
print("cases with mixed Pathology:",
      (df.groupby("Case")["Pathology"].nunique() > 1).sum())

# Hold-out partitions: sizes and class balance
for col in ["HOB", "HOP"]:
    print("\n", col)
    print(pd.crosstab(df[col], df["Pathology"]))
    print(pd.crosstab(df[col], df["BIRADS"]))

# 5-fold: class balance per fold
for col in ["K5B", "K5P"]:
    print("\n", col)
    print(pd.crosstab(df[col], df["Pathology"], normalize="index").round(3))