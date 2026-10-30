from pathlib import Path
import pandas as pd
from collections import Counter
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
BUSBRA_DIR = DATA_DIR / "BUSBRA" / "BUSBRA"
IMG_DIR = BUSBRA_DIR / "images"

df = pd.read_csv(BUSBRA_DIR / "bus_data.csv")

def img_path(i: str) -> Path:
    return IMG_DIR / f"{i}.png"

size_bad, modes = [], Counter()
for _, r in df.iterrows():
    with Image.open(img_path(r["ID"])) as im:
        modes[im.mode] += 1
        if im.size != (r["Width"], r["Height"]):
            size_bad.append((r["ID"], im.size, (r["Width"], r["Height"])))

print("color modes:", dict(modes))
print("size mismatches:", len(size_bad), size_bad[:5])
print(df[["Width", "Height"]].describe().loc[["min", "max", "mean"]])

test_ids = set(pd.read_csv(ROOT_DIR / "splits" / "test_ids.csv")["ID"])
for i, actual, table in size_bad:
    row = df[df["ID"] == i].iloc[0]
    print(i, "| bbox:", row["BBOX"], "| actual:", actual,
          "| in test:", i in test_ids)