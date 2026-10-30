from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
BUSBRA_DIR = DATA_DIR / "BUSBRA" / "BUSBRA"

df = pd.read_csv(BUSBRA_DIR / "bus_data.csv")
print(df.head())