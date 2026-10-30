from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
BUSBRA_DIR = DATA_DIR / "BUSBRA" / "BUSBRA"
IMG_DIR = BUSBRA_DIR / "images"
SPLIT_DIR = ROOT_DIR / "splits"
LABELS_CSV = BUSBRA_DIR / "bus_data.csv"

LABEL_MAP = {"benign": 0, "malignant": 1}


def img_path(id_: str) -> Path:
    return IMG_DIR / f"{id_}.png"


def load_split(name: str) -> pd.DataFrame:
    """name is 'pool' or 'test'. Returns rows with ID, Case, label, path."""
    assert name in {"pool", "test"}
    ids = pd.read_csv(SPLIT_DIR / f"{name}_ids.csv")["ID"]
    df = pd.read_csv(LABELS_CSV)
    df = df[df["ID"].isin(ids)].reset_index(drop=True)
    df["label"] = df["Pathology"].map(LABEL_MAP)
    df["path"] = df["ID"].map(img_path)
    return df[["ID", "Case", "label", "path"]]

if __name__ == "__main__":
    pool, test = load_split("pool"), load_split("test")
    assert len(pool) == 1489 and len(test) == 386
    assert set(pool["Case"]).isdisjoint(test["Case"])
    assert set(pool["ID"]).isdisjoint(test["ID"])
    for df in (pool, test):
        assert df["label"].isin([0, 1]).all()
        assert df["path"].map(lambda p: p.exists()).all()
    print("data checks passed")