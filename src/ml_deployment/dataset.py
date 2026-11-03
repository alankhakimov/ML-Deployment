import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .data import load_split

# Class handles image, label loading for PyTorch DataLoader
class BUSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]) # a PIL image, still just pixels
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

def check_batch():
    ds = BUSDataset(load_split("pool"))

    print("length:", len(ds))
    img, label = ds[0]
    print("type:", type(img), "| mode:", img.mode, "| size:", img.size, "| label:", label)

    # save a few examples from each class so you can look at them
    df = ds.df
    for name, lab in [("benign", 0), ("malignant", 1)]:
        for k, i in enumerate(df.index[df["label"] == lab][:4]):
            ds[i][0].save(f"preview_{name}_{k}.png")

if __name__ == "__main__":
    check_batch()