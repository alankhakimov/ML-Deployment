import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BUSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"])
        return self.transform(img), torch.tensor(row["label"], dtype=torch.float32)