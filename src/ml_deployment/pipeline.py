from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ml_deployment.data import load_split, make_folds
from ml_deployment.dataset import BUSDataset
from ml_deployment.transforms import build_transform


@dataclass
class Config:
    weights_path: str = ""  # fill in
    img_size: int = 224
    n_folds: int = 5
    val_fold: int = 0
    batch_size: int = 16
    num_workers: int = 0
    lr: float = 1e-4
    epochs: int = 10
    seed: int = 42


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)

    # ---- stage 1: which images go where -------------------------------------
    def load_data(self):
        c = self.cfg
        pool = load_split("pool")
        test_ids = set(load_split("test")["ID"])

        folded = make_folds(pool, c.n_folds, c.seed)
        self.train_df = folded[folded["fold"] != c.val_fold].reset_index(drop=True)
        self.val_df = folded[folded["fold"] == c.val_fold].reset_index(drop=True)

        # the guards that make the test set untouchable from here on
        for part in (self.train_df, self.val_df):
            assert set(part["ID"]).isdisjoint(test_ids), "test image leaked into training data"
        assert set(self.train_df["Case"]).isdisjoint(self.val_df["Case"]), "case split across train/val"

    # ---- stage 2: how an image becomes a tensor -----------------------------
    def build_transforms(self):
        self.tf = build_transform(self.cfg.img_size)

    # ---- stage 3: batching ---------------------------------------------------
    def build_loaders(self):
        c = self.cfg
        self.train_loader = DataLoader(
            BUSDataset(self.train_df, self.tf),
            batch_size=c.batch_size, shuffle=True, num_workers=c.num_workers,
        )
        self.val_loader = DataLoader(
            BUSDataset(self.val_df, self.tf),
            batch_size=c.batch_size, shuffle=False, num_workers=c.num_workers,
        )

    # ---- stage 4: the network -----------------------------------------------
    def build_model(self):
        raise NotImplementedError

    # ---- stages still to write ----------------------------------------------
    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError