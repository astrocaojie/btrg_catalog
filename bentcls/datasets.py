import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from .utils import norm_img_robust

class BentClsDataset(Dataset):
    """
    Binary classification dataset:
      - image: png (L, 128x128)
      - label: bent (0/1)
    """
    def __init__(self, df: pd.DataFrame, img_root: str, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _aug(self, x: np.ndarray) -> np.ndarray:
        # x: (H,W) in [0,1]
        import random
        if random.random() < 0.5:
            x = np.fliplr(x).copy()
        if random.random() < 0.5:
            x = np.flipud(x).copy()
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k).copy()

        # gentle intensity jitter
        if random.random() < 0.6:
            a = 0.90 + 0.20 * random.random()   # 0.90~1.10
            b = -0.04 + 0.08 * random.random()  # -0.04~0.04
            x = np.clip(a * x + b, 0.0, 1.0)
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fp = os.path.join(self.img_root, row["file"])
        img = np.array(Image.open(fp).convert("L"), dtype=np.float32)
        img = norm_img_robust(img)

        if self.augment:
            img = self._aug(img)

        # ResNet expects 3-ch, replicate
        x = torch.from_numpy(img[None, ...]).float()  # (1,H,W)
        x = x.repeat(3, 1, 1)                         # (3,H,W)

        y = torch.tensor(float(row["bent"]), dtype=torch.float32)  # scalar 0/1
        return x, y, row["file"]

def load_label_csv(label_csv: str, img_root: str, drop_uncertain: bool = True) -> pd.DataFrame:
    df = pd.read_csv(label_csv)

    if drop_uncertain and "uncertain" in df.columns:
        df = df[df["uncertain"].fillna(0).astype(int) == 0].copy()

    # keep only existing files
    df["exists"] = df["file"].apply(lambda x: os.path.exists(os.path.join(img_root, x)))
    df = df[df["exists"]].copy()
    df.drop(columns=["exists"], inplace=True)

    # ensure bent is 0/1 int
    df["bent"] = df["bent"].astype(int)
    return df
