import os, json, random
import numpy as np
import torch

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def norm_img_robust(x: np.ndarray) -> np.ndarray:
    """Robust per-image normalization -> [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    x = (x - p1) / (p99 - p1 + 1e-6)
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
