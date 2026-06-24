#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
from PIL import Image

IN_H5 = "/shared/main/caojie/meerkat/candidates/source_candidates.h5"

OUT_PNG_DIR = "/shared/main/caojie/meerkat/candidates/00_export/png"
OUT_META    = "/shared/main/caojie/meerkat/candidates/00_export/meta.csv"

MAX_EXPORT = 3000  # 先导出3000张。要全导出改成 None

os.makedirs(OUT_PNG_DIR, exist_ok=True)

def stretch_to_uint8(img):
    img = np.array(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    med = np.median(img)
    x = img - med

    lo, hi = np.percentile(x, [1, 99.7])
    if hi <= lo:
        lo, hi = x.min(), x.max() + 1e-6

    x = np.clip(x, lo, hi)

    # asinh stretch
    x = np.arcsinh(x - lo + 1e-6)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (255.0 * x).astype(np.uint8)

def main():
    rows = []
    with h5py.File(IN_H5, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith("cand_")])
        if MAX_EXPORT is not None:
            keys = keys[:MAX_EXPORT]

        for k in tqdm(keys, desc="Export PNG"):
            g = f[k]
            img = g["Img"][...]
            x8 = stretch_to_uint8(img)

            png_path = os.path.join(OUT_PNG_DIR, f"{k}.png")
            Image.fromarray(x8).save(png_path)

            rows.append({
                "cand_id": k,
                "png_path": png_path,
                "origin_h5": g.attrs.get("origin_h5", ""),
                "origin_group": g.attrs.get("origin_group", ""),
                "RA": g.attrs.get("RA", np.nan),
                "DEC": g.attrs.get("DEC", np.nan),
                "score": g.attrs.get("score", np.nan),
                "peak_snr": g.attrs.get("peak_snr", np.nan),
                "selected_by_topk": g.attrs.get("selected_by_topk", 0),
                "selected_by_fallback": g.attrs.get("selected_by_fallback", 0),
            })

    pd.DataFrame(rows).to_csv(OUT_META, index=False)
    print("[DONE] PNG:", OUT_PNG_DIR)
    print("[DONE] META:", OUT_META)

if __name__ == "__main__":
    main()
