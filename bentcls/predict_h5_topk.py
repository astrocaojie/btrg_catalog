#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch

# =========================
# HARD-CODE SETTINGS
# =========================

# Input H5 (the 12,872 candidate groups)
IN_H5 = "/shared/main/caojie/meerkat/candidates/source_candidates.h5"

# Choose the model for inference
MODEL_NAME = "efficientnet_b0"   # "resnet18" or "convnext_tiny"

# Path to best checkpoint
MODEL_PT = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_efficientnet_b0_pre1/best.pt"
# If you switch MODEL_NAME, update MODEL_PT accordingly.

# Output directory
OUT_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls"

# Keep top fraction
KEEP_FRAC = 0.20   # top 20%

# Inference batch size
BATCH_INFER = 256

# Output subset H5 path
OUT_H5 = "/shared/main/caojie/meerkat/bestsource_top20pct.h5"

# =========================


def norm_img_robust(x: np.ndarray) -> np.ndarray:
    """Robust per-image normalization -> [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    x = (x - p1) / (p99 - p1 + 1e-6)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def copy_top_groups_to_h5(in_h5: str, out_h5: str, top_groups: list, scores_csv: str, keep_frac: float):
    """Copy selected groups from in_h5 to out_h5, preserving all attrs/datasets."""
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    t0 = time.time()
    copied = 0
    missing = 0

    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        # copy root attrs
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # provenance
        fout.attrs["subset_from_h5"] = in_h5
        fout.attrs["subset_scores_csv"] = scores_csv
        fout.attrs["subset_keep_frac"] = float(keep_frac)
        fout.attrs["subset_topk"] = int(len(top_groups))
        fout.attrs["subset_created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fout.attrs["subset_model_name"] = MODEL_NAME
        fout.attrs["subset_model_pt"] = MODEL_PT

        for gname in tqdm(top_groups, desc="copy top groups -> H5"):
            if gname not in fin:
                missing += 1
                continue
            # Copy the whole group (datasets + attrs) into fout
            fin.copy(fin[gname], fout, name=gname)
            copied += 1

        fout.attrs["num_groups"] = int(copied)
        fout.attrs["missing_groups"] = int(missing)

    dt = time.time() - t0
    print(f"[DONE] subset H5 saved: {out_h5}")
    print(f"[INFO] copied={copied} missing={missing} seconds={dt:.1f}")


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out_all = os.path.join(OUT_DIR, f"bent_scores_{MODEL_NAME}.csv")
    out_top = os.path.join(OUT_DIR, f"bent_top{int(KEEP_FRAC*100)}pct_{MODEL_NAME}.csv")

    if not os.path.exists(IN_H5):
        raise FileNotFoundError(f"IN_H5 not found: {IN_H5}")
    if not os.path.exists(MODEL_PT):
        raise FileNotFoundError(f"MODEL_PT not found: {MODEL_PT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] IN_H5:", IN_H5)
    print("[INFO] MODEL_NAME:", MODEL_NAME)
    print("[INFO] MODEL_PT:", MODEL_PT)
    print("[INFO] KEEP_FRAC:", KEEP_FRAC)
    print("[INFO] BATCH_INFER:", BATCH_INFER)
    print("[INFO] OUT_H5:", OUT_H5)

    # Import here so `python -m` works
    from .models import build_model

    # Load model
    model = build_model(model_name=MODEL_NAME, pretrained=False).to(device).eval()
    ckpt = torch.load(MODEL_PT, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    rows = []
    buf_x = []
    buf_meta = []

    def flush():
        nonlocal buf_x, buf_meta, rows
        if not buf_x:
            return

        X = np.stack(buf_x, axis=0)            # (B,H,W)
        Xt = torch.from_numpy(X[:, None]).to(device)  # (B,1,H,W)
        Xt = Xt.repeat(1, 3, 1, 1)            # (B,3,H,W)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            p = torch.sigmoid(model(Xt).squeeze(1)).detach().cpu().numpy()

        for (gname, ra, dec, isl), p_ in zip(buf_meta, p):
            rows.append({
                "group": gname,
                "P_bent": float(p_),
                "RA": float(ra),
                "DEC": float(dec),
                "Isl_id": isl,
            })

        buf_x = []
        buf_meta = []

    # ----- inference over H5
    with h5py.File(IN_H5, "r") as f:
        keys = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group)])
        print("[INFO] groups:", len(keys))

        for gname in tqdm(keys, desc="infer"):
            g = f[gname]
            if "Img" not in g:
                continue
            img = np.squeeze(g["Img"][()])
            if img.ndim != 2:
                continue

            x = norm_img_robust(img)
            buf_x.append(x)
            buf_meta.append((
                gname,
                g.attrs.get("RA", np.nan),
                g.attrs.get("DEC", np.nan),
                g.attrs.get("Isl_id", np.nan),
            ))

            if len(buf_x) >= BATCH_INFER:
                flush()

        flush()

    # ----- save CSVs
    df = pd.DataFrame(rows).sort_values("P_bent", ascending=False).reset_index(drop=True)
    df.to_csv(out_all, index=False)
    print("[DONE] all scores:", out_all, "N=", len(df))

    k = max(1, int(len(df) * KEEP_FRAC))
    df_top = df.head(k).copy()
    df_top.to_csv(out_top, index=False)
    print(f"[DONE] top {int(KEEP_FRAC*100)}%:", out_top, "K=", k)

    # ----- save subset H5
    top_groups = df_top["group"].tolist()
    copy_top_groups_to_h5(IN_H5, OUT_H5, top_groups, out_all, KEEP_FRAC)


if __name__ == "__main__":
    main()
