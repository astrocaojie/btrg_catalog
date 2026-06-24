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
# HARD-CODE SETTINGS (edit here only)
# =========================

# Input H5 (your candidate set)
IN_H5 = "/shared/main/caojie/meerkat/candidates/source_candidates.h5"

# Which classifier to use
MODEL_NAME = "efficientnet_b0"     # "resnet18" / "convnext_tiny"

# !!! FIXED: your real checkpoint path (based on your earlier output)
MODEL_PT = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_efficientnet_b0_pre1/best.pt"
# If you switch MODEL_NAME, update MODEL_PT accordingly, e.g.:
# MODEL_PT = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_resnet18_pre1/best.pt"

# Output directory
OUT_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls"

# Threshold: P_bent >= THR => bent_pred=1
THR = 0.90

# Batch size for inference
BATCH_INFER = 256

# Whether to write a new H5 containing only predicted-bent groups
WRITE_BENT_H5 = True

# Output predicted-bent H5
OUT_BENT_H5 = "/shared/main/caojie/meerkat/candidates_effb0.h5"

# =========================


def norm_img_robust(x: np.ndarray) -> np.ndarray:
    """Robust per-image normalization -> [0,1]."""
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    x = (x - p1) / (p99 - p1 + 1e-6)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def copy_groups_to_h5(in_h5: str, out_h5: str, groups: list, scores_csv: str, thr: float):
    """Copy selected groups from in_h5 to out_h5, preserving all attrs/datasets."""
    _ensure_parent(out_h5)

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
        fout.attrs["subset_threshold"] = float(thr)
        fout.attrs["subset_created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fout.attrs["subset_model_name"] = MODEL_NAME
        fout.attrs["subset_model_pt"] = MODEL_PT

        for gname in tqdm(groups, desc="copy predicted-bent -> H5"):
            if gname not in fin:
                missing += 1
                continue
            fin.copy(fin[gname], fout, name=gname)
            copied += 1

        fout.attrs["num_groups"] = int(copied)
        fout.attrs["missing_groups"] = int(missing)

    dt = time.time() - t0
    print(f"[DONE] predicted-bent H5 saved: {out_h5}")
    print(f"[INFO] copied={copied} missing={missing} seconds={dt:.1f}")


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    out_csv = os.path.join(
        OUT_DIR,
        f"bent_pred_thr{str(THR).replace('.','p')}_{MODEL_NAME}.csv"
    )

    if not os.path.exists(IN_H5):
        raise FileNotFoundError(f"IN_H5 not found: {IN_H5}")
    if not os.path.exists(MODEL_PT):
        raise FileNotFoundError(f"MODEL_PT not found: {MODEL_PT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] IN_H5:", IN_H5)
    print("[INFO] MODEL_NAME:", MODEL_NAME)
    print("[INFO] MODEL_PT:", MODEL_PT)
    print("[INFO] THR:", THR)
    print("[INFO] BATCH_INFER:", BATCH_INFER)
    print("[INFO] out_csv:", out_csv)
    print("[INFO] WRITE_BENT_H5:", WRITE_BENT_H5)
    if WRITE_BENT_H5:
        print("[INFO] OUT_BENT_H5:", OUT_BENT_H5)

    # import here so `python -m bentcls.predict_h5` works
    from .models import build_model

    # load model
    model = build_model(model_name=MODEL_NAME, pretrained=False).to(device).eval()
    ckpt = torch.load(MODEL_PT, map_location="cpu")
    # expect ckpt["model"] as saved by train_compare.py
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    rows = []
    buf_x, buf_meta = [], []

    def flush():
        nonlocal buf_x, buf_meta, rows
        if not buf_x:
            return

        X = np.stack(buf_x, axis=0)                 # (B,H,W)
        Xt = torch.from_numpy(X[:, None]).to(device) # (B,1,H,W)
        Xt = Xt.repeat(1, 3, 1, 1)                  # (B,3,H,W)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            p = torch.sigmoid(model(Xt).squeeze(1)).detach().cpu().numpy()

        for (gname, ra, dec, isl), p_ in zip(buf_meta, p):
            pb = float(p_)
            pred = 1 if pb >= THR else 0
            rows.append({
                "group": gname,
                "P_bent": pb,
                "bent_pred": int(pred),
                "RA": float(ra),
                "DEC": float(dec),
                "Isl_id": isl,
            })

        buf_x, buf_meta = [], []

    # inference over H5
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

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n_pred = int((df["bent_pred"] == 1).sum())
    print("[DONE] saved CSV:", out_csv)
    print(f"[INFO] predicted bent: {n_pred} / {len(df)} ({n_pred/max(len(df),1)*100:.2f}%)")

    if WRITE_BENT_H5:
        bent_groups = df.loc[df["bent_pred"] == 1, "group"].tolist()
        copy_groups_to_h5(IN_H5, OUT_BENT_H5, bent_groups, out_csv, THR)


if __name__ == "__main__":
    main()
