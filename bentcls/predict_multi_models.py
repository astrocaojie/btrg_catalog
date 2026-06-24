#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run bent-tail inference for multiple trained models and cross-match their candidates.

Outputs:
1) One full-score CSV per model
2) One predicted-bent subset H5 per model
3) One cross-matched CSV merging all model outputs by `group`
4) Optional union/intersection subset H5 files
"""

import os
import time
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# =========================
# HARD-CODE SETTINGS
# =========================
IN_H5 = "/shared/main/caojie/meerkat/candidates/source_candidates.h5"
OUT_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/multi_model_infer"
BATCH_INFER = 256

MODEL_SPECS = [
    {
        "tag": "resnet18",
        "model_name": "resnet18",
        "model_pt": "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_resnet18_pre1/best.pt",
        "thr": 0.90,
        "write_subset_h5": True,
    },
    {
        "tag": "efficientnet_b0",
        "model_name": "efficientnet_b0",
        "model_pt": "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_efficientnet_b0_pre1/best.pt",
        "thr": 0.90,
        "write_subset_h5": True,
    },
    {
        "tag": "convnext_tiny",
        "model_name": "convnext_tiny",
        "model_pt": "/home/caojie/work/Galaxy-Morphology/bent_catalog/candidates/05_bent_cls/cmp_convnext_tiny_pre1/best.pt",
        "thr": 0.90,
        "write_subset_h5": True,
    },
]

WRITE_CROSSMATCH_UNION_H5 = True
WRITE_CROSSMATCH_INTERSECTION_H5 = True
# =========================


def norm_img_robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    x = (x - p1) / (p99 - p1 + 1e-6)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def copy_groups_to_h5(in_h5: str, out_h5: str, groups: List[str], attrs: Dict):
    ensure_parent(out_h5)
    copied = 0
    missing = 0

    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for k, v in attrs.items():
            fout.attrs[k] = v

        for gname in tqdm(groups, desc=f"copy -> {os.path.basename(out_h5)}"):
            if gname not in fin:
                missing += 1
                continue
            fin.copy(fin[gname], fout, name=gname)
            copied += 1

        fout.attrs["num_groups"] = int(copied)
        fout.attrs["missing_groups"] = int(missing)


@torch.no_grad()
def infer_one_model(spec: Dict, in_h5: str, out_dir: str) -> Dict:
    from .models import build_model

    model_name = spec["model_name"]
    model_pt = spec["model_pt"]
    tag = spec["tag"]
    thr = float(spec.get("thr", 0.90))

    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"Checkpoint not found for {tag}: {model_pt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name=model_name, pretrained=False).to(device).eval()

    ckpt = torch.load(model_pt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    csv_all = os.path.join(out_dir, f"{tag}_scores.csv")
    csv_cand = os.path.join(out_dir, f"{tag}_candidates_thr{str(thr).replace('.','p')}.csv")
    h5_cand = os.path.join(out_dir, f"{tag}_candidates_thr{str(thr).replace('.','p')}.h5")

    rows = []
    buf_x = []
    buf_meta = []

    def flush():
        nonlocal buf_x, buf_meta, rows
        if not buf_x:
            return

        x = np.stack(buf_x, axis=0)
        xt = torch.from_numpy(x[:, None]).to(device)
        xt = xt.repeat(1, 3, 1, 1)

        use_amp = (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            p = torch.sigmoid(model(xt).squeeze(1)).detach().cpu().numpy()

        for (gname, ra, dec, isl), score in zip(buf_meta, p):
            pb = float(score)
            pred = int(pb >= thr)
            rows.append({
                "group": gname,
                "RA": float(ra),
                "DEC": float(dec),
                "Isl_id": isl,
                "P_bent": pb,
                "bent_pred": pred,
                "model_tag": tag,
                "model_name": model_name,
                "threshold": thr,
            })

        buf_x = []
        buf_meta = []

    with h5py.File(in_h5, "r") as f:
        keys = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group)])
        for gname in tqdm(keys, desc=f"infer {tag}"):
            g = f[gname]
            if "Img" not in g:
                continue
            img = np.squeeze(g["Img"][()])
            if img.ndim != 2:
                continue

            buf_x.append(norm_img_robust(img))
            buf_meta.append((
                gname,
                g.attrs.get("RA", np.nan),
                g.attrs.get("DEC", np.nan),
                g.attrs.get("Isl_id", np.nan),
            ))

            if len(buf_x) >= BATCH_INFER:
                flush()
        flush()

    df = pd.DataFrame(rows).sort_values("P_bent", ascending=False).reset_index(drop=True)
    df.to_csv(csv_all, index=False)

    df_cand = df[df["bent_pred"] == 1].copy().reset_index(drop=True)
    df_cand.to_csv(csv_cand, index=False)

    if bool(spec.get("write_subset_h5", True)):
        copy_groups_to_h5(
            in_h5,
            h5_cand,
            df_cand["group"].tolist(),
            attrs={
                "subset_from_h5": in_h5,
                "subset_model_tag": tag,
                "subset_model_name": model_name,
                "subset_model_pt": model_pt,
                "subset_threshold": thr,
                "subset_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "subset_scores_csv": csv_cand,
            },
        )

    return {
        "tag": tag,
        "model_name": model_name,
        "model_pt": model_pt,
        "threshold": thr,
        "csv_all": csv_all,
        "csv_candidates": csv_cand,
        "h5_candidates": h5_cand,
        "n_all": int(len(df)),
        "n_candidates": int(len(df_cand)),
    }


def crossmatch_model_outputs(model_infos: List[Dict], out_dir: str, in_h5: str):
    merged = None

    for info in model_infos:
        tag = info["tag"]
        df = pd.read_csv(info["csv_all"])
        df = df[["group", "RA", "DEC", "Isl_id", "P_bent", "bent_pred"]].copy()
        df = df.rename(columns={
            "P_bent": f"P_bent_{tag}",
            "bent_pred": f"bent_pred_{tag}",
        })

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["group", "RA", "DEC", "Isl_id"], how="outer")

    if merged is None:
        raise RuntimeError("No model outputs to cross-match")

    pred_cols = [c for c in merged.columns if c.startswith("bent_pred_")]
    score_cols = [c for c in merged.columns if c.startswith("P_bent_")]

    merged["n_models_pred_bent"] = merged[pred_cols].fillna(0).astype(int).sum(axis=1)
    merged["pred_any_model"] = (merged["n_models_pred_bent"] >= 1).astype(int)
    merged["pred_all_models"] = (merged["n_models_pred_bent"] == len(pred_cols)).astype(int)

    if score_cols:
        merged["P_bent_mean"] = merged[score_cols].mean(axis=1, skipna=True)
        merged["P_bent_max"] = merged[score_cols].max(axis=1, skipna=True)
        merged["P_bent_min"] = merged[score_cols].min(axis=1, skipna=True)

    merged = merged.sort_values(
        ["pred_all_models", "n_models_pred_bent", "P_bent_mean", "P_bent_max"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out_csv = os.path.join(out_dir, "crossmatch_all_models.csv")
    merged.to_csv(out_csv, index=False)

    out_summary = os.path.join(out_dir, "crossmatch_summary.csv")
    summary_rows = []
    for info in model_infos:
        summary_rows.append({
            "tag": info["tag"],
            "model_name": info["model_name"],
            "threshold": info["threshold"],
            "n_all": info["n_all"],
            "n_candidates": info["n_candidates"],
        })
    summary_rows.append({
        "tag": "union",
        "model_name": "crossmatch",
        "threshold": np.nan,
        "n_all": int(len(merged)),
        "n_candidates": int(merged["pred_any_model"].sum()),
    })
    summary_rows.append({
        "tag": "intersection",
        "model_name": "crossmatch",
        "threshold": np.nan,
        "n_all": int(len(merged)),
        "n_candidates": int(merged["pred_all_models"].sum()),
    })
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    if WRITE_CROSSMATCH_UNION_H5:
        union_groups = merged.loc[merged["pred_any_model"] == 1, "group"].dropna().tolist()
        copy_groups_to_h5(
            in_h5,
            os.path.join(out_dir, "crossmatch_union_any_model.h5"),
            union_groups,
            attrs={
                "subset_rule": "union_of_all_models",
                "subset_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "subset_crossmatch_csv": out_csv,
            },
        )

    if WRITE_CROSSMATCH_INTERSECTION_H5:
        inter_groups = merged.loc[merged["pred_all_models"] == 1, "group"].dropna().tolist()
        copy_groups_to_h5(
            in_h5,
            os.path.join(out_dir, "crossmatch_intersection_all_models.h5"),
            inter_groups,
            attrs={
                "subset_rule": "intersection_of_all_models",
                "subset_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "subset_crossmatch_csv": out_csv,
            },
        )

    return out_csv, out_summary


def main():
    ensure_dir(OUT_DIR)

    if not os.path.exists(IN_H5):
        raise FileNotFoundError(f"IN_H5 not found: {IN_H5}")

    model_infos = []
    for spec in MODEL_SPECS:
        model_infos.append(infer_one_model(spec, IN_H5, OUT_DIR))

    out_csv, out_summary = crossmatch_model_outputs(model_infos, OUT_DIR, IN_H5)

    print("[DONE] per-model inference finished")
    for info in model_infos:
        print(
            f"[INFO] {info['tag']}: "
            f"all={info['n_all']} candidates={info['n_candidates']} "
            f"csv={info['csv_candidates']}"
        )
    print("[DONE] crossmatch csv:", out_csv)
    print("[DONE] crossmatch summary:", out_summary)


if __name__ == "__main__":
    main()
