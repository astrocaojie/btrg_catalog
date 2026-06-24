#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk, skeletonize


# =========================
# ✅按你的目录改这里
# =========================
PNG_DIR = "/shared/main/caojie/meerkat/candidates/00_export/png"         # 你的3000张png
META_CSV = "/shared/main/caojie/meerkat/candidates/00_export/meta.csv"  # 有就用，没有也可

OUT_IMG_DIR = "/shared/main/caojie/meerkat/candidates/01_annot/images"
OUT_LIST = "/shared/main/caojie/meerkat/candidates/01_annot/annot_list.csv"
OUT_FEATURES = "/shared/main/caojie/meerkat/candidates/01_annot/features_3000.csv"

# 你想挑多少张用于第一轮标注（建议 800~1200）
N_TOTAL = 500

# 分层配额（会按实际数量自动调整）
N_A = 450   # “最像弯尾”：端点>=2 且骨架长
N_B = 250   # “一般扩展”：面积/轴比/骨架中等
N_C = 100   # “难例/边界”：伪SNR高但结构不明显（防漏弱尾）
# =========================

CENTER = (64, 64)  # 128x128


def robust_stats(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return med, mad


def endpoints_count(skel):
    sk = skel.astype(np.uint8)
    p = np.pad(sk, 1, mode="constant")
    ys, xs = np.where(p == 1)
    cnt = 0
    for y, x in zip(ys, xs):
        nb = p[y-1:y+2, x-1:x+2].sum() - 1
        if nb == 1:
            cnt += 1
    return int(cnt)


def build_center_mask(img8):
    """
    From 8-bit PNG build a "center component" mask:
    - robust threshold using median + k*MAD, clipped into [p70, p90]
    - remove small objects, closing
    - keep component containing center; if none, keep largest component
    """
    img = img8.astype(np.float32)

    med, mad = robust_stats(img)
    thr = med + 2.5 * mad

    p70, p90, p97 = np.percentile(img, [70, 90, 97])
    thr = float(np.clip(thr, p70, p90))

    bw = (img >= thr)
    bw = remove_small_objects(bw, 10)
    bw = binary_closing(bw, disk(1))

    if bw.sum() == 0:
        return None, {"thr": thr, "note": "empty_bw"}

    lab = label(bw, connectivity=2)
    cy, cx = CENTER
    cl = lab[cy, cx]
    if cl != 0:
        return (lab == cl), {"thr": thr, "note": "center_component"}

    props = regionprops(lab)
    if not props:
        return None, {"thr": thr, "note": "no_props"}

    props.sort(key=lambda r: r.area, reverse=True)
    return (lab == props[0].label), {"thr": thr, "note": "largest_component"}


def compute_features_from_png(png_path):
    img8 = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)

    # pseudo SNR in PNG space: (peak - median) / MAD
    med, mad = robust_stats(img8)
    peak = float(np.max(img8))
    pseudo_snr = float((peak - med) / mad)

    mask, info = build_center_mask(img8)
    if mask is None:
        return {
            "png": png_path,
            "cand_id": os.path.splitext(os.path.basename(png_path))[0],
            "pseudo_snr": pseudo_snr,
            "area": 0.0, "elong": 1.0, "major": 0.0, "minor": 0.0,
            "sk_len": 0.0, "n_end": 0,
            "score_ext": 0.0,
            "mask_note": info.get("note", "none"),
            "thr": info.get("thr", np.nan),
        }

    lab = label(mask, connectivity=2)
    rp = regionprops(lab)[0]
    area = float(rp.area)
    major = float(rp.major_axis_length) if rp.major_axis_length else 0.0
    minor = float(rp.minor_axis_length) if rp.minor_axis_length else 0.0
    elong = float(major / (minor + 1e-6)) if major > 0 else 1.0

    sk = skeletonize(mask)
    sk_len = float(sk.sum())
    n_end = int(endpoints_count(sk))

    score_ext = area + 2.0 * sk_len + 20.0 * max(0.0, elong - 1.5) + 10.0 * max(0.0, n_end - 2)

    return {
        "png": png_path,
        "cand_id": os.path.splitext(os.path.basename(png_path))[0],
        "pseudo_snr": pseudo_snr,
        "area": area, "elong": elong, "major": major, "minor": minor,
        "sk_len": sk_len, "n_end": n_end,
        "score_ext": float(score_ext),
        "mask_note": info.get("note", ""),
        "thr": info.get("thr", np.nan),
    }


def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    png_list = sorted(glob.glob(os.path.join(PNG_DIR, "cand_*.png")))
    if not png_list:
        raise RuntimeError(f"No cand_*.png found in {PNG_DIR}")

    print(f"[INFO] Found PNG: {len(png_list)}")

    feats = []
    for p in tqdm(png_list, desc="Compute features"):
        feats.append(compute_features_from_png(p))

    df = pd.DataFrame(feats)

    # merge meta if exists
    if os.path.isfile(META_CSV):
        meta = pd.read_csv(META_CSV)
        df = df.merge(meta, on="cand_id", how="left")

    os.makedirs(os.path.dirname(OUT_FEATURES), exist_ok=True)
    df.to_csv(OUT_FEATURES, index=False)
    print("[INFO] Saved features:", OUT_FEATURES)

    # ===== Stratified selection =====
    # A: most bent-like: endpoints>=2 & decent skeleton length & decent pseudo_snr
    A = df[(df["n_end"] >= 2) & (df["sk_len"] >= 8) & (df["pseudo_snr"] >= 8)].copy()
    A = A.sort_values(["sk_len", "pseudo_snr", "score_ext"], ascending=False).head(N_A)

    exclude_ids = set(A["cand_id"])

    # B: general extended (mid-score examples)
    B_pool = df[(df["area"] >= 25) & (df["pseudo_snr"] >= 7)].copy()
    B_pool = B_pool[~B_pool["cand_id"].isin(exclude_ids)]
    B_pool = B_pool.sort_values("score_ext", ascending=False).reset_index(drop=True)

    if len(B_pool) > 0:
        mid_start = max(0, len(B_pool)//2 - N_B//2)
        B = B_pool.iloc[mid_start:mid_start+N_B].copy()
    else:
        B = B_pool.head(0)

    exclude_ids |= set(B["cand_id"])

    # C: hard/borderline: high pseudo_snr but <=1 endpoint
    C_pool = df[(df["pseudo_snr"] >= 10) & (df["n_end"] <= 1)].copy()
    C_pool = C_pool[~C_pool["cand_id"].isin(exclude_ids)]
    C_pool = C_pool.sort_values(["pseudo_snr", "sk_len", "score_ext"], ascending=False).head(N_C)

    sel = pd.concat([A, B, C_pool]).drop_duplicates(subset=["cand_id"]).reset_index(drop=True)

    # fill up to N_TOTAL by highest score_ext
    if len(sel) < N_TOTAL:
        need = N_TOTAL - len(sel)
        rest = df[~df["cand_id"].isin(set(sel["cand_id"]))].copy()
        rest = rest.sort_values("score_ext", ascending=False).head(need)
        sel = pd.concat([sel, rest]).drop_duplicates(subset=["cand_id"]).reset_index(drop=True)

    # trim if exceeding N_TOTAL (priority: A > B > C > rest)
    if len(sel) > N_TOTAL:
        Aset, Bset, Cset = set(A["cand_id"]), set(B["cand_id"]), set(C_pool["cand_id"])

        def prio(cid):
            if cid in Aset: return 0
            if cid in Bset: return 1
            if cid in Cset: return 2
            return 3

        sel["prio"] = sel["cand_id"].apply(prio)
        sel = sel.sort_values(["prio", "score_ext"], ascending=[True, False]).head(N_TOTAL).drop(columns=["prio"])

    print(f"[INFO] Selected for annotation: {len(sel)} (A={len(A)}, B={len(B)}, C={len(C_pool)})")
    sel.to_csv(OUT_LIST, index=False)
    print("[INFO] Saved annot list:", OUT_LIST)

    # copy images
    copied = 0
    for _, r in sel.iterrows():
        src = r["png"]
        dst = os.path.join(OUT_IMG_DIR, os.path.basename(src))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    print("[DONE] Copied images:", copied, "->", OUT_IMG_DIR)
    print("[TIP] 用 labelme 打开这个目录开始标注：", OUT_IMG_DIR)


if __name__ == "__main__":
    main()
