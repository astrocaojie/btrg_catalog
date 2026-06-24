#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
import h5py
from tqdm import tqdm

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, disk, skeletonize

# =======================
# IO paths
# =======================
IN_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/source_crop_all/"
H5_GLOB = os.path.join(IN_DIR, "*.h5")
OUT_H5  = "/shared/main/caojie/meerkat/candidates/source_candidates.h5"

# =======================
# Cutout constants
# =======================
CENTER = (64, 64)
FILL_VALUE = 0.0
FILL_EPS = 1e-8

# =======================
# Stage0 filtering thresholds
# =======================
BAD_FILL_FRAC_THR = 0.35      # fill > 35% => bad partial
BAD_BBOX_FRAC_THR = 0.60      # valid bbox too narrow => bad partial
NOISE_PEAKSNR_THR = 6.0       # pure noise remove (local peak SNR)

# =======================
# Mask extraction
# =======================
NSIGMA_SHAPE = 4.0            # low threshold for "score" (more inclusive)
NSIGMA_FALLBACK = 7.0         # high threshold for fallback (much cleaner)
MIN_OBJ_PIX = 8

# =======================
# Candidate selection
# =======================
KEEP_FRAC = 0.20              # Top 20% by score
FALLBACK_MAX_FRAC = 0.05      # fallback最多占 all_kept 的 5%（保证“少量保底”）

# Strict fallback thresholds (based on HIGH-threshold mask features)
FALLBACK_PEAKSNR_THR = 10.0   # stronger than before to avoid noise/point sources
FALLBACK_NEND_THR = 2         # >=2 endpoints (double-tail-like)
FALLBACK_SKLEN_THR = 12       # skeleton length threshold
FALLBACK_AREA_HI_THR = 20     # area at high threshold

# -----------------------
# Utils
# -----------------------
def robust_rms(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = np.nanstd(x)
    if not np.isfinite(rms) or rms <= 0:
        rms = 1e-6
    return float(med), float(rms)

def estimate_local_bkg(img, r_in=20, r_out=55):
    img = np.asarray(img, dtype=np.float32)
    yy, xx = np.indices(img.shape)
    cy, cx = CENTER
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    ring = img[(rr >= r_in) & (rr <= r_out)]
    if ring.size < 200:
        return robust_rms(img)
    return robust_rms(ring)

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

def is_bad_partial(img):
    img = np.asarray(img)
    valid = np.abs(img - FILL_VALUE) > FILL_EPS
    valid_frac = valid.mean()
    fill_frac = float(1.0 - valid_frac)

    if fill_frac > BAD_FILL_FRAC_THR:
        return True, {"fill_frac": fill_frac}

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return True, {"fill_frac": 1.0}

    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    if (w < BAD_BBOX_FRAC_THR * img.shape[1]) or (h < BAD_BBOX_FRAC_THR * img.shape[0]):
        return True, {"fill_frac": fill_frac, "valid_bbox_w": w, "valid_bbox_h": h}

    return False, {"fill_frac": fill_frac, "valid_bbox_w": w, "valid_bbox_h": h}

def build_center_mask(img, med, rms, nsigma):
    """Threshold at nsigma, keep only connected component containing center. Return mask or None."""
    thr = med + nsigma * rms
    bw = (img > thr)
    bw = remove_small_objects(bw, MIN_OBJ_PIX)
    bw = binary_closing(bw, disk(1))
    if bw.sum() == 0:
        return None

    lab = label(bw, connectivity=2)
    cy, cx = CENTER
    cl = lab[cy, cx]
    if cl == 0:
        return None
    return (lab == cl)

def main_object_masks(img):
    """Return (mask_lo, mask_hi, bginfo)."""
    img = np.asarray(img, dtype=np.float32)
    med, rms = estimate_local_bkg(img)
    peak_snr = float((np.nanmax(img) - med) / rms)

    mask_lo = build_center_mask(img, med, rms, NSIGMA_SHAPE)
    mask_hi = build_center_mask(img, med, rms, NSIGMA_FALLBACK)

    return mask_lo, mask_hi, {"med": med, "rms": rms, "peak_snr": peak_snr}

def compute_features(mask):
    """Compute area/elong/skeleton/endpoints + score from a mask."""
    if mask is None:
        return {
            "area": 0.0, "major": 0.0, "minor": 0.0, "elong": 1.0,
            "sk_len": 0.0, "n_end": 0.0,
            "score": 0.0,
            "note": "no_mask"
        }

    lab = label(mask, connectivity=2)
    rp = regionprops(lab)[0]
    area = float(rp.area)
    major = float(rp.major_axis_length) if rp.major_axis_length else 0.0
    minor = float(rp.minor_axis_length) if rp.minor_axis_length else 0.0
    elong = (major / (minor + 1e-6)) if major > 0 else 1.0

    sk = skeletonize(mask)
    sk_len = float(sk.sum())
    n_end = float(endpoints_count(sk))

    # extension score (low mask): bigger => more structured/extended
    score = area + 2.0*sk_len + 20.0*max(0.0, elong-1.5) + 10.0*max(0.0, n_end-2.0)

    return {
        "area": area, "major": major, "minor": minor, "elong": float(elong),
        "sk_len": sk_len, "n_end": n_end,
        "score": float(score),
        "note": ""
    }

def is_fallback_candidate_strict(d):
    """Strict fallback using HIGH-threshold features only."""
    peak_snr = float(d.get("peak_snr", 0.0))
    if peak_snr < FALLBACK_PEAKSNR_THR:
        return False

    area_hi = float(d.get("area_hi", 0.0))
    sk_len_hi = float(d.get("sk_len_hi", 0.0))
    n_end_hi  = float(d.get("n_end_hi", 0.0))

    # 必须高阈值下仍有“像双尾/双瓣”的迹象
    if (n_end_hi >= FALLBACK_NEND_THR) and (area_hi >= FALLBACK_AREA_HI_THR):
        return True
    if (sk_len_hi >= FALLBACK_SKLEN_THR) and (n_end_hi >= FALLBACK_NEND_THR):
        return True

    return False

def copy_all_datasets(src_group, dst_group):
    """Copy all datasets inside a group (Img, Source_id, Source_Name...)."""
    for k, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            data = item[...]
            dst_group.create_dataset(
                k,
                data=data,
                dtype=item.dtype,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

def copy_all_attrs(src_obj, dst_obj):
    """Copy all attributes of a group or file."""
    for k in src_obj.attrs.keys():
        try:
            dst_obj.attrs[k] = src_obj.attrs[k]
        except Exception:
            dst_obj.attrs[k] = str(src_obj.attrs[k])

# -----------------------
# Main
# -----------------------
def main():
    h5_list = sorted(glob.glob(H5_GLOB))
    if not h5_list:
        raise RuntimeError(f"No input h5 found: {H5_GLOB}")

    print(f"[INFO] Found h5: {len(h5_list)}")
    print("[INFO] Stage0: remove bad_partial + bad_noise; compute features...")

    all_kept = []
    total_groups = 0
    bad_partial = 0
    bad_noise = 0

    for h5_path in tqdm(h5_list, desc="Scan H5"):
        with h5py.File(h5_path, "r") as f:
            for gname in f.keys():
                if not gname.startswith("data_"):
                    continue
                g = f[gname]
                if "Img" not in g:
                    continue

                total_groups += 1
                img = g["Img"][...]

                isbad, _ = is_bad_partial(img)
                if isbad:
                    bad_partial += 1
                    continue

                mask_lo, mask_hi, bginfo = main_object_masks(img)
                peak_snr = float(bginfo.get("peak_snr", np.nan))

                if (not np.isfinite(peak_snr)) or (peak_snr < NOISE_PEAKSNR_THR):
                    bad_noise += 1
                    continue

                feats_lo = compute_features(mask_lo)  # used for score ranking
                feats_hi = compute_features(mask_hi)  # used for strict fallback

                all_kept.append({
                    "h5_path": h5_path,
                    "group": gname,
                    "peak_snr": peak_snr,

                    # low-mask features
                    "score": feats_lo["score"],
                    "area": feats_lo["area"],
                    "elong": feats_lo["elong"],
                    "sk_len": feats_lo["sk_len"],
                    "n_end": feats_lo["n_end"],

                    # high-mask features
                    "area_hi": feats_hi["area"],
                    "sk_len_hi": feats_hi["sk_len"],
                    "n_end_hi": feats_hi["n_end"],
                })

    if len(all_kept) == 0:
        raise RuntimeError("No groups left after filtering. Try lowering NOISE_PEAKSNR_THR or relaxing partial thresholds.")

    print("[INFO] Total groups scanned:", total_groups)
    print("[INFO] Removed bad_partial:", bad_partial)
    print("[INFO] Removed bad_noise:", bad_noise)
    print("[INFO] Remaining after Stage0:", len(all_kept))

    # Top-K by score
    all_kept.sort(key=lambda d: d["score"], reverse=True)
    n_top = max(1, int(len(all_kept) * KEEP_FRAC))
    topk = all_kept[:n_top]
    topk_keys = {(d["h5_path"], d["group"]) for d in topk}

    # Strict fallback (small)
    fallback_all = [d for d in all_kept if is_fallback_candidate_strict(d) and (d["h5_path"], d["group"]) not in topk_keys]

    # cap fallback size to keep it "small"
    max_fb = max(1, int(len(all_kept) * FALLBACK_MAX_FRAC))
    if len(fallback_all) > max_fb:
        # choose the most "structured" among fallback by score (still based on low-mask score)
        fallback_all.sort(key=lambda d: d["score"], reverse=True)
        fallback = fallback_all[:max_fb]
    else:
        fallback = fallback_all

    fallback_keys = {(d["h5_path"], d["group"]) for d in fallback}

    # Union
    cand_map = {}
    for d in topk:
        cand_map[(d["h5_path"], d["group"])] = d
    for d in fallback:
        cand_map[(d["h5_path"], d["group"])] = d

    candidates = list(cand_map.values())
    candidates.sort(key=lambda d: d["score"], reverse=True)

    print("[INFO] TopK:", len(topk), f"(KEEP_FRAC={KEEP_FRAC})")
    print("[INFO] Fallback strict (raw):", len(fallback_all))
    print("[INFO] Fallback strict (capped):", len(fallback), f"(cap={max_fb}, FALLBACK_MAX_FRAC={FALLBACK_MAX_FRAC})")
    print("[INFO] Union candidates:", len(candidates))

    # Write merged H5
    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)
    if os.path.exists(OUT_H5):
        os.remove(OUT_H5)

    print("[INFO] Writing merged candidates to:", OUT_H5)

    with h5py.File(OUT_H5, "w") as fout:
        # file-level attrs (summary)
        fout.attrs["source_h5_glob"] = H5_GLOB
        fout.attrs["num_input_h5"] = len(h5_list)
        fout.attrs["total_groups_scanned"] = total_groups
        fout.attrs["removed_bad_partial"] = bad_partial
        fout.attrs["removed_bad_noise"] = bad_noise
        fout.attrs["remaining_after_stage0"] = len(all_kept)

        fout.attrs["keep_fraction_topk"] = KEEP_FRAC
        fout.attrs["fallback_max_frac"] = FALLBACK_MAX_FRAC
        fout.attrs["num_topk"] = len(topk)
        fout.attrs["num_fallback_raw"] = len(fallback_all)
        fout.attrs["num_fallback_capped"] = len(fallback)
        fout.attrs["num_candidates_union"] = len(candidates)

        fout.attrs["noise_peaksnr_thr"] = NOISE_PEAKSNR_THR
        fout.attrs["bad_fill_frac_thr"] = BAD_FILL_FRAC_THR
        fout.attrs["bad_bbox_frac_thr"] = BAD_BBOX_FRAC_THR

        fout.attrs["nsigma_shape_lo"] = NSIGMA_SHAPE
        fout.attrs["nsigma_fallback_hi"] = NSIGMA_FALLBACK
        fout.attrs["fallback_peaksnr_thr"] = FALLBACK_PEAKSNR_THR
        fout.attrs["fallback_nend_thr"] = FALLBACK_NEND_THR
        fout.attrs["fallback_sklen_thr"] = FALLBACK_SKLEN_THR
        fout.attrs["fallback_area_hi_thr"] = FALLBACK_AREA_HI_THR

        # Copy groups
        for i, item in enumerate(tqdm(candidates, desc="Write candidates")):
            src_h5 = item["h5_path"]
            src_gn = item["group"]
            key = (src_h5, src_gn)

            gnew = fout.create_group(f"cand_{i:06d}")

            # Traceback + selection reason
            gnew.attrs["origin_h5"] = src_h5
            gnew.attrs["origin_group"] = src_gn

            gnew.attrs["score"] = float(item["score"])
            gnew.attrs["peak_snr"] = float(item["peak_snr"])

            gnew.attrs["area_lo"] = float(item["area"])
            gnew.attrs["elong_lo"] = float(item["elong"])
            gnew.attrs["sk_len_lo"] = float(item["sk_len"])
            gnew.attrs["n_end_lo"] = float(item["n_end"])

            gnew.attrs["area_hi"] = float(item["area_hi"])
            gnew.attrs["sk_len_hi"] = float(item["sk_len_hi"])
            gnew.attrs["n_end_hi"] = float(item["n_end_hi"])

            gnew.attrs["selected_by_topk"] = int(key in topk_keys)
            gnew.attrs["selected_by_fallback"] = int(key in fallback_keys)

            with h5py.File(src_h5, "r") as fin:
                gsrc = fin[src_gn]

                # Keep all original group attrs + datasets
                copy_all_attrs(gsrc, gnew)
                copy_all_datasets(gsrc, gnew)

                # store input-file attrs with prefix (avoid collisions)
                for k in fin.attrs.keys():
                    try:
                        gnew.attrs[f"origin_file_{k}"] = fin.attrs[k]
                    except Exception:
                        gnew.attrs[f"origin_file_{k}"] = str(fin.attrs[k])

    print("[DONE] Saved:", OUT_H5)


if __name__ == "__main__":
    main()
