#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
findhost_new_catalog.py

Catalog-driven host finding:
- Use catalog RA/DEC as the primary index
- Match radio/WISE cutouts by WCS footprint
- Reproject radio to the WISE grid
- Build a robust main radio mask anchored near the catalog source center
- Search the host near the source center, constrained by the radio mask
- Save Figure-2-style overlays and a host candidate CSV

Compared with the label-driven version:
- no label files are required
- the source center comes from the catalog RA/DEC
- suitable for batch host finding on the whole BT catalog
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from reproject import reproject_interp
from scipy import ndimage as ndi


# =======================
# Default paths
# =======================
DEFAULT_CATALOG = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog.fits"
DEFAULT_RADIODIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_300/"
DEFAULT_WISEDIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/WISE_benthost/"
DEFAULT_OUTDIR = "/shared/main/caojie/meerkat/host_wise_out_catalog"


# =======================
# Plot style
# =======================
WISE_CMAP = "gray"
DEFAULT_WISE_GAMMA = 1.25
DEFAULT_GRID_ALPHA = 0.18

CONTOUR_COLOR = "#ff00ff"
CONTOUR_LW = 1.2
BOUNDARY_LW = 1.6
CONTOUR_ALPHA = 0.95
CONTOUR_FACTOR = np.sqrt(2.0)
CONTOUR_LEVELS_N = 10

HOST_COLOR = "#FFD000"
HOST_MS = 16
HOST_MEW = 2.8


def squeeze_to_2d(data: np.ndarray) -> np.ndarray:
    d = np.squeeze(data)
    if d.ndim != 2:
        if d.ndim > 2:
            d = d.reshape(d.shape[-2], d.shape[-1])
        else:
            raise ValueError(f"Cannot squeeze to 2D: {data.shape} -> {d.shape}")
    return d.astype(np.float32)


def robust_rms_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def load_fits_image(path: str):
    with fits.open(path, memmap=False) as hdul:
        img = squeeze_to_2d(hdul[0].data)
        wcs = WCS(hdul[0].header).celestial
    return img, wcs


def get_center_sky(wcs: WCS, shape2d):
    h, w = shape2d
    return SkyCoord.from_pixel((w - 1) / 2.0, (h - 1) / 2.0, wcs)


def build_cutout_items(folder: str):
    files = sorted(list(Path(folder).rglob("*.fits")))
    items = []
    for fp in files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                img2d = squeeze_to_2d(hdul[0].data)
                w = WCS(hdul[0].header).celestial
                shape = img2d.shape
            c = get_center_sky(w, shape)
            items.append({"path": str(fp), "wcs": w, "shape": shape, "center": c})
        except Exception:
            continue
    return items


def match_by_footprint(target_sky: SkyCoord, items):
    if not items:
        return None, np.inf

    centers = SkyCoord([it["center"] for it in items])
    seps = target_sky.separation(centers).arcsec
    order = np.argsort(seps)

    for j in order:
        it = items[int(j)]
        w = it["wcs"]
        h, width = it["shape"]
        x, y = w.world_to_pixel(target_sky)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if (-1 <= x <= (width - 0)) and (-1 <= y <= (h - 0)):
            return it["path"], float(seps[int(j)])

    j0 = int(order[0])
    return None, float(seps[j0])


def estimate_bg_sigma_edge(img: np.ndarray, border_frac: float = 0.25):
    h, w = img.shape
    b = max(1, int(min(h, w) * border_frac))
    bg = np.zeros_like(img, dtype=bool)
    bg[:b, :] = True
    bg[-b:, :] = True
    bg[:, :b] = True
    bg[:, -b:] = True

    x = img[bg]
    x = x[np.isfinite(x)]
    if x.size < 50:
        bg_med = float(np.nanmedian(img))
        res = img - bg_med
        res = res[np.isfinite(res)]
        return bg_med, float(robust_rms_mad(res))

    bg_med = float(np.median(x))
    res = x - bg_med
    neg = res[res < 0]

    if neg.size >= 30:
        sigma = 1.4826 * np.median(np.abs(neg))
    else:
        sigma = 1.4826 * np.median(np.abs(res))

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(res))

    return bg_med, float(sigma)


def build_main_radio_mask_snr(
    img2d: np.ndarray,
    bg_med: float,
    sigma: float,
    *,
    anchor_xy,
    low_sigma: float = 2.0,
    seed_r_pix: float = 35.0,
    min_area_pix: int = 12,
    max_area_frac: float = 0.25,
    close_iter: int = 1,
):
    img = np.asarray(img2d, float)
    h, w = img.shape
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0")

    thr = bg_med + float(low_sigma) * sigma
    m_low = np.isfinite(img) & (img > thr)
    m_low = ndi.binary_closing(m_low, iterations=int(close_iter))

    lab, nlab = ndi.label(m_low)
    if nlab == 0:
        peak_val = float(np.nanmax(img))
        peak_snr = (peak_val - bg_med) / sigma if np.isfinite(peak_val) else np.nan
        return np.zeros_like(m_low, bool), peak_val, float(peak_snr), float(low_sigma)

    areas = np.bincount(lab.ravel())[1:]
    valid_ids = np.where(areas >= int(min_area_pix))[0] + 1
    max_area = int(float(max_area_frac) * h * w)
    valid_ids = np.array([lid for lid in valid_ids if areas[lid - 1] <= max_area], dtype=int)

    if valid_ids.size == 0:
        peak_val = float(np.nanmax(img))
        peak_snr = (peak_val - bg_med) / sigma if np.isfinite(peak_val) else np.nan
        return np.zeros_like(m_low, bool), peak_val, float(peak_snr), float(low_sigma)

    ax, ay = anchor_xy
    chosen_id = None
    if np.isfinite(ax) and np.isfinite(ay):
        yy, xx = np.ogrid[:h, :w]
        seed = (xx - ax) ** 2 + (yy - ay) ** 2 <= float(seed_r_pix) ** 2
        seed_hits = seed & m_low
        if np.any(seed_hits):
            hit_ids = np.unique(lab[seed_hits])
            hit_ids = hit_ids[hit_ids > 0]
            hit_ids = np.array([lid for lid in hit_ids if lid in set(valid_ids.tolist())], dtype=int)
            if hit_ids.size > 0:
                best, best_peak = None, -np.inf
                for lid in hit_ids:
                    v = np.nanmax(img[(lab == lid) & seed])
                    if v > best_peak:
                        best_peak = v
                        best = int(lid)
                chosen_id = best

    if chosen_id is None:
        best, best_d = None, np.inf
        for lid in valid_ids:
            ys, xs = np.where(lab == lid)
            if xs.size == 0:
                continue
            cx = xs.mean()
            cy = ys.mean()
            d = np.hypot(cx - ax, cy - ay)
            if d < best_d:
                best_d = d
                best = int(lid)
        chosen_id = best

    m_main = (lab == int(chosen_id))
    peak_val = float(np.nanmax(img[m_main])) if np.any(m_main) else float(np.nanmax(img))
    peak_snr = (peak_val - bg_med) / sigma if np.isfinite(peak_val) else np.nan
    return m_main.astype(bool), peak_val, float(peak_snr), float(low_sigma)


def build_contour_levels(peak_amp: float, peak_snr: float, sigma: float):
    start_sigma = 3.0
    if np.isfinite(peak_snr):
        start_sigma = float(np.clip(0.60 * peak_snr, 1.5, 3.0))

    levels = []
    if np.isfinite(peak_amp) and peak_amp > 0:
        lv = start_sigma * sigma
        for _ in range(CONTOUR_LEVELS_N):
            if lv >= 0.98 * peak_amp:
                break
            levels.append(lv)
            lv *= CONTOUR_FACTOR

    if len(levels) == 0:
        if np.isfinite(peak_amp) and peak_amp > 0:
            levels = [0.5 * peak_amp, 0.75 * peak_amp]
        else:
            levels = [start_sigma * sigma]

    return levels, start_sigma


def find_host_near_source(
    wise_img: np.ndarray,
    wise_wcs: WCS,
    src_sky: SkyCoord,
    *,
    radio_mask: np.ndarray | None = None,
    radio_img: np.ndarray | None = None,
    host_r_max_pix: float = 22.0,
    smooth_sigma_pix: float = 1.2,
    centroid_halfsize: int = 3,
    mask_dilate_pix: int = 2,
    radio_peak_weight: float = 0.25,
    dist_src_weight: float = 0.30,
    boundary_weight: float = 0.35,
    boundary_tau_pix: float = 2.5,
    peak_tau_pix: float = 18.0,
):
    img = np.asarray(wise_img, float)
    h, w = img.shape
    sm = ndi.gaussian_filter(img, float(smooth_sigma_pix)) if smooth_sigma_pix > 0 else img

    sx, sy = wise_wcs.world_to_pixel(src_sky)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return None

    yy, xx = np.ogrid[:h, :w]
    dist_src = np.hypot(xx - sx, yy - sy)
    near_src = dist_src <= float(host_r_max_pix)
    finite = np.isfinite(sm)

    rm_dil = None
    rm_in = None
    dist_inside = None
    if radio_mask is not None:
        rm0 = np.asarray(radio_mask, bool)
        if rm0.shape != (h, w):
            raise ValueError(f"radio_mask shape {rm0.shape} != wise_img shape {(h, w)}")
        rm_dil = ndi.binary_dilation(rm0, iterations=int(mask_dilate_pix))
        rm_in = ndi.binary_erosion(rm0, iterations=1)
        if not np.any(rm_in):
            rm_in = rm0
        dist_inside = ndi.distance_transform_edt(rm_dil)

    radio_peak_x = radio_peak_y = None
    if (radio_img is not None) and (rm_dil is not None):
        rimg = np.asarray(radio_img, float)
        rr = near_src & rm_dil
        if np.any(rr):
            tmp = np.where(rr, rimg, -np.inf)
            if np.isfinite(tmp).any():
                py, px = np.unravel_index(int(np.nanargmax(tmp)), tmp.shape)
                radio_peak_y, radio_peak_x = int(py), int(px)

    if rm_in is not None and np.any(near_src & rm_in & finite):
        search = near_src & rm_in & finite
        stage = "src_near+mask_interior"
    elif rm_dil is not None and np.any(near_src & rm_dil & finite):
        search = near_src & rm_dil & finite
        stage = "src_near+mask_dilated"
    else:
        search = near_src & finite
        stage = "src_near_only"

    local_max = (sm == ndi.maximum_filter(sm, size=3, mode="nearest"))
    cand_mask = search & local_max
    if np.sum(cand_mask) < 3:
        cand_mask = search
        stage += "_fallbackNoLocalMax"

    vals = sm[cand_mask]
    if vals.size == 0 or not np.isfinite(vals).any():
        return None

    v_med = np.nanmedian(vals)
    v_mad = np.nanmedian(np.abs(vals - v_med))
    scale = v_mad * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(vals)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

    z = (sm - v_med) / scale
    dpen_src = (dist_src / max(1.0, float(host_r_max_pix))) * float(dist_src_weight)

    bpen = 0.0
    if dist_inside is not None:
        tau = max(0.5, float(boundary_tau_pix))
        bpen = (1.0 - np.clip(dist_inside / tau, 0.0, 1.0)) * float(boundary_weight)

    rbonus = 0.0
    if (radio_peak_x is not None) and (radio_peak_y is not None):
        dpk = np.hypot(xx - radio_peak_x, yy - radio_peak_y)
        tau = max(1.0, float(peak_tau_pix))
        rbonus = np.clip(1.0 - dpk / tau, 0.0, 1.0) * float(radio_peak_weight)

    score = z - dpen_src - bpen + rbonus
    score2 = np.where(cand_mask, score, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(score2)), score2.shape)

    hs = int(centroid_halfsize)
    x0, x1 = max(0, ix - hs), min(w, ix + hs + 1)
    y0, y1 = max(0, iy - hs), min(h, iy + hs + 1)
    patch = sm[y0:y1, x0:x1]
    allow_patch = cand_mask[y0:y1, x0:x1]

    if patch.size == 0 or not np.any(allow_patch):
        cx2, cy2 = float(ix), float(iy)
    else:
        pmin = np.nanmin(patch[allow_patch])
        p = np.clip(patch - pmin, 0, None) * allow_patch
        if np.nansum(p) <= 0:
            cx2, cy2 = float(ix), float(iy)
        else:
            yy2, xx2 = np.mgrid[y0:y1, x0:x1]
            cx2 = float(np.nansum(xx2 * p) / np.nansum(p))
            cy2 = float(np.nansum(yy2 * p) / np.nansum(p))

    xi, yi = int(round(cx2)), int(round(cy2))
    if not (0 <= xi < w and 0 <= yi < h):
        cx2, cy2 = float(ix), float(iy)
        stage += "_centroidClipped"
    else:
        if not near_src[yi, xi]:
            cx2, cy2 = float(ix), float(iy)
            stage += "_centroidClipped"
        if rm_dil is not None and not rm_dil[yi, xi]:
            cx2, cy2 = float(ix), float(iy)
            stage += "_centroidClipped"

    b = np.nanmedian(img[~near_src]) if np.any(~near_src) else np.nanmedian(img)
    s = np.nanstd(img[~near_src]) if np.any(~near_src) else np.nanstd(img)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    host_snr = (img[int(round(cy2)), int(round(cx2))] - b) / s

    host_sky = SkyCoord.from_pixel(cx2, cy2, wise_wcs)
    return host_sky, float(host_snr), stage


def render_fig2(
    out_png,
    wise_img,
    wise_wcs,
    radio_on_wise,
    bg_med_radio,
    m_main,
    contour_levels,
    src_sky,
    host_sky,
    title,
    wise_gamma,
    grid_alpha,
):
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(projection=wise_wcs)

    norm = ImageNormalize(wise_img, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(wise_img, origin="lower", cmap=WISE_CMAP, norm=norm, interpolation="nearest")

    if wise_gamma and wise_gamma != 1.0:
        alpha_black = np.clip((wise_gamma - 1.0) * 0.18, 0.0, 0.35)
        ax.imshow(
            np.zeros_like(wise_img),
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=alpha_black,
            interpolation="nearest",
        )

    radio_main = np.where(m_main, radio_on_wise - bg_med_radio, np.nan)
    ax.contour(radio_main, levels=contour_levels, colors=CONTOUR_COLOR, linewidths=CONTOUR_LW, alpha=CONTOUR_ALPHA)
    ax.contour(m_main.astype(float), levels=[0.5], colors=CONTOUR_COLOR, linewidths=BOUNDARY_LW, alpha=CONTOUR_ALPHA)

    ax.scatter(
        src_sky.ra.deg,
        src_sky.dec.deg,
        transform=ax.get_transform("world"),
        s=36,
        marker="+",
        c="#00c8ff",
        linewidths=1.8,
        zorder=11,
    )

    if host_sky is not None:
        ax.scatter(
            host_sky.ra.deg,
            host_sky.dec.deg,
            transform=ax.get_transform("world"),
            s=HOST_MS * 12,
            marker="x",
            c=HOST_COLOR,
            linewidths=HOST_MEW,
            zorder=12,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.grid(color="white", alpha=float(grid_alpha), linestyle="--", linewidth=0.6)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def pick_col(tab: Table, candidates):
    lower_map = {c.lower(): c for c in tab.colnames}
    for cand in candidates:
        if cand in tab.colnames:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=DEFAULT_CATALOG)
    ap.add_argument("--radio_dir", default=DEFAULT_RADIODIR)
    ap.add_argument("--wise_dir", default=DEFAULT_WISEDIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUTDIR)
    ap.add_argument("--max_rows", type=int, default=0, help="0 means process all rows")
    ap.add_argument("--seed_r_pix", type=float, default=35.0)
    ap.add_argument("--host_r_max_pix", type=float, default=22.0)
    ap.add_argument("--smooth_sigma_pix", type=float, default=1.2)
    ap.add_argument("--centroid_halfsize", type=int, default=3)
    ap.add_argument("--wise_gamma", type=float, default=DEFAULT_WISE_GAMMA)
    ap.add_argument("--grid_alpha", type=float, default=DEFAULT_GRID_ALPHA)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "RUN_TAG_fig2_bycatalog.txt").write_text(f"run_tag={run_tag}\n")

    tab = Table.read(args.catalog)
    ra_col = pick_col(tab, ["RA", "ra", "RA_deg", "ra_deg", "RAJ2000", "raj2000"])
    dec_col = pick_col(tab, ["DEC", "dec", "DEC_deg", "dec_deg", "DEJ2000", "dej2000"])
    id_col = pick_col(tab, ["Source_id", "source_id", "ID", "id"])
    name_col = pick_col(tab, ["Source_Name", "source_name", "name", "Name"])
    if ra_col is None or dec_col is None:
        raise RuntimeError("Cannot find RA/DEC columns in catalog")

    wise_items = build_cutout_items(args.wise_dir)
    if len(wise_items) == 0:
        raise RuntimeError(f"No WISE FITS found in {args.wise_dir}")

    radio_items = build_cutout_items(args.radio_dir)
    if len(radio_items) == 0:
        raise RuntimeError(f"No RADIO FITS found in {args.radio_dir}")

    n_total = len(tab)
    if args.max_rows and args.max_rows > 0:
        n_total = min(n_total, int(args.max_rows))

    rows = []
    n_ok = n_no_radio = n_no_wise = n_err = 0

    for idx in range(n_total):
        row = tab[idx]
        ra = float(row[ra_col])
        dec = float(row[dec_col])
        src_sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

        sid = str(row[id_col]) if id_col is not None else f"row{idx:06d}"
        sname = str(row[name_col]) if name_col is not None else sid

        try:
            radio_path, best_rsep = match_by_footprint(src_sky, radio_items)
            if radio_path is None:
                n_no_radio += 1
                rows.append({
                    "row_idx": idx,
                    "source_id": sid,
                    "source_name": sname,
                    "status": "missing_radio",
                    "RA": ra,
                    "DEC": dec,
                    "best_radio_sep_arcsec": float(best_rsep),
                })
                continue

            wise_path, best_wsep = match_by_footprint(src_sky, wise_items)
            if wise_path is None:
                n_no_wise += 1
                rows.append({
                    "row_idx": idx,
                    "source_id": sid,
                    "source_name": sname,
                    "status": "no_wise_match",
                    "RA": ra,
                    "DEC": dec,
                    "best_radio_sep_arcsec": float(best_rsep),
                    "best_wise_sep_arcsec": float(best_wsep),
                    "radio_path": radio_path,
                })
                continue

            wise_img, wise_wcs = load_fits_image(wise_path)
            radio_img, radio_wcs = load_fits_image(radio_path)
            radio_on_wise, _ = reproject_interp((radio_img, radio_wcs), wise_wcs, shape_out=wise_img.shape)

            bg_med, sigma = estimate_bg_sigma_edge(radio_on_wise, border_frac=0.25)
            if not np.isfinite(sigma) or sigma <= 0:
                raise RuntimeError("sigma estimate failed")

            sx, sy = wise_wcs.world_to_pixel(src_sky)
            m_main, peak_val, peak_snr, low_sigma_eff = build_main_radio_mask_snr(
                radio_on_wise,
                bg_med,
                sigma,
                anchor_xy=(sx, sy),
                low_sigma=2.0,
                seed_r_pix=float(args.seed_r_pix),
                max_area_frac=0.25,
            )

            peak_amp = peak_val - bg_med
            levels, start_sigma = build_contour_levels(peak_amp, peak_snr, sigma)

            host = find_host_near_source(
                wise_img,
                wise_wcs,
                src_sky,
                radio_mask=m_main,
                radio_img=radio_on_wise,
                host_r_max_pix=float(args.host_r_max_pix),
                smooth_sigma_pix=float(args.smooth_sigma_pix),
                centroid_halfsize=int(args.centroid_halfsize),
            )

            if host is None:
                host_sky, host_snr, stage = None, np.nan, "none"
            else:
                host_sky, host_snr, stage = host

            out_png = out_dir / "overlays" / f"{sid}.png"
            title = (
                f"{sid} | bg={bg_med:.2e} | sigma={sigma:.2e} | peakSNR={peak_snr:.2f} | "
                f"lowSigma={low_sigma_eff:.2f} | startSigma={start_sigma:.2f} | "
                f"stage={stage} | wiseSep={best_wsep:.1f}\" | radioSep={best_rsep:.1f}\" | run={run_tag}"
            )

            render_fig2(
                out_png=str(out_png),
                wise_img=wise_img,
                wise_wcs=wise_wcs,
                radio_on_wise=radio_on_wise,
                bg_med_radio=bg_med,
                m_main=m_main,
                contour_levels=levels,
                src_sky=src_sky,
                host_sky=host_sky,
                title=title,
                wise_gamma=float(args.wise_gamma),
                grid_alpha=float(args.grid_alpha),
            )

            n_ok += 1
            rows.append({
                "row_idx": idx,
                "source_id": sid,
                "source_name": sname,
                "status": "ok",
                "RA": ra,
                "DEC": dec,
                "radio_path": radio_path,
                "wise_path": wise_path,
                "best_radio_sep_arcsec": float(best_rsep),
                "best_wise_sep_arcsec": float(best_wsep),
                "bg_med_radio": float(bg_med),
                "sigma_radio": float(sigma),
                "peak_val": float(peak_val),
                "peak_amp": float(peak_amp),
                "peak_snr": float(peak_snr) if np.isfinite(peak_snr) else np.nan,
                "low_sigma_eff": float(low_sigma_eff) if np.isfinite(low_sigma_eff) else np.nan,
                "start_sigma": float(start_sigma) if np.isfinite(start_sigma) else np.nan,
                "mask_pixels": int(np.sum(m_main)),
                "host_ra_deg": float(host_sky.ra.deg) if host_sky else np.nan,
                "host_dec_deg": float(host_sky.dec.deg) if host_sky else np.nan,
                "host_snr": float(host_snr) if np.isfinite(host_snr) else np.nan,
                "stage": stage,
                "overlay_png": str(out_png),
            })

        except Exception as e:
            n_err += 1
            rows.append({
                "row_idx": idx,
                "source_id": sid,
                "source_name": sname,
                "status": f"error:{type(e).__name__}",
                "RA": ra,
                "DEC": dec,
                "error_msg": str(e),
            })

    out_csv = out_dir / "host_candidates_fig2_bycatalog.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[RUN] run_tag={run_tag}")
    print(f"[RUN] rows={n_total} | ok={n_ok} | missing_radio={n_no_radio} | no_wise_match={n_no_wise} | errors={n_err}")
    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Overlays: {out_dir / 'overlays'}")
    print(f"[OK] Tag file: {out_dir / 'RUN_TAG_fig2_bycatalog.txt'}")


if __name__ == "__main__":
    main()
