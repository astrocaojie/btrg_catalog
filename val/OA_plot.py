#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OA_plot.py (Fixed Version)

Batch:
1) Parse CRTF labels (C + >=2 T)
2) Match label(C) -> nearest FITS cutout center (WCS center)
3) Match label(C) -> nearest catalog row (by sky separation)
4) Compute OA by paper formula
5) Print OA_calc vs OA_cat (ONLY print)
6) Save paper-style schematic PNG:
   - adaptive zoom window (source fills canvas)
   - connected-component mask (>=2σ) outline (magenta dotted)
   - centerline + rays + OA arc
   - OA text auto placed away from source

Paths:
- FITS_DIR:  /home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_300/
- LABEL_DIR: /home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_500/oa_labels/
- CAT_PATH:  /home/caojie/work/Galaxy-Morphology/bent_catalog/BT_final_catalog.fits
- OUT_DIR:   /shared/main/caojie/meerkat/oa_out/
"""

import os
import re
import glob
import math
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge
from scipy import ndimage as ndi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec



# =======================
# Paths (as you provided)
# =======================
FITS_DIR  = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_300/"
LABEL_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_500/oa_labels/"
CAT_PATH  = "/home/caojie/work/Galaxy-Morphology/bent_catalog/BT_final_catalog.fits"
OUT_DIR   = "/shared/main/caojie/meerkat/oa_out/"


# =======================
# Matching thresholds
# =======================
MAX_SEP_FITS_LABEL = 3.0 * u.arcmin   # label(C) to cutout center
MAX_SEP_LABEL_CAT  = 30.0 * u.arcsec  # label(C) to catalog row


# =======================
# Plot style knobs
# =======================
CMAP = "gray"              # ✅ 修复: 从 "garys" 改为 "gray"
THRESH_SIGMA = 3.5         # mask threshold
PAD_FRAC = 0.18
MIN_PAD_PIX = 20
MASK_LS = (0, (1.5, 2.5))
MASK_COLOR = (0.85, 0.0, 0.85)         # magenta
CENTERLINE_COLOR = (0.0, 0.75, 0.75)   # cyan/teal
RAY_COLOR = (0.05, 0.05, 0.05)         # near black
POINT_COLOR = "#2b83ba"
FIGSIZE = (6.2, 6.2)
DPI = 220

# =======================
# Robust scalar helpers  ✅关键修复点
# =======================
def _f(x):
    """Convert scalar/Quantity/ndarray to python float."""
    if hasattr(x, "to_value"):
        try:
            x = x.to_value()
        except Exception:
            pass
    x = np.asarray(x)
    return float(x.reshape(-1)[0])

def _i(x):
    """Round-to-nearest int safely for ndarray/scalar."""
    return int(np.rint(_f(x)))


# =======================
# Helpers
# =======================
def pick_col(tbl, candidates):
    lower_map = {c.lower(): c for c in tbl.colnames}
    for cand in candidates:
        if cand in tbl.colnames:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# =======================
# Parse CRTF labels
# =======================
def _normalize_dec_token(tok: str) -> str:
    tok = tok.strip()
    if "deg" in tok.lower() or ":" in tok:
        return tok
    if re.match(r"^[\+\-]?\d+\.\d+\.\d+(\.\d+)?$", tok):
        parts = tok.split(".")
        if len(parts) >= 3:
            d, m, s = parts[0], parts[1], ".".join(parts[2:])
            try:
                sign = "-" if d.strip().startswith("-") else "+"
                d_abs = d.strip().lstrip("+-")
                d_abs_i = int(d_abs)
                d_new = f"{sign}{d_abs_i}"
            except Exception:
                d_new = d
            return f"{d_new}:{m}:{s}"
    return tok

def _normalize_ra_token(tok: str) -> str:
    tok = tok.strip()
    if "deg" in tok.lower() or ":" in tok:
        return tok
    if re.match(r"^\d+\.\d+\.\d+(\.\d+)?$", tok):
        parts = tok.split(".")
        if len(parts) >= 3:
            h, m, s = parts[0], parts[1], ".".join(parts[2:])
            return f"{h}:{m}:{s}"
    return tok

def parse_crtf_symbols(label_path: str):
    """
    Return (C, Ts)
      - C: SkyCoord
      - Ts: list[SkyCoord], length >=2 expected
    """
    C = None
    Ts = []
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("symbol"):
                continue

            m_label = re.search(r'label="([^"]+)"', line)
            if not m_label:
                continue
            lab = m_label.group(1).strip().upper()

            m_xy = re.search(r"symbol\s+\[\[\s*([^,]+)\s*,\s*([^\]]+)\]\s*,", line)
            if not m_xy:
                continue

            ra_tok = m_xy.group(1).strip()
            dec_tok = m_xy.group(2).strip()

            if "deg" in ra_tok.lower() or "deg" in dec_tok.lower():
                ra_val = float(ra_tok.lower().replace("deg", "").strip())
                dec_val = float(dec_tok.lower().replace("deg", "").strip())
                coord = SkyCoord(ra=ra_val*u.deg, dec=dec_val*u.deg, frame="icrs")
            else:
                ra_s = _normalize_ra_token(ra_tok)
                dec_s = _normalize_dec_token(dec_tok)
                coord = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")

            if lab == "C":
                C = coord
            elif lab == "T":
                Ts.append(coord)

    return C, Ts


# =======================
# OA computation (paper)
# =======================
def compute_oa_deg(C: SkyCoord, T1: SkyCoord, T2: SkyCoord) -> float:
    L_CT1 = C.separation(T1).to(u.rad).value
    L_CT2 = C.separation(T2).to(u.rad).value
    L_T1T2 = T1.separation(T2).to(u.rad).value
    denom = 2.0 * L_CT1 * L_CT2
    if denom <= 0:
        return float("nan")
    cosv = (L_CT1**2 + L_CT2**2 - L_T1T2**2) / denom
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return math.degrees(math.acos(cosv))


# =======================
# Mask + robust background
# =======================
def robust_sigma_border(data, border_frac=0.18):
    ny, nx = data.shape
    bx = int(nx * border_frac)
    by = int(ny * border_frac)
    m = np.zeros_like(data, dtype=bool)
    m[:by, :] = True
    m[-by:, :] = True
    m[:, :bx] = True
    m[:, -bx:] = True
    v = data[m & np.isfinite(data)]
    if v.size < 50:
        v = data[np.isfinite(data)]
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(v)
    return med, sig


def build_connected_mask(
    data, wcs, C_world,
    high_sigma=3.5, low_sigma=2.0,
    close_iters=2, open_iters=1, erode_iters=1,
    min_pix=40
):
    """
    更“论文风格”的紧致连通 mask：
    1) 先用 high_sigma 得到核心
    2) 再用 low_sigma 做连通域生长（只保留与核心相连的部分）
    3) closing 让尾巴/桥连起来，opening 去噪点，最后轻微 erosion 收边界
    4) 只保留包含 C 的主连通体（找不到则 fallback 到包含峰值的主块）
    """
    bkg, sig = robust_sigma_border(data)
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros_like(data, dtype=bool), bkg, sig

    thr_hi = bkg + high_sigma * sig
    thr_lo = bkg + low_sigma * sig

    finite = np.isfinite(data)
    core = finite & (data >= thr_hi)
    loose = finite & (data >= thr_lo)

    if core.sum() == 0:
        return np.zeros_like(data, dtype=bool), bkg, sig

    # --- 形态学：先 closing 连起来，再 opening 去噪点 ---
    st = ndi.generate_binary_structure(2, 1)
    if close_iters > 0:
        loose = ndi.binary_closing(loose, structure=st, iterations=close_iters)
        core  = ndi.binary_closing(core,  structure=st, iterations=max(1, close_iters-1))
    if open_iters > 0:
        loose = ndi.binary_opening(loose, structure=st, iterations=open_iters)

    # --- 选主块：优先包含 C 的 loose 连通体 ---
    lab, nlab = ndi.label(loose)
    if nlab == 0:
        return np.zeros_like(data, dtype=bool), bkg, sig

    Cx, Cy = wcs.world_to_pixel(C_world)
    ix, iy = _i(Cx), _i(Cy)

    chosen = 0
    if 0 <= ix < data.shape[1] and 0 <= iy < data.shape[0]:
        chosen = int(lab[iy, ix])

    if chosen == 0:
        # fallback：在 core 内找峰值所在的连通块（再映射到 loose）
        yy, xx = np.where(core)
        if yy.size == 0:
            # 再 fallback：loose 内峰值
            yy, xx = np.where(loose)
            if yy.size == 0:
                return np.zeros_like(data, dtype=bool), bkg, sig
        k = int(np.argmax(data[yy, xx]))
        chosen = int(lab[yy[k], xx[k]])

    mask = (lab == chosen)

    # --- 只保留与 core 相连的部分（避免 loose 外扩） ---
    core_in_mask = core & mask
    if core_in_mask.any():
        # 从 core_in_mask 做 flood-fill，只在 mask 里走
        seed_lab, _ = ndi.label(core_in_mask)
        # 取最大的 core seed
        sizes = np.bincount(seed_lab.ravel())
        sizes[0] = 0
        seed_id = int(np.argmax(sizes))
        seed = (seed_lab == seed_id)

        # iterative dilation constrained within mask
        grow = seed.copy()
        for _ in range(400):  # 足够大，300x300 内很快
            dil = ndi.binary_dilation(grow, structure=st, iterations=1)
            new = dil & mask
            if new.sum() == grow.sum():
                break
            grow = new
        mask = grow

    # --- 填洞 + 轻微 erosion 收边界（让“松”的边界变紧） ---
    mask = ndi.binary_fill_holes(mask)
    if erode_iters > 0:
        mask = ndi.binary_erosion(mask, structure=st, iterations=erode_iters)

    # --- 去掉太小的碎块（保险） ---
    lab2, n2 = ndi.label(mask)
    if n2 > 1:
        sizes = np.bincount(lab2.ravel())
        keep = np.zeros_like(sizes, dtype=bool)
        keep[sizes >= min_pix] = True
        keep[0] = False
        mask = keep[lab2]

    return mask.astype(bool), bkg, sig



def bbox_from_mask_or_points(mask, pts_xy, shape, pad_frac=0.18, min_pad=20):
    ny, nx = shape
    if mask is not None and mask.any():
        ys, xs = np.where(mask)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    else:
        xs = np.array([_f(p[0]) for p in pts_xy], dtype=float)
        ys = np.array([_f(p[1]) for p in pts_xy], dtype=float)
        xmin, xmax = np.nanmin(xs), np.nanmax(xs)
        ymin, ymax = np.nanmin(ys), np.nanmax(ys)

    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)
    pad = max(min_pad, int(pad_frac * max(w, h)))

    xmin = max(0, int(math.floor(xmin - pad)))
    xmax = min(nx - 1, int(math.ceil(xmax + pad)))
    ymin = max(0, int(math.floor(ymin - pad)))
    ymax = min(ny - 1, int(math.ceil(ymax + pad)))
    return xmin, xmax, ymin, ymax


def choose_text_position(data, mask, Cx, Cy, r_pix, n_try=18):
    ny, nx = data.shape
    Cx, Cy = _f(Cx), _f(Cy)
    best = None
    best_score = None
    for k in range(n_try):
        ang = 2 * math.pi * k / n_try
        x = Cx + r_pix * math.cos(ang)
        y = Cy + r_pix * math.sin(ang)
        ix, iy = _i(x), _i(y)  # ✅修复 round(ndarray)
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        if mask is not None and mask.any() and mask[iy, ix]:
            continue
        val = data[iy, ix]
        if not np.isfinite(val):
            continue
        score = abs(val)
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y)
    return best if best is not None else (Cx + r_pix, Cy + r_pix)


# =======================
# Plotting (paper style)
# =======================
def plot_schematic(fits_path, C, T1, T2, oa_deg, out_png):
    """
    Paper-style OA schematic with:
      - tight, connected mask (constrained around a CURVED spine derived from emission)
      - smoothed mask outline
      - WCS axis labels not clipped
      - colorbar same height as main panel (no stray 'X' label)
      - OA text forced outside (dilated) mask to avoid covering emission

    NOTE: Plot style is unchanged; ONLY the mask constraint logic is updated so the mask
          follows the real bent emission instead of straight "tube" segments.
    """
    import os, math, heapq
    import numpy as np
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import ndimage as ndi

    # ---------------- load ----------------
    with fits.open(fits_path, memmap=False) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
        w = WCS(hdul[0].header).celestial

    ny, nx = data.shape

    # ---------------- world -> pixel ----------------
    Cx, Cy = w.world_to_pixel(C);    Cx, Cy = _f(Cx), _f(Cy)
    T1x, T1y = w.world_to_pixel(T1); T1x, T1y = _f(T1x), _f(T1y)
    T2x, T2y = w.world_to_pixel(T2); T2x, T2y = _f(T2x), _f(T2y)

    def _clip_xy(x, y):
        x = float(np.clip(x, 0, nx - 1))
        y = float(np.clip(y, 0, ny - 1))
        return x, y

    # ---------------- mask (tight + connected) ----------------
    mask0, bkg, sig = build_connected_mask(
        data, w, C,
        high_sigma=max(THRESH_SIGMA, 3.8),
        low_sigma=2.2,
        close_iters=1,
        open_iters=1,
        erode_iters=2,
        min_pix=40
    )
    mask0 = (mask0.astype(bool) if mask0 is not None else np.zeros_like(data, bool))

    # =========================================================
    # ✅ ONLY CHANGE: constrain mask around a CURVED spine
    #     (instead of straight C->T1 / C->T2 distance tubes)
    # =========================================================
    st = ndi.generate_binary_structure(2, 1)

    def _nearest_true_pixel(binimg, x, y, r=70):
        iy, ix = int(round(y)), int(round(x))
        y0 = max(0, iy-r); y1 = min(ny-1, iy+r)
        x0 = max(0, ix-r); x1 = min(nx-1, ix+r)
        cut = binimg[y0:y1+1, x0:x1+1]
        ys, xs = np.where(cut)
        if xs.size == 0:
            return None
        xs = xs + x0
        ys = ys + y0
        dx = xs - x
        dy = ys - y
        k = int(np.argmin(dx*dx + dy*dy))
        return int(xs[k]), int(ys[k])

    def _weighted_path(mask_bin, start_xy, goal_xy, img):
        sx, sy = start_xy
        gx, gy = goal_xy
        if not mask_bin[sy, sx] or not mask_bin[gy, gx]:
            return None

        vals = img[mask_bin]
        if vals.size == 0:
            return None
        lo, hi = np.nanpercentile(vals, [8, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = (hi - lo) if hi > lo else 1.0

        def normI(x, y):
            v = img[y, x]
            if not np.isfinite(v):
                return 0.0
            t = (v - lo) / denom
            return float(np.clip(t, 0.0, 1.0))

        h, w_ = mask_bin.shape
        INF = 1e18
        dist = np.full((h, w_), INF, dtype=float)
        prev = np.full((h, w_, 2), -1, dtype=int)

        pq = []
        dist[sy, sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))

        nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        eps = 2e-3
        while pq:
            d, x, y = heapq.heappop(pq)
            if d != dist[y, x]:
                continue
            if x == gx and y == gy:
                break
            for dx, dy in nbrs:
                xx2, yy2 = x+dx, y+dy
                if xx2 < 0 or xx2 >= w_ or yy2 < 0 or yy2 >= h:
                    continue
                if not mask_bin[yy2, xx2]:
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                inten = normI(xx2, yy2)
                wgt = step * (1.0 / (eps + inten))
                nd = d + wgt
                if nd < dist[yy2, xx2]:
                    dist[yy2, xx2] = nd
                    prev[yy2, xx2, 0] = x
                    prev[yy2, xx2, 1] = y
                    heapq.heappush(pq, (nd, xx2, yy2))

        if not np.isfinite(dist[gy, gx]) or dist[gy, gx] >= INF/2:
            return None

        path = []
        x, y = gx, gy
        while not (x == sx and y == sy):
            path.append((x, y))
            px, py = prev[y, x]
            if px < 0:
                return None
            x, y = px, py
        path.append((sx, sy))
        path.reverse()
        return path

    d1 = float(np.hypot(T1x - Cx, T1y - Cy))
    d2 = float(np.hypot(T2x - Cx, T2y - Cy))
    dmin = max(1.0, min(d1, d2))

    # tube half-width in pixels (same scaling as before)
    tube_w = float(np.clip(0.13 * dmin, 6.0, 26.0))

    yy, xx = np.indices((ny, nx), dtype=float)

    # keep a core disk around C (same as before)
    core_r = float(np.clip(0.18 * dmin, 8.0, 30.0))
    core_disk = ((xx - Cx)**2 + (yy - Cy)**2) <= core_r*core_r

    # Build curved spine (T1->C->T2) inside mask0; fallback to piecewise-linear if needed
    spine = np.zeros((ny, nx), dtype=bool)
    if mask0 is not None and mask0.any():
        pT1 = _nearest_true_pixel(mask0, T1x, T1y, r=90)
        pC  = _nearest_true_pixel(mask0, Cx,  Cy,  r=90)
        pT2 = _nearest_true_pixel(mask0, T2x, T2y, r=90)
        if pT1 and pC and pT2:
            path1 = _weighted_path(mask0, pT1, pC, data)
            path2 = _weighted_path(mask0, pC,  pT2, data)
            if path1 and path2:
                for (x, y) in (path1 + path2[1:]):
                    if 0 <= x < nx and 0 <= y < ny:
                        spine[y, x] = True

    if not spine.any():
        t = np.linspace(0, 1, 160)
        xs1 = (T1x + (Cx - T1x)*t).astype(float)
        ys1 = (T1y + (Cy - T1y)*t).astype(float)
        xs2 = (Cx  + (T2x - Cx)*t).astype(float)
        ys2 = (Cy  + (T2y - Cy)*t).astype(float)
        xs = np.concatenate([xs1, xs2[1:]])
        ys = np.concatenate([ys1, ys2[1:]])
        xi = np.clip(np.rint(xs).astype(int), 0, nx-1)
        yi = np.clip(np.rint(ys).astype(int), 0, ny-1)
        spine[yi, xi] = True

    spine = ndi.binary_dilation(spine, structure=st, iterations=1)
    dist_to_spine = ndi.distance_transform_edt(~spine)
    tube = dist_to_spine <= tube_w

    # Constrain to curved tube + core disk
    mask = mask0 & (tube | core_disk)

    # Gentle regularization (same spirit as your old post-processing)
    if mask.any():
        mask = ndi.binary_closing(mask, structure=st, iterations=1)
        mask = ndi.binary_dilation(mask, structure=st, iterations=1)

    # keep only the component containing (nearest mask pixel to) C
    if mask.any():
        lab, nlab = ndi.label(mask, structure=st)
        ix, iy = _i(Cx), _i(Cy)
        chosen = 0
        if 0 <= ix < nx and 0 <= iy < ny:
            chosen = int(lab[iy, ix])
        if chosen == 0:
            ys, xs = np.where(mask)
            k = int(np.argmin((xs - Cx)**2 + (ys - Cy)**2))
            chosen = int(lab[ys[k], xs[k]])
        mask = (lab == chosen)

    mask_bin = mask.astype(bool)

    # smoothed mask for contour (visual only)
    mask_smooth = None
    if mask_bin.any():
        mask_smooth = ndi.gaussian_filter(mask_bin.astype(float), sigma=1.0)

    # dilated mask for label placement safety (avoid label box touching emission)
    mask_label_guard = mask_bin.copy()
    if mask_label_guard.any():
        mask_label_guard = ndi.binary_dilation(mask_label_guard, structure=st, iterations=10)

    def _mask_at(x, y, guard=False):
        m = mask_label_guard if guard else mask_bin
        if m is None or not np.any(m):
            return False
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return False
        return bool(m[iy, ix])

    def _unit(vx, vy):
        n = math.hypot(vx, vy) + 1e-9
        return vx/n, vy/n

    # =========================================================
    # 1) fixed square zoom centered on C (stable layout)
    # =========================================================
    half = max(max(d1, d2) * 1.18, 70.0)
    xmin = int(np.floor(Cx - half)); xmax = int(np.ceil(Cx + half))
    ymin = int(np.floor(Cy - half)); ymax = int(np.ceil(Cy + half))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(nx - 1, xmax); ymax = min(ny - 1, ymax)

    sub = data[ymin:ymax+1, xmin:xmax+1]
    finite = np.isfinite(sub)
    if finite.any():
        vmin, vmax = np.nanpercentile(sub[finite], [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(sub[finite]), np.nanmax(sub[finite])
    else:
        vmin, vmax = 0.0, 1.0

    # =========================================================
    # 2) intensity-weighted shortest-path centerline INSIDE mask (T1->C->T2)
    # =========================================================
    def _weighted_path2(mask_bin, start_xy, goal_xy, img):
        # (kept identical to your original centerline code)
        sx, sy = start_xy
        gx, gy = goal_xy
        if not mask_bin[sy, sx] or not mask_bin[gy, gx]:
            return None

        vals = img[mask_bin]
        if vals.size == 0:
            return None
        lo, hi = np.nanpercentile(vals, [8, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = (hi - lo) if hi > lo else 1.0

        def normI(x, y):
            v = img[y, x]
            if not np.isfinite(v):
                return 0.0
            t = (v - lo) / denom
            return float(np.clip(t, 0.0, 1.0))

        h, w_ = mask_bin.shape
        INF = 1e18
        dist = np.full((h, w_), INF, dtype=float)
        prev = np.full((h, w_, 2), -1, dtype=int)

        pq = []
        dist[sy, sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))

        nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        eps = 2e-3
        while pq:
            d, x, y = heapq.heappop(pq)
            if d != dist[y, x]:
                continue
            if x == gx and y == gy:
                break
            for dx, dy in nbrs:
                xx2, yy2 = x+dx, y+dy
                if xx2 < 0 or xx2 >= w_ or yy2 < 0 or yy2 >= h:
                    continue
                if not mask_bin[yy2, xx2]:
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                inten = normI(xx2, yy2)
                wgt = step * (1.0 / (eps + inten))
                nd = d + wgt
                if nd < dist[yy2, xx2]:
                    dist[yy2, xx2] = nd
                    prev[yy2, xx2, 0] = x
                    prev[yy2, xx2, 1] = y
                    heapq.heappush(pq, (nd, xx2, yy2))

        if not np.isfinite(dist[gy, gx]) or dist[gy, gx] >= INF/2:
            return None

        path = []
        x, y = gx, gy
        while not (x == sx and y == sy):
            path.append((x, y))
            px, py = prev[y, x]
            if px < 0:
                return None
            x, y = px, py
        path.append((sx, sy))
        path.reverse()
        return path

    centerline_xy = None
    if mask_bin.any():
        pT1 = _nearest_true_pixel(mask_bin, T1x, T1y, r=90)
        pC  = _nearest_true_pixel(mask_bin, Cx,  Cy,  r=90)
        pT2 = _nearest_true_pixel(mask_bin, T2x, T2y, r=90)
        if pT1 and pC and pT2:
            path1 = _weighted_path2(mask_bin, pT1, pC, data)
            path2 = _weighted_path2(mask_bin, pC,  pT2, data)
            if path1 and path2:
                centerline_xy = [(float(x), float(y)) for (x, y) in path1] + [(float(x), float(y)) for (x, y) in path2[1:]]
                clx0 = np.array([p[0] for p in centerline_xy])
                cly0 = np.array([p[1] for p in centerline_xy])
                k0 = int(np.argmin((clx0 - Cx)**2 + (cly0 - Cy)**2))
                centerline_xy[k0] = (float(Cx), float(Cy))

    if centerline_xy is None:
        t = np.linspace(0, 1, 30)
        seg1 = list(zip((T1x + (Cx-T1x)*t).astype(float), (T1y + (Cy-T1y)*t).astype(float)))
        seg2 = list(zip((Cx  + (T2x-Cx)*t).astype(float),  (Cy  + (T2y-Cy)*t).astype(float)))
        centerline_xy = seg1 + seg2[1:]

    clx = np.array([p[0] for p in centerline_xy], dtype=float)
    cly = np.array([p[1] for p in centerline_xy], dtype=float)

    # =========================================================
    # 3) canvas + colorbar same height
    # =========================================================
    fig = plt.figure(figsize=(7.6, 6.2), dpi=DPI, facecolor="white")
    ax = fig.add_subplot(111, projection=w)

    im = ax.imshow(data, origin="lower", cmap="OrRd", vmin=vmin, vmax=vmax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.coords[0].set_axislabel("Right Ascension (J2000)", minpad=1.2, fontsize=11, color="black")
    ax.coords[1].set_axislabel("Declination (J2000)",     minpad=1.2, fontsize=11, color="black")
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[0].set_format_unit(u.hourangle)
    ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.coords[1].set_format_unit(u.deg)
    ax.coords[0].set_ticklabel(size=9, color="black")
    ax.coords[1].set_ticklabel(size=9, color="black")
    try:
        ax.coords[0].offset_text.set_visible(False)
        ax.coords[1].offset_text.set_visible(False)
    except Exception:
        pass

    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.6)

    # mask outline (smoothed)
    if mask_smooth is not None and np.isfinite(mask_smooth).any() and mask_bin.any():
        ax.contour(mask_smooth, levels=[0.5],
                   colors=[MASK_COLOR], linewidths=1.6, linestyles=[MASK_LS],
                   zorder=5)

    # two rays (black)
    ax.plot([Cx, T1x], [Cy, T1y], color="k", linewidth=1.8, alpha=0.85, zorder=6)
    ax.plot([Cx, T2x], [Cy, T2y], color="k", linewidth=1.8, alpha=0.85, zorder=6)

    # centerline
    ax.plot(clx, cly, linestyle=(0, (3.0, 3.0)), linewidth=2.4,
            color=(0.10, 0.35, 0.95), alpha=0.98, zorder=7)

    # points
    ax.scatter([T1x, T2x], [T1y, T2y], s=54, c=POINT_COLOR, edgecolors="none", zorder=9)
    ax.scatter([Cx], [Cy], s=95, marker="x", c=POINT_COLOR, linewidths=2.6, zorder=10)

    # =========================================================
    # 4) OA arc
    # =========================================================
    v1 = np.array([T1x - Cx, T1y - Cy], dtype=float)
    v2 = np.array([T2x - Cx, T2y - Cy], dtype=float)
    dmin2 = min(np.hypot(*v1), np.hypot(*v2))

    r = 0.12 * dmin2
    r = max(r, 8.0)
    r = min(r, 0.18 * dmin2)

    ang1 = (math.degrees(math.atan2(v1[1], v1[0])) + 360.0) % 360.0
    ang2 = (math.degrees(math.atan2(v2[1], v2[0])) + 360.0) % 360.0
    oa = float(oa_deg)

    delta_ccw = (ang2 - ang1 + 360.0) % 360.0
    delta_cw  = (ang1 - ang2 + 360.0) % 360.0
    if abs(delta_ccw - oa) <= abs(delta_cw - oa):
        theta1, theta2 = ang1, ang1 + delta_ccw
        midang = ang1 + 0.5 * delta_ccw
    else:
        theta1, theta2 = ang1, ang1 - delta_cw
        midang = ang1 - 0.5 * delta_cw

    ax.add_patch(Arc((Cx, Cy), 2*r, 2*r,
                     theta1=theta1, theta2=theta2,
                     linewidth=2.2, color="k", alpha=0.85, zorder=8))

    # =========================================================
    # 5) labels: keep outside (DILATED) mask
    # =========================================================
    def annotate_box(text, x, y, dx, dy):
        ax.annotate(
            text, (x, y),
            xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center",
            fontsize=10, weight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2),
            zorder=30
        )

    def smart_offset(px, py):
        vx, vy = _unit(px - Cx, py - Cy)
        nx_, ny_ = (-vy, vx)
        cand = [(18*nx_, 18*ny_), (-18*nx_, -18*ny_)]
        for dx, dy in cand:
            x_try = px + 0.9*dx
            y_try = py + 0.9*dy
            if not _mask_at(x_try, y_try, guard=True):
                return int(round(dx)), int(round(dy))
        return int(round(cand[0][0])), int(round(cand[0][1]))

    dx1, dy1 = smart_offset(T1x, T1y)
    dx2, dy2 = smart_offset(T2x, T2y)

    k = int(np.argmin((clx - Cx)**2 + (cly - Cy)**2))
    a = max(0, k-8); b = min(len(clx)-1, k+8)
    tvx, tvy = _unit(clx[b]-clx[a], cly[b]-cly[a])
    cnx, cny = (-tvy, tvx)
    dxC, dyC = int(round(18*cnx)), int(round(18*cny))
    if _mask_at(Cx + 0.9*dxC, Cy + 0.9*dyC, guard=True):
        dxC, dyC = -dxC, -dyC

    annotate_box("T$_1$", T1x, T1y, dx1, dy1)
    annotate_box("T$_2$", T2x, T2y, dx2, dy2)
    annotate_box("C",     Cx,  Cy,  dxC, dyC)

    def find_outside_position():
        for sgn in (+1.0, -1.0):
            vx, vy = math.cos(math.radians(midang))*sgn, math.sin(math.radians(midang))*sgn
            x = Cx + (2.20*r) * vx
            y = Cy + (2.20*r) * vy
            x, y = _clip_xy(x, y)
            if not _mask_at(x, y, guard=True):
                return x, y
            for _ in range(500):
                x, y = _clip_xy(x + 3.5*vx, y + 3.5*vy)
                if not _mask_at(x, y, guard=True):
                    return x, y

        best = None
        for ang in np.linspace(0, 2*np.pi, 96, endpoint=False):
            vx, vy = math.cos(ang), math.sin(ang)
            x = Cx + (2.05*r) * vx
            y = Cy + (2.05*r) * vy
            x, y = _clip_xy(x, y)
            if _mask_at(x, y, guard=True):
                for _ in range(400):
                    x, y = _clip_xy(x + 3.5*vx, y + 3.5*vy)
                    if not _mask_at(x, y, guard=True):
                        break
            if not _mask_at(x, y, guard=True):
                d = (x-Cx)*(x-Cx) + (y-Cy)*(y-Cy)
                if best is None or d < best[0]:
                    best = (d, x, y)
        if best is not None:
            return best[1], best[2]
        return _clip_xy(Cx + 2.6*r, Cy + 2.6*r)

    ox, oy = find_outside_position()
    annotate_box(f"OA = {oa:.1f}°", ox, oy, 0, 0)

    # =========================================================
    # 6) colorbar: same height, no stray x-label
    # =========================================================
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.8%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Flux density/(mJy beam$^{-1}$)")
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(axis='y', labelright=True, labelleft=False, direction='out', pad=6)
    cax.set_xlabel("")
    cax.set_xticks([])
    cax.tick_params(axis="x", bottom=False, labelbottom=False)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cbar.formatter = fmt
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()

    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.12, top=0.96)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
def plot_schematic(fits_path, C, T1, T2, oa_deg, out_png):
    """
    Paper-style OA schematic with:
      - tight, connected mask (constrained around a CURVED spine derived from emission)
      - smoothed mask outline
      - WCS axis labels not clipped
      - colorbar same height as main panel (no stray 'X' label)
      - OA text forced outside (dilated) mask to avoid covering emission

    NOTE: Plot style is unchanged; ONLY the mask constraint logic is updated so the mask
          follows the real bent emission instead of straight "tube" segments.
    """
    import os, math, heapq
    import numpy as np
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc
    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy import ndimage as ndi

    # ---------------- load ----------------
    with fits.open(fits_path, memmap=False) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
        w = WCS(hdul[0].header).celestial

    ny, nx = data.shape

    # ---------------- world -> pixel ----------------
    Cx, Cy = w.world_to_pixel(C);    Cx, Cy = _f(Cx), _f(Cy)
    T1x, T1y = w.world_to_pixel(T1); T1x, T1y = _f(T1x), _f(T1y)
    T2x, T2y = w.world_to_pixel(T2); T2x, T2y = _f(T2x), _f(T2y)

    def _clip_xy(x, y):
        x = float(np.clip(x, 0, nx - 1))
        y = float(np.clip(y, 0, ny - 1))
        return x, y

    # ---------------- mask (tight + connected) ----------------
    mask0, bkg, sig = build_connected_mask(
        data, w, C,
        high_sigma=max(THRESH_SIGMA, 3.8),
        low_sigma=2.2,
        close_iters=1,
        open_iters=1,
        erode_iters=2,
        min_pix=40
    )
    mask0 = (mask0.astype(bool) if mask0 is not None else np.zeros_like(data, bool))

    # =========================================================
    # ✅ ONLY CHANGE: constrain mask around a CURVED spine
    #     (instead of straight C->T1 / C->T2 distance tubes)
    # =========================================================
    st = ndi.generate_binary_structure(2, 1)

    def _nearest_true_pixel(binimg, x, y, r=70):
        iy, ix = int(round(y)), int(round(x))
        y0 = max(0, iy-r); y1 = min(ny-1, iy+r)
        x0 = max(0, ix-r); x1 = min(nx-1, ix+r)
        cut = binimg[y0:y1+1, x0:x1+1]
        ys, xs = np.where(cut)
        if xs.size == 0:
            return None
        xs = xs + x0
        ys = ys + y0
        dx = xs - x
        dy = ys - y
        k = int(np.argmin(dx*dx + dy*dy))
        return int(xs[k]), int(ys[k])

    def _weighted_path(mask_bin, start_xy, goal_xy, img):
        sx, sy = start_xy
        gx, gy = goal_xy
        if not mask_bin[sy, sx] or not mask_bin[gy, gx]:
            return None

        vals = img[mask_bin]
        if vals.size == 0:
            return None
        lo, hi = np.nanpercentile(vals, [8, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = (hi - lo) if hi > lo else 1.0

        def normI(x, y):
            v = img[y, x]
            if not np.isfinite(v):
                return 0.0
            t = (v - lo) / denom
            return float(np.clip(t, 0.0, 1.0))

        h, w_ = mask_bin.shape
        INF = 1e18
        dist = np.full((h, w_), INF, dtype=float)
        prev = np.full((h, w_, 2), -1, dtype=int)

        pq = []
        dist[sy, sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))

        nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        eps = 2e-3
        while pq:
            d, x, y = heapq.heappop(pq)
            if d != dist[y, x]:
                continue
            if x == gx and y == gy:
                break
            for dx, dy in nbrs:
                xx2, yy2 = x+dx, y+dy
                if xx2 < 0 or xx2 >= w_ or yy2 < 0 or yy2 >= h:
                    continue
                if not mask_bin[yy2, xx2]:
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                inten = normI(xx2, yy2)
                wgt = step * (1.0 / (eps + inten))
                nd = d + wgt
                if nd < dist[yy2, xx2]:
                    dist[yy2, xx2] = nd
                    prev[yy2, xx2, 0] = x
                    prev[yy2, xx2, 1] = y
                    heapq.heappush(pq, (nd, xx2, yy2))

        if not np.isfinite(dist[gy, gx]) or dist[gy, gx] >= INF/2:
            return None

        path = []
        x, y = gx, gy
        while not (x == sx and y == sy):
            path.append((x, y))
            px, py = prev[y, x]
            if px < 0:
                return None
            x, y = px, py
        path.append((sx, sy))
        path.reverse()
        return path

    d1 = float(np.hypot(T1x - Cx, T1y - Cy))
    d2 = float(np.hypot(T2x - Cx, T2y - Cy))
    dmin = max(1.0, min(d1, d2))

    # tube half-width in pixels (same scaling as before)
    tube_w = float(np.clip(0.13 * dmin, 6.0, 26.0))

    yy, xx = np.indices((ny, nx), dtype=float)

    # keep a core disk around C (same as before)
    core_r = float(np.clip(0.18 * dmin, 8.0, 30.0))
    core_disk = ((xx - Cx)**2 + (yy - Cy)**2) <= core_r*core_r

    # Build curved spine (T1->C->T2) inside mask0; fallback to piecewise-linear if needed
    spine = np.zeros((ny, nx), dtype=bool)
    if mask0 is not None and mask0.any():
        pT1 = _nearest_true_pixel(mask0, T1x, T1y, r=90)
        pC  = _nearest_true_pixel(mask0, Cx,  Cy,  r=90)
        pT2 = _nearest_true_pixel(mask0, T2x, T2y, r=90)
        if pT1 and pC and pT2:
            path1 = _weighted_path(mask0, pT1, pC, data)
            path2 = _weighted_path(mask0, pC,  pT2, data)
            if path1 and path2:
                for (x, y) in (path1 + path2[1:]):
                    if 0 <= x < nx and 0 <= y < ny:
                        spine[y, x] = True

    if not spine.any():
        t = np.linspace(0, 1, 160)
        xs1 = (T1x + (Cx - T1x)*t).astype(float)
        ys1 = (T1y + (Cy - T1y)*t).astype(float)
        xs2 = (Cx  + (T2x - Cx)*t).astype(float)
        ys2 = (Cy  + (T2y - Cy)*t).astype(float)
        xs = np.concatenate([xs1, xs2[1:]])
        ys = np.concatenate([ys1, ys2[1:]])
        xi = np.clip(np.rint(xs).astype(int), 0, nx-1)
        yi = np.clip(np.rint(ys).astype(int), 0, ny-1)
        spine[yi, xi] = True

    spine = ndi.binary_dilation(spine, structure=st, iterations=1)
    dist_to_spine = ndi.distance_transform_edt(~spine)
    tube = dist_to_spine <= tube_w

    # Constrain to curved tube + core disk
    mask = mask0 & (tube | core_disk)

    # Gentle regularization (same spirit as your old post-processing)
    if mask.any():
        mask = ndi.binary_closing(mask, structure=st, iterations=1)
        mask = ndi.binary_dilation(mask, structure=st, iterations=1)

    # keep only the component containing (nearest mask pixel to) C
    if mask.any():
        lab, nlab = ndi.label(mask, structure=st)
        ix, iy = _i(Cx), _i(Cy)
        chosen = 0
        if 0 <= ix < nx and 0 <= iy < ny:
            chosen = int(lab[iy, ix])
        if chosen == 0:
            ys, xs = np.where(mask)
            k = int(np.argmin((xs - Cx)**2 + (ys - Cy)**2))
            chosen = int(lab[ys[k], xs[k]])
        mask = (lab == chosen)

    mask_bin = mask.astype(bool)

    # smoothed mask for contour (visual only)
    mask_smooth = None
    if mask_bin.any():
        mask_smooth = ndi.gaussian_filter(mask_bin.astype(float), sigma=1.0)

    # dilated mask for label placement safety (avoid label box touching emission)
    mask_label_guard = mask_bin.copy()
    if mask_label_guard.any():
        mask_label_guard = ndi.binary_dilation(mask_label_guard, structure=st, iterations=10)

    def _mask_at(x, y, guard=False):
        m = mask_label_guard if guard else mask_bin
        if m is None or not np.any(m):
            return False
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            return False
        return bool(m[iy, ix])

    def _unit(vx, vy):
        n = math.hypot(vx, vy) + 1e-9
        return vx/n, vy/n

    # =========================================================
    # 1) fixed square zoom centered on C (stable layout)
    # =========================================================
    half = max(max(d1, d2) * 1.18, 70.0)
    xmin = int(np.floor(Cx - half)); xmax = int(np.ceil(Cx + half))
    ymin = int(np.floor(Cy - half)); ymax = int(np.ceil(Cy + half))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(nx - 1, xmax); ymax = min(ny - 1, ymax)

    sub = data[ymin:ymax+1, xmin:xmax+1]
    finite = np.isfinite(sub)
    if finite.any():
        vmin, vmax = np.nanpercentile(sub[finite], [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(sub[finite]), np.nanmax(sub[finite])
    else:
        vmin, vmax = 0.0, 1.0

    # =========================================================
    # 2) intensity-weighted shortest-path centerline INSIDE mask (T1->C->T2)
    # =========================================================
    def _weighted_path2(mask_bin, start_xy, goal_xy, img):
        # (kept identical to your original centerline code)
        sx, sy = start_xy
        gx, gy = goal_xy
        if not mask_bin[sy, sx] or not mask_bin[gy, gx]:
            return None

        vals = img[mask_bin]
        if vals.size == 0:
            return None
        lo, hi = np.nanpercentile(vals, [8, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        denom = (hi - lo) if hi > lo else 1.0

        def normI(x, y):
            v = img[y, x]
            if not np.isfinite(v):
                return 0.0
            t = (v - lo) / denom
            return float(np.clip(t, 0.0, 1.0))

        h, w_ = mask_bin.shape
        INF = 1e18
        dist = np.full((h, w_), INF, dtype=float)
        prev = np.full((h, w_, 2), -1, dtype=int)

        pq = []
        dist[sy, sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))

        nbrs = [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]
        eps = 2e-3
        while pq:
            d, x, y = heapq.heappop(pq)
            if d != dist[y, x]:
                continue
            if x == gx and y == gy:
                break
            for dx, dy in nbrs:
                xx2, yy2 = x+dx, y+dy
                if xx2 < 0 or xx2 >= w_ or yy2 < 0 or yy2 >= h:
                    continue
                if not mask_bin[yy2, xx2]:
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                inten = normI(xx2, yy2)
                wgt = step * (1.0 / (eps + inten))
                nd = d + wgt
                if nd < dist[yy2, xx2]:
                    dist[yy2, xx2] = nd
                    prev[yy2, xx2, 0] = x
                    prev[yy2, xx2, 1] = y
                    heapq.heappush(pq, (nd, xx2, yy2))

        if not np.isfinite(dist[gy, gx]) or dist[gy, gx] >= INF/2:
            return None

        path = []
        x, y = gx, gy
        while not (x == sx and y == sy):
            path.append((x, y))
            px, py = prev[y, x]
            if px < 0:
                return None
            x, y = px, py
        path.append((sx, sy))
        path.reverse()
        return path

    centerline_xy = None
    if mask_bin.any():
        pT1 = _nearest_true_pixel(mask_bin, T1x, T1y, r=90)
        pC  = _nearest_true_pixel(mask_bin, Cx,  Cy,  r=90)
        pT2 = _nearest_true_pixel(mask_bin, T2x, T2y, r=90)
        if pT1 and pC and pT2:
            path1 = _weighted_path2(mask_bin, pT1, pC, data)
            path2 = _weighted_path2(mask_bin, pC,  pT2, data)
            if path1 and path2:
                centerline_xy = [(float(x), float(y)) for (x, y) in path1] + [(float(x), float(y)) for (x, y) in path2[1:]]
                clx0 = np.array([p[0] for p in centerline_xy])
                cly0 = np.array([p[1] for p in centerline_xy])
                k0 = int(np.argmin((clx0 - Cx)**2 + (cly0 - Cy)**2))
                centerline_xy[k0] = (float(Cx), float(Cy))

    if centerline_xy is None:
        t = np.linspace(0, 1, 30)
        seg1 = list(zip((T1x + (Cx-T1x)*t).astype(float), (T1y + (Cy-T1y)*t).astype(float)))
        seg2 = list(zip((Cx  + (T2x-Cx)*t).astype(float),  (Cy  + (T2y-Cy)*t).astype(float)))
        centerline_xy = seg1 + seg2[1:]

    clx = np.array([p[0] for p in centerline_xy], dtype=float)
    cly = np.array([p[1] for p in centerline_xy], dtype=float)

    # =========================================================
    # 3) canvas + colorbar same height
    # =========================================================
    fig = plt.figure(figsize=(7.6, 6.2), dpi=DPI, facecolor="white")
    ax = fig.add_subplot(111, projection=w)

    im = ax.imshow(data, origin="lower", cmap="OrRd", vmin=vmin, vmax=vmax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.coords[0].set_axislabel("Right Ascension (J2000)", minpad=1.2, fontsize=11, color="black")
    ax.coords[1].set_axislabel("Declination (J2000)",     minpad=1.2, fontsize=11, color="black")
    ax.coords[0].set_major_formatter('hh:mm:ss')
    ax.coords[0].set_format_unit(u.hourangle)
    ax.coords[1].set_major_formatter('dd:mm:ss')
    ax.coords[1].set_format_unit(u.deg)
    ax.coords[0].set_ticklabel(size=9, color="black")
    ax.coords[1].set_ticklabel(size=9, color="black")
    try:
        ax.coords[0].offset_text.set_visible(False)
        ax.coords[1].offset_text.set_visible(False)
    except Exception:
        pass

    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.6)

    # mask outline (smoothed)
    if mask_smooth is not None and np.isfinite(mask_smooth).any() and mask_bin.any():
        ax.contour(mask_smooth, levels=[0.5],
                   colors=[MASK_COLOR], linewidths=1.6, linestyles=[MASK_LS],
                   zorder=5)

    # two rays (black)
    ax.plot([Cx, T1x], [Cy, T1y], color="k", linewidth=1.8, alpha=0.85, zorder=6)
    ax.plot([Cx, T2x], [Cy, T2y], color="k", linewidth=1.8, alpha=0.85, zorder=6)

    # centerline
    ax.plot(clx, cly, linestyle=(0, (3.0, 3.0)), linewidth=2.4,
            color=(0.10, 0.35, 0.95), alpha=0.98, zorder=7)

    # points
    ax.scatter([T1x, T2x], [T1y, T2y], s=54, c=POINT_COLOR, edgecolors="none", zorder=9)
    ax.scatter([Cx], [Cy], s=95, marker="x", c=POINT_COLOR, linewidths=2.6, zorder=10)

    # =========================================================
    # 4) OA arc
    # =========================================================
    v1 = np.array([T1x - Cx, T1y - Cy], dtype=float)
    v2 = np.array([T2x - Cx, T2y - Cy], dtype=float)
    dmin2 = min(np.hypot(*v1), np.hypot(*v2))

    r = 0.12 * dmin2
    r = max(r, 8.0)
    r = min(r, 0.18 * dmin2)

    ang1 = (math.degrees(math.atan2(v1[1], v1[0])) + 360.0) % 360.0
    ang2 = (math.degrees(math.atan2(v2[1], v2[0])) + 360.0) % 360.0
    oa = float(oa_deg)

    delta_ccw = (ang2 - ang1 + 360.0) % 360.0
    delta_cw  = (ang1 - ang2 + 360.0) % 360.0
    if abs(delta_ccw - oa) <= abs(delta_cw - oa):
        theta1, theta2 = ang1, ang1 + delta_ccw
        midang = ang1 + 0.5 * delta_ccw
    else:
        theta1, theta2 = ang1, ang1 - delta_cw
        midang = ang1 - 0.5 * delta_cw

    ax.add_patch(Arc((Cx, Cy), 2*r, 2*r,
                     theta1=theta1, theta2=theta2,
                     linewidth=2.2, color="k", alpha=0.85, zorder=8))

    # =========================================================
    # 5) labels: keep outside (DILATED) mask
    # =========================================================
    def annotate_box(text, x, y, dx, dy):
        ax.annotate(
            text, (x, y),
            xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center",
            fontsize=10, weight="bold",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2),
            zorder=30
        )

    def smart_offset(px, py):
        vx, vy = _unit(px - Cx, py - Cy)
        nx_, ny_ = (-vy, vx)
        cand = [(18*nx_, 18*ny_), (-18*nx_, -18*ny_)]
        for dx, dy in cand:
            x_try = px + 0.9*dx
            y_try = py + 0.9*dy
            if not _mask_at(x_try, y_try, guard=True):
                return int(round(dx)), int(round(dy))
        return int(round(cand[0][0])), int(round(cand[0][1]))

    dx1, dy1 = smart_offset(T1x, T1y)
    dx2, dy2 = smart_offset(T2x, T2y)

    k = int(np.argmin((clx - Cx)**2 + (cly - Cy)**2))
    a = max(0, k-8); b = min(len(clx)-1, k+8)
    tvx, tvy = _unit(clx[b]-clx[a], cly[b]-cly[a])
    cnx, cny = (-tvy, tvx)
    dxC, dyC = int(round(18*cnx)), int(round(18*cny))
    if _mask_at(Cx + 0.9*dxC, Cy + 0.9*dyC, guard=True):
        dxC, dyC = -dxC, -dyC

    annotate_box("T$_1$", T1x, T1y, dx1, dy1)
    annotate_box("T$_2$", T2x, T2y, dx2, dy2)
    annotate_box("C",     Cx,  Cy,  dxC, dyC)

    def find_outside_position():
        for sgn in (+1.0, -1.0):
            vx, vy = math.cos(math.radians(midang))*sgn, math.sin(math.radians(midang))*sgn
            x = Cx + (2.20*r) * vx
            y = Cy + (2.20*r) * vy
            x, y = _clip_xy(x, y)
            if not _mask_at(x, y, guard=True):
                return x, y
            for _ in range(500):
                x, y = _clip_xy(x + 3.5*vx, y + 3.5*vy)
                if not _mask_at(x, y, guard=True):
                    return x, y

        best = None
        for ang in np.linspace(0, 2*np.pi, 96, endpoint=False):
            vx, vy = math.cos(ang), math.sin(ang)
            x = Cx + (2.05*r) * vx
            y = Cy + (2.05*r) * vy
            x, y = _clip_xy(x, y)
            if _mask_at(x, y, guard=True):
                for _ in range(400):
                    x, y = _clip_xy(x + 3.5*vx, y + 3.5*vy)
                    if not _mask_at(x, y, guard=True):
                        break
            if not _mask_at(x, y, guard=True):
                d = (x-Cx)*(x-Cx) + (y-Cy)*(y-Cy)
                if best is None or d < best[0]:
                    best = (d, x, y)
        if best is not None:
            return best[1], best[2]
        return _clip_xy(Cx + 2.6*r, Cy + 2.6*r)

    ox, oy = find_outside_position()
    annotate_box(f"OA = {oa:.1f}°", ox, oy, 0, 0)

    # =========================================================
    # 6) colorbar: same height, no stray x-label
    # =========================================================
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.8%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Flux density/(mJy beam$^{-1}$)")
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(axis='y', labelright=True, labelleft=False, direction='out', pad=6)
    cax.set_xlabel("")
    cax.set_xticks([])
    cax.tick_params(axis="x", bottom=False, labelbottom=False)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cbar.formatter = fmt
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()

    fig.subplots_adjust(left=0.18, right=0.86, bottom=0.12, top=0.96)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)



# =======================
# Catalog coords (robust)
# =======================
def build_catalog_coords(cat: Table):
    ra_col = pick_col(cat, ["RA", "ra", "RAJ2000", "raj2000", "host_ra", "HOST_RA", "ra_host", "RA_host", "ra0"])
    dec_col = pick_col(cat, ["DEC", "Dec", "dec", "DEJ2000", "dej2000", "host_dec", "HOST_DEC", "dec_host", "DEC_host", "dec0"])
    if ra_col is None or dec_col is None:
        raise RuntimeError(f"Cannot find RA/DEC columns in catalog. Columns:\n{cat.colnames}")

    ra = cat[ra_col]
    dec = cat[dec_col]

    # string sexagesimal?
    if ra.dtype.kind in ("U", "S", "O") or dec.dtype.kind in ("U", "S", "O"):
        ra_s = np.array(ra).astype(str)
        dec_s = np.array(dec).astype(str)
        is_hms = np.array([(":" in s) or ("h" in s.lower()) for s in ra_s]).any()
        if is_hms:
            coords = SkyCoord(ra_s, dec_s, unit=(u.hourangle, u.deg), frame="icrs")
        else:
            coords = SkyCoord(ra_s.astype(float)*u.deg, dec_s.astype(float)*u.deg, frame="icrs")
    else:
        coords = SkyCoord(np.array(ra, dtype=float)*u.deg, np.array(dec, dtype=float)*u.deg, frame="icrs")

    return coords


# =======================
# FITS center coords
# =======================
def fits_center_coord(fits_path):
    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.squeeze(hdul[0].data)
        w = WCS(hdr).celestial

    ny, nx = data.shape[-2], data.shape[-1]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    return SkyCoord.from_pixel(cx, cy, w)


# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cat = Table.read(CAT_PATH)
    oa_col = pick_col(cat, ["OA", "oa", "opening_angle", "OA_deg", "oa_deg"])
    if oa_col is None:
        raise RuntimeError(f"Cannot find OA column in catalog. Columns:\n{cat.colnames}")

    cat_coords = build_catalog_coords(cat)

    fits_list = sorted(glob.glob(os.path.join(FITS_DIR, "*.fits")))
    if not fits_list:
        print(f"[ERROR] No FITS found in {FITS_DIR}")
        return

    fits_centers = []
    for fp in fits_list:
        try:
            fits_centers.append(fits_center_coord(fp))
        except Exception as e:
            fits_centers.append(None)
            print(f"[WARN] FITS center failed: {os.path.basename(fp)}  {e}")

    label_files = sorted([p for p in glob.glob(os.path.join(LABEL_DIR, "*")) if os.path.isfile(p)])
    if not label_files:
        print(f"[ERROR] No label files found in {LABEL_DIR}")
        return

    labels = []
    for lp in label_files:
        try:
            C, Ts = parse_crtf_symbols(lp)
            if C is None or len(Ts) < 2:
                continue
            labels.append((lp, C, Ts))
        except Exception:
            continue

    if not labels:
        print("[ERROR] No valid labels parsed (need C and >=2 T).")
        return

    print("========== MATCHING / OA CHECK ==========")
    print(f"[INFO] FITS:   {len(fits_list)}")
    print(f"[INFO] LABELS: {len(labels)} valid")
    print(f"[INFO] Catalog rows: {len(cat)}")
    print(f"[INFO] OA column: {oa_col}")
    print(f"[INFO] MAX_SEP_FITS_LABEL = {_f(MAX_SEP_FITS_LABEL.to(u.arcsec)):.1f}\"")
    print(f"[INFO] MAX_SEP_LABEL_CAT  = {_f(MAX_SEP_LABEL_CAT.to(u.arcsec)):.1f}\"")
    print("========================================\n")

    used_fits = set()
    n_match = 0
    n_no_fits = 0
    n_no_cat = 0

    diffs = []

    # label(C) -> nearest FITS center
    for (lp, C, Ts) in labels:
        best_i = None
        best_sep = None

        for i, fc in enumerate(fits_centers):
            if fc is None:
                continue
            sep = C.separation(fc)
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_i = i

        if best_i is None or best_sep is None or best_sep > MAX_SEP_FITS_LABEL:
            n_no_fits += 1
            sep_arc = _f(best_sep.to(u.arcsec)) if best_sep is not None else float("nan")
            print(f"[NO FITS] {os.path.basename(lp)}  sep={sep_arc:.1f}\"")
            continue

        fp = fits_list[best_i]
        used_fits.add(fp)

        # choose 2 Ts: if >2, take two farthest from C (stable)
        if len(Ts) > 2:
            seps = np.array([C.separation(t).arcsec for t in Ts], dtype=float)
            idx = np.argsort(seps)[::-1][:2]
            Ts2 = [Ts[i] for i in idx]
        else:
            Ts2 = Ts[:2]

        # order by PA for consistent T1/T2
        pa = np.array([C.position_angle(t).wrap_at(360*u.deg).deg for t in Ts2], dtype=float)
        order = np.argsort(pa)
        T1, T2 = Ts2[order[0]], Ts2[order[1]]

        oa_calc = float(compute_oa_deg(C, T1, T2))

        # label(C) -> nearest catalog row
        idx_cat, sep2d, _ = C.match_to_catalog_sky(cat_coords)

        if sep2d > MAX_SEP_LABEL_CAT:
            n_no_cat += 1
            sep_lc = _f(sep2d.to(u.arcsec))
            print(f"[NO CAT ] {os.path.basename(fp)}  label={os.path.basename(lp)}  sep={sep_lc:.2f}\"  OA_calc={oa_calc:.2f}")
            continue

        oa_cat = cat[oa_col][idx_cat]
        oa_cat = float(oa_cat) if np.isfinite(oa_cat) else float("nan")
        diff = oa_calc - oa_cat if np.isfinite(oa_cat) else float("nan")
        diffs.append(diff)

        sep_fl = _f(best_sep.to(u.arcsec))
        sep_lc = _f(sep2d.to(u.arcsec))

        print(f"{os.path.basename(fp):>30s} | sep(F-L)={sep_fl:6.1f}\" | sep(L-CAT)={sep_lc:6.2f}\" | "
              f"OA_calc={oa_calc:7.2f} | OA_cat={oa_cat:7.2f} | diff={diff:7.2f}")

        out_png = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(fp))[0] + "_OA.png")
        try:
            plot_schematic(fp, C, T1, T2, oa_calc, out_png)
            n_match += 1
        except Exception as e:
            print(f"[WARN] plot failed: {os.path.basename(fp)}  {e}")

    print("\n========== SUMMARY ==========")
    print(f"[OK] matched & plotted: {n_match}")
    print(f"[MISS] label->fits failed: {n_no_fits}")
    print(f"[MISS] label->catalog failed: {n_no_cat}")
    if len(diffs) > 0:
        diffs = np.array([d for d in diffs if np.isfinite(d)], dtype=float)
        if diffs.size > 0:
            print(f"[DIFF] mean={diffs.mean():.3f}  median={np.median(diffs):.3f}  std={diffs.std():.3f}  max_abs={np.max(np.abs(diffs)):.3f} (deg)")
    print(f"[OUT] {OUT_DIR}")
    print("================================")


if __name__ == "__main__":
    main()