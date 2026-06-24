#!/usr/bin/env python3
# -*- coding: utf-8 -*-ƒ

import os, glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import ndimage as ndi
from astropy.stats import sigma_clipped_stats

# ===================== 路径（按你的） =====================
IN_GLOB = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/cube_bent_new/*.fits"
OUT_DIR = "/shared/main/caojie/meerkat/alpha_out"
# ==========================================================

LOW_CH  = [0, 1, 2]
HIGH_CH = [4, 5, 6]

CONTOUR_START_SIGMA = 3.0
CONTOUR_LEVELS_N = 8

# ====== ✅ 关键：双阈值连通，让桥/尾连起来 ======
HIGH_SIGMA_CONN = 3.0   # 核心阈值
LOW_SIGMA_CONN  = 2.0   # 连通阈值（桥/尾经常在2-3σ）
POSITIVE_ONLY = True

# 形态处理：先连通再适当膨胀追边缘
CONN_CLOSE_ITER = 2     # 连通闭运算（增强桥接）
CONN_DILATE_ITER = 6    # 追边缘（你之前用的6）

# ====== ✅ 高频只做“可用性” ======
PRE_SMOOTH_I_SIGMA = 2.0
REQUIRE_HIGH_POSITIVE_ONLY = True

# ====== ✅ 防溢出关键：noise floor + alpha裁剪 ======
K_LOW_FLOOR  = 1.0
K_HIGH_FLOOR = 1.5
ALPHA_CLIP_MIN = -5.0
ALPHA_CLIP_MAX =  1.0

# ====== 误差过滤 ======
AERR_MAX = 1.2

# ====== 绘图风格（保持不变） ======
ALPHA_CMAP = "rainbow"
AUTO_RANGE = True
PCT_LO, PCT_HI = 2, 98
SHIFT_ZERO_START = True
ALPHA_SMOOTH_SIGMA = 1.0

# ====== ✅ 固定画布/布局（保证每张一样） ======
FIGSIZE = (7.0, 5.4)
DPI = 200
# 主图固定占位：右侧留给colorbar（不使用tight_layout）
ADJ = dict(left=0.10, right=0.84, bottom=0.10, top=0.97)
# colorbar固定位置（相对figure坐标）
CBAR_RECT = (0.1, 0.12, 0.035, 0.80)  # (x0,y0,w,h)

# ====== ✅ 自动zoom（只改视野，不改画布）=====
AUTO_ZOOM = True
ZOOM_PAD_FRAC = 0.18
ZOOM_MIN_SIZE = 80
ZOOM_USE_ALPHA = True

SAVE_PNG = True
WRITE_I_FITS = False


def rms_background(img, clip_sigma=3.0, maxiters=10, exclude_hi_percentile=90):
    x = np.array(img, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 1000:
        return np.nan
    hi = np.nanpercentile(x, exclude_hi_percentile)
    x = x[x < hi]
    if x.size < 1000:
        return np.nan
    _, _, std = sigma_clipped_stats(x, sigma=clip_sigma, maxiters=maxiters)
    return float(std)


def find_freq_fits_axis(header):
    naxis = header.get("NAXIS", None)
    if naxis is None:
        raise ValueError("Header missing NAXIS")
    for ax in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{ax}", "")).upper()
        if "FREQ" in ctype:
            return ax
    raise ValueError("Cannot find frequency axis: no CTYPEi contains 'FREQ'.")


def freqs_from_header(header, freq_fits_axis, nchan):
    crval = header.get(f"CRVAL{freq_fits_axis}")
    cdelt = header.get(f"CDELT{freq_fits_axis}")
    crpix = header.get(f"CRPIX{freq_fits_axis}")
    cunit = header.get(f"CUNIT{freq_fits_axis}", "Hz")
    if crval is None or cdelt is None or crpix is None:
        raise ValueError("Frequency axis missing CRVAL/CDELT/CRPIX")

    i = np.arange(nchan, dtype=float)
    f = crval + ((i + 1.0) - crpix) * cdelt
    try:
        return (f * u.Unit(cunit)).to(u.Hz).value.astype(float)
    except Exception:
        return (f * u.Hz).value.astype(float)


def pixscale_deg_from_wcs2d(wcs2d):
    return float(np.mean(np.abs(wcs2d.wcs.cdelt)))


def read_beams_per_chan(header, nchan):
    beams = []
    for k in range(nchan):
        bmaj = header.get(f"BMAJ{k}", header.get(f"BMAJ{k+1}", header.get("BMAJ")))
        bmin = header.get(f"BMIN{k}", header.get(f"BMIN{k+1}", header.get("BMIN")))
        bpa  = header.get(f"BPA{k}",  header.get(f"BPA{k+1}",  header.get("BPA", 0.0)))
        if bmaj is None or bmin is None:
            raise ValueError(f"Missing beam keywords for channel {k}")
        beams.append((float(bmaj), float(bmin), float(bpa)))
    return beams


def gaussian_kernel_from_extra(extra_bmaj_deg, extra_bmin_deg, bpa_deg, pixscale_deg):
    if extra_bmaj_deg <= 0 or extra_bmin_deg <= 0:
        return None
    sig_y = (extra_bmaj_deg / 2.355) / pixscale_deg
    sig_x = (extra_bmin_deg / 2.355) / pixscale_deg
    if sig_x <= 0 or sig_y <= 0:
        return None

    theta = np.deg2rad(bpa_deg)
    size = int(np.ceil(8 * max(sig_x, sig_y)))
    size = max(size, 9)
    if size % 2 == 0:
        size += 1

    yy, xx = np.mgrid[-size//2:size//2+1, -size//2:size//2+1]
    xrot = np.cos(theta) * xx + np.sin(theta) * yy
    yrot = -np.sin(theta) * xx + np.cos(theta) * yy
    g = np.exp(-0.5 * ((xrot / sig_x) ** 2 + (yrot / sig_y) ** 2))
    g /= np.sum(g)
    return g.astype(float)


def smooth_to_common_beam(cube, beams_deg, pixscale_deg):
    bmaj_arr = np.array([b[0] for b in beams_deg], dtype=float)
    bmin_arr = np.array([b[1] for b in beams_deg], dtype=float)
    bpa_arr  = np.array([b[2] for b in beams_deg], dtype=float)

    tgt_bmaj = float(np.max(bmaj_arr))
    tgt_bmin = float(np.max(bmin_arr))
    tgt_bpa  = float(np.median(bpa_arr))
    target_idx = int(np.argmax(bmaj_arr))

    sig_tmaj = tgt_bmaj / 2.355
    sig_tmin = tgt_bmin / 2.355

    out = np.empty_like(cube, dtype=float)
    for k in range(cube.shape[0]):
        bmaj, bmin, _ = beams_deg[k]
        sig_imaj = bmaj / 2.355
        sig_imin = bmin / 2.355

        extra_maj = np.sqrt(max(sig_tmaj**2 - sig_imaj**2, 0.0)) * 2.355
        extra_min = np.sqrt(max(sig_tmin**2 - sig_imin**2, 0.0)) * 2.355
        ker = gaussian_kernel_from_extra(extra_maj, extra_min, tgt_bpa, pixscale_deg)

        out[k] = cube[k] if ker is None else convolve_fft(cube[k], ker, allow_huge=True)

    return out, (tgt_bmaj, tgt_bmin, tgt_bpa), target_idx


def alpha_two_band(S_low, S_high, nu_low, nu_high):
    return np.log(S_high / S_low) / np.log(nu_high / nu_low)


def alpha_err_two_band(S_low, S_high, nu_low, nu_high, rms_low, rms_high):
    denom = np.abs(np.log(nu_high / nu_low))
    sig_lnS1 = rms_low  / np.clip(S_low,  1e-30, None)
    sig_lnS2 = rms_high / np.clip(S_high, 1e-30, None)
    return np.sqrt(sig_lnS1**2 + sig_lnS2**2) / denom


def _guess_freq_np_axis(data, hdr):
    arr = np.asarray(data)
    naxis = int(hdr.get("NAXIS", arr.ndim))
    freq_fits_axis = find_freq_fits_axis(hdr)

    freq_np_axis = naxis - freq_fits_axis
    if arr.ndim == naxis and 0 <= freq_np_axis < arr.ndim:
        return freq_np_axis, freq_fits_axis

    cand = [i for i, s in enumerate(arr.shape) if s == 7]
    if len(cand) == 1:
        return cand[0], freq_fits_axis
    if len(cand) > 1:
        return cand[-1], freq_fits_axis
    raise ValueError(f"Cannot infer frequency numpy axis from shape={arr.shape}")


def _to_chan_y_x(data, hdr):
    arr = np.asarray(data, dtype=float)
    freq_np_axis, freq_fits_axis = _guess_freq_np_axis(arr, hdr)
    arr = np.moveaxis(arr, freq_np_axis, 0)

    while arr.ndim > 3:
        ones = [d for d in range(1, arr.ndim) if arr.shape[d] == 1]
        if not ones:
            break
        arr = np.squeeze(arr, axis=ones[0])

    if arr.ndim != 3:
        raise ValueError(f"After axis handling, expect (chan,y,x), got {arr.shape}")
    return arr, freq_fits_axis


def _auto_zoom_limits(mask, ny, nx, pad_frac=0.18, min_size=80):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, nx - 1, 0, ny - 1

    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    # ---- 强制视野为正方形（以较大边为准）----
    side = max(w, h)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * side

    xmin = int(np.floor(cx - half))
    xmax = int(np.ceil (cx + half))
    ymin = int(np.floor(cy - half))
    ymax = int(np.ceil (cy + half))

    # padding
    pad = int(np.ceil(side * pad_frac))
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    # enforce minimum window
    w2 = xmax - xmin + 1
    if w2 < min_size:
        extra = (min_size - w2) // 2 + 1
        xmin -= extra; xmax += extra
        ymin -= extra; ymax += extra  # 保持方形

    # clip to bounds，同时尽量保持方形
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(nx - 1, xmax); ymax = min(ny - 1, ymax)

    # 再次修正为方形（边界裁切可能破坏方形）
    w3 = xmax - xmin + 1
    h3 = ymax - ymin + 1
    side3 = max(w3, h3)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * side3
    xmin = int(np.floor(cx - half)); xmax = int(np.ceil(cx + half))
    ymin = int(np.floor(cy - half)); ymax = int(np.ceil(cy + half))
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(nx - 1, xmax); ymax = min(ny - 1, ymax)

    return xmin, xmax, ymin, ymax



def connected_mask_two_threshold(I_ref, rms_ref, high_sigma=3.0, low_sigma=2.0,
                                 close_iter=2, dilate_iter=6, positive_only=True):
    """
    在 low 阈值中只保留与 high 阈值相连的连通域，从而：
    - 低SNR桥/尾（2-3σ）能连上
    - 不会引入远处噪声岛
    """
    st = ndi.generate_binary_structure(2, 2)  # 8-connect

    high = np.isfinite(I_ref) & (I_ref > high_sigma * rms_ref)
    low  = np.isfinite(I_ref) & (I_ref >  low_sigma * rms_ref)

    if positive_only:
        high &= (I_ref > 0)
        low  &= (I_ref > 0)

    # 先在low里做closing让桥更容易连
    if close_iter and close_iter > 0:
        low = ndi.binary_closing(low, structure=st, iterations=int(close_iter))

    lab, nlab = ndi.label(low, structure=st)
    if nlab == 0:
        return high  # fallback

    # 只保留与high相交的label
    touch = np.unique(lab[high])
    touch = touch[touch != 0]
    if touch.size == 0:
        # fallback: 退回 high
        m = high
    else:
        m = np.isin(lab, touch)

    # 填洞 + 追边缘
    m = ndi.binary_fill_holes(m)
    if dilate_iter and dilate_iter > 0:
        m = ndi.binary_dilation(m, iterations=int(dilate_iter))

    return m


def plot_alpha(out_png, wcs2d, alpha, I_ref, rms_ref, beam_deg, pixscale_deg, zoom_mask=None):
    """
    固定三件事 + 自适应放大（推荐版本）：
    - 画布固定：figsize/dpi固定；保存不tight
    - 画框固定：主轴/色条轴用add_axes固定矩形
    - 显示窗口自适应：优先用(平滑后的)zoom_mask框住主源区域，并强制正方形窗口
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from scipy import ndimage as ndi

    # ====== 你主要调这两个来“放大/缩小视野” ======
    PAD_FRAC = 0.08      # padding占bbox的比例；越小越“放大”（建议 0.05~0.12）
    MIN_SIZE = 140       # 最小窗口像素；越小越“放大”（建议 120~180）
    FALLBACK_WIN = 180   # zoom_mask为空/无效时的固定窗口（越小越放大）

    # 固定版式（主画框/色条永远一样大）
    FRAME_RECT = (0.10, 0.12, 0.74, 0.80)   # left,bottom,width,height
    CBAR_RECT  = (0.86, 0.12, 0.035, 0.80)  # 与主画框等高

    # 风格：关键是别tight，否则不同tick文本会导致输出尺寸变化
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    })

    ny, nx = alpha.shape

    # ---------- helper：固定窗口，且边界平移保证窗口恒定 ----------
    def _fixed_window(cx, cy, win, nx, ny):
        win = int(win)
        win = min(win, nx, ny)
        if win % 2 == 1:
            win += 1
            win = min(win, nx, ny)
            if win % 2 == 1:
                win -= 1

        half = win // 2
        cx_i = int(np.round(cx))
        cy_i = int(np.round(cy))

        xmin = cx_i - half
        xmax = xmin + win - 1
        ymin = cy_i - half
        ymax = ymin + win - 1

        if xmin < 0:
            shift = -xmin
            xmin += shift
            xmax += shift
        if xmax >= nx:
            shift = xmax - (nx - 1)
            xmin -= shift
            xmax -= shift

        if ymin < 0:
            shift = -ymin
            ymin += shift
            ymax += shift
        if ymax >= ny:
            shift = ymax - (ny - 1)
            ymin -= shift
            ymax -= shift

        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(nx - 1, xmax); ymax = min(ny - 1, ymax)
        return xmin, xmax, ymin, ymax

    # ---------- helper：把任意bbox强制成“正方形窗口”，并保证不越界 ----------
    def _square_bbox(xmin, xmax, ymin, ymax, nx, ny):
        xmin = int(xmin); xmax = int(xmax); ymin = int(ymin); ymax = int(ymax)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        side = max(w, h)

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = side // 2

        xmin2 = int(np.round(cx)) - half
        xmax2 = xmin2 + side - 1
        ymin2 = int(np.round(cy)) - half
        ymax2 = ymin2 + side - 1

        # 平移回填（保持side不变）
        if xmin2 < 0:
            s = -xmin2
            xmin2 += s; xmax2 += s
        if xmax2 >= nx:
            s = xmax2 - (nx - 1)
            xmin2 -= s; xmax2 -= s
        if ymin2 < 0:
            s = -ymin2
            ymin2 += s; ymax2 += s
        if ymax2 >= ny:
            s = ymax2 - (ny - 1)
            ymin2 -= s; ymax2 -= s

        xmin2 = max(0, xmin2); ymin2 = max(0, ymin2)
        xmax2 = min(nx - 1, xmax2); ymax2 = min(ny - 1, ymax2)
        return xmin2, xmax2, ymin2, ymax2

    # =========================
    # 1) 找中心：优先 zoom_mask 内的 I_ref 峰值
    # =========================
    if zoom_mask is not None:
        zm = np.array(zoom_mask, dtype=bool) & np.isfinite(I_ref)
    else:
        zm = np.isfinite(I_ref)

    if np.any(zm):
        I_tmp = np.array(I_ref, dtype=float, copy=True)
        I_tmp[~zm] = -np.inf
        cy, cx = np.unravel_index(np.nanargmax(I_tmp), I_tmp.shape)  # (y,x)
        cx = float(cx); cy = float(cy)
    else:
        cx = 0.5 * (nx - 1)
        cy = 0.5 * (ny - 1)

    # =========================
    # 2) 自适应显示窗口（关键：放大源 + mask边界平滑）
    # =========================
    use_auto = (zoom_mask is not None) and np.any(np.array(zoom_mask, dtype=bool))

    if use_auto:
        # --- mask平滑（让bbox/边界更顺）---
        m = np.array(zoom_mask, dtype=bool)
        mf = ndi.gaussian_filter(m.astype(float), sigma=1.0) > 0.5   # sigma=0.8~1.2
        mf = ndi.binary_closing(mf, iterations=2)
        mf = ndi.binary_opening(mf, iterations=1)
        mf = ndi.binary_fill_holes(mf)

        if np.any(mf):
            ys, xs = np.where(mf)
            xmin = xs.min(); xmax = xs.max()
            ymin = ys.min(); ymax = ys.max()

            # pad
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            pad = int(np.ceil(PAD_FRAC * max(w, h)))

            xmin = max(0, xmin - pad)
            xmax = min(nx - 1, xmax + pad)
            ymin = max(0, ymin - pad)
            ymax = min(ny - 1, ymax + pad)

            # 最小窗口约束（避免特别小的源被裁得过小/太抖）
            w2 = xmax - xmin + 1
            h2 = ymax - ymin + 1
            side = max(w2, h2, int(MIN_SIZE))

            # 以 bbox 中心为中心扩展到 side，并强制正方形 + 不越界
            cx_box = 0.5 * (xmin + xmax)
            cy_box = 0.5 * (ymin + ymax)
            xmin, xmax, ymin, ymax = _fixed_window(cx_box, cy_box, side, nx, ny)
        else:
            xmin, xmax, ymin, ymax = _fixed_window(cx, cy, FALLBACK_WIN, nx, ny)
    else:
        xmin, xmax, ymin, ymax = _fixed_window(cx, cy, FALLBACK_WIN, nx, ny)

    # 强制正方形（保证画面比例一致、看起来更舒服）
    xmin, xmax, ymin, ymax = _square_bbox(xmin, xmax, ymin, ymax, nx, ny)

    # =========================
    # 3) 颜色范围：用显著发射区估计（避免噪声撑爆）
    # =========================
    levels = CONTOUR_START_SIGMA * rms_ref * (2.0 ** np.arange(CONTOUR_LEVELS_N))

    good = np.isfinite(alpha) & np.isfinite(I_ref) & (I_ref > CONTOUR_START_SIGMA * rms_ref)
    if AUTO_RANGE and np.any(good):
        vmin, vmax = np.nanpercentile(alpha[good], [PCT_LO, PCT_HI])
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = -3.0, -0.5
    else:
        vmin, vmax = -3.0, -0.5

    # 平滑显示（不改数据）
    alpha_show = alpha
    if ALPHA_SMOOTH_SIGMA and ALPHA_SMOOTH_SIGMA > 0:
        a = np.array(alpha, dtype=float, copy=True)
        bad = ~np.isfinite(a)
        a[bad] = 0.0
        w = (~bad).astype(float)
        w_s = ndi.gaussian_filter(w, sigma=ALPHA_SMOOTH_SIGMA)
        a_s = ndi.gaussian_filter(a, sigma=ALPHA_SMOOTH_SIGMA)
        alpha_show = a_s / np.clip(w_s, 1e-6, None)
        alpha_show[bad] = np.nan

    # shift到0开始
    if SHIFT_ZERO_START:
        alpha_disp = alpha_show - vmin
        vmin_plot, vmax_plot = 0.0, (vmax - vmin)
        cbar_label = r"Spectral Index"
    else:
        alpha_disp = alpha_show
        vmin_plot, vmax_plot = vmin, vmax
        cbar_label = r"Spectral Index"

    # =========================
    # 4) 固定画布 + 固定画框 + 固定色条
    # =========================
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes(FRAME_RECT, projection=wcs2d)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)

    im = ax.imshow(
        alpha_disp,
        origin="lower",
        vmin=vmin_plot,
        vmax=vmax_plot,
        cmap=ALPHA_CMAP,
        interpolation="nearest",
        zorder=1,
    )
    ax.contour(I_ref, levels=levels, colors="k", linewidths=1.0, alpha=0.95, zorder=3)

    # 固定视野（自适应窗口 + 正方形）
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # =========================
    # 5) 坐标轴标题：只用coords，别混用set_xlabel（否则会“跑很远”）
    # =========================
    ax.set_xlabel("")
    ax.set_ylabel("")
    try:
        ax.coords[0].set_axislabel("RA (J2000)", minpad=0.55)
        ax.coords[1].set_axislabel("DEC (J2000)", minpad=0.35)
        ax.coords[0].set_ticklabel(size=12, pad=2)
        ax.coords[1].set_ticklabel(size=12, pad=2)
        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)
        ax.coords[0].set_ticks(size=6)
        ax.coords[1].set_ticks(size=6)
        ax.coords[0].set_ticks(size=3, minor=True)
        ax.coords[1].set_ticks(size=3, minor=True)
    except Exception:
        ax.tick_params(axis="both", which="major", pad=2, labelsize=12, length=6, width=1.1, direction="in")

    # 色条等高且固定
    cax = fig.add_axes(CBAR_RECT)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, rotation=90, labelpad=10)
    cbar.ax.tick_params(direction="in", length=6, width=1.1, pad=3)
    cbar.outline.set_linewidth(1.2)

    # beam（左下角规整）
    bmaj, bmin, bpa = beam_deg
    bw = (bmaj / pixscale_deg)
    bh = (bmin / pixscale_deg)
    ax.add_patch(
        Ellipse(
            (xmin + 0.10 * (xmax - xmin), ymin + 0.12 * (ymax - ymin)),
            bw, bh, angle=bpa,
            transform=ax.get_transform("pixel"),
            facecolor="0.75",
            edgecolor="0.25",
            lw=1.0,
            zorder=4,
        )
    )

    # 保存：明确不tight，输出画布永远一样大
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)




def write_2d_fits(out_path, wcs2d, img2d, beam_deg, history_lines):
    hdr = wcs2d.to_header()
    bmaj, bmin, bpa = beam_deg
    hdr["BMAJ"] = float(bmaj)
    hdr["BMIN"] = float(bmin)
    hdr["BPA"]  = float(bpa)
    for s in history_lines:
        hdr.add_history(str(s).encode("ascii", "ignore").decode("ascii"))
    fits.writeto(out_path, img2d.astype("f4"), header=hdr, overwrite=True)


def process_one(in_fits):
    with fits.open(in_fits, memmap=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data

    wcs2d = WCS(hdr).celestial
    pixscale_deg = pixscale_deg_from_wcs2d(wcs2d)

    cube, freq_fits_axis = _to_chan_y_x(data, hdr)  # (chan,y,x)
    nchan, ny, nx = cube.shape

    freqs = freqs_from_header(hdr, freq_fits_axis, nchan)
    beams = read_beams_per_chan(hdr, nchan)
    cube_sm, beam_deg, target_idx = smooth_to_common_beam(cube, beams, pixscale_deg)

    I_low  = np.nanmean(cube_sm[LOW_CH], axis=0)
    I_high = np.nanmean(cube_sm[HIGH_CH], axis=0)
    nu_low  = float(np.nanmean(freqs[LOW_CH]))
    nu_high = float(np.nanmean(freqs[HIGH_CH]))

    rms_low  = rms_background(I_low)
    rms_high = rms_background(I_high)
    if not np.isfinite(rms_low) or not np.isfinite(rms_high):
        raise ValueError("background rms estimate failed (NaN)")

    # ✅ 新mask：双阈值连通（解决“连不起来”）
    mask0 = connected_mask_two_threshold(
        I_low, rms_low,
        high_sigma=HIGH_SIGMA_CONN,
        low_sigma=LOW_SIGMA_CONN,
        close_iter=CONN_CLOSE_ITER,
        dilate_iter=CONN_DILATE_ITER,
        positive_only=POSITIVE_ONLY
    )

    # 平滑I再算alpha
    I_low_use, I_high_use = I_low, I_high
    if PRE_SMOOTH_I_SIGMA and PRE_SMOOTH_I_SIGMA > 0:
        I_low_use  = ndi.gaussian_filter(I_low,  sigma=PRE_SMOOTH_I_SIGMA)
        I_high_use = ndi.gaussian_filter(I_high, sigma=PRE_SMOOTH_I_SIGMA)

    mask = mask0 & np.isfinite(I_low_use) & np.isfinite(I_high_use)
    if POSITIVE_ONLY:
        mask &= (I_low_use > 0)
    if REQUIRE_HIGH_POSITIVE_ONLY:
        mask &= (I_high_use > 0)

    alpha = np.full((ny, nx), np.nan, dtype=float)
    aerr  = np.full((ny, nx), np.nan, dtype=float)

    # noise floor
    S_low_eff  = np.maximum(I_low_use,  K_LOW_FLOOR  * rms_low)
    S_high_eff = np.maximum(I_high_use, K_HIGH_FLOOR * rms_high)

    alpha[mask] = alpha_two_band(S_low_eff[mask], S_high_eff[mask], nu_low, nu_high)
    aerr[mask]  = alpha_err_two_band(S_low_eff[mask], S_high_eff[mask], nu_low, nu_high, rms_low, rms_high)

    alpha = np.clip(alpha, ALPHA_CLIP_MIN, ALPHA_CLIP_MAX)

    if AERR_MAX is not None:
        good_err = np.isfinite(aerr) & (aerr <= float(AERR_MAX))
        alpha[~good_err] = np.nan
        aerr[~good_err]  = np.nan

    # zoom_mask：优先用有效alpha，否则用 I_low > 3σ
    if ZOOM_USE_ALPHA:
        zoom_mask = np.isfinite(alpha)
        if zoom_mask.sum() < 50:
            zoom_mask = np.isfinite(I_low) & (I_low > CONTOUR_START_SIGMA * rms_low)
    else:
        zoom_mask = np.isfinite(I_low) & (I_low > CONTOUR_START_SIGMA * rms_low)

    base = os.path.basename(in_fits).replace(".fits", "")
    out_alpha = os.path.join(OUT_DIR, base + "_alpha_2band.fits")
    out_aerr  = os.path.join(OUT_DIR, base + "_alphaerr_2band.fits")
    out_png   = os.path.join(OUT_DIR, base + "_alpha_2band.png")

    hist = [
        "Two-band spectral index map from 7ch cube",
        "Definition: S_nu ∝ nu^alpha",
        f"LOW_CH={LOW_CH} HIGH_CH={HIGH_CH}",
        f"nu_low={nu_low/1e6:.3f} MHz nu_high={nu_high/1e6:.3f} MHz",
        f"rms_low={rms_low:.6g} rms_high={rms_high:.6g}",
        f"mask(conn): high={HIGH_SIGMA_CONN}σ low={LOW_SIGMA_CONN}σ close={CONN_CLOSE_ITER} dilate={CONN_DILATE_ITER}",
        f"FIGSIZE={FIGSIZE} DPI={DPI} CBAR_RECT={CBAR_RECT}",
    ]

    write_2d_fits(out_alpha, wcs2d, alpha, beam_deg, hist + ["alpha = ln(S_high/S_low)/ln(nu_high/nu_low)"])
    write_2d_fits(out_aerr,  wcs2d, aerr,  beam_deg, hist + ["alphaerr from propagation using rms_low/rms_high"])

    if SAVE_PNG:
        plot_alpha(out_png, wcs2d, alpha, I_low, rms_low, beam_deg, pixscale_deg, zoom_mask=zoom_mask)

    print(f"[OK] {in_fits}")
    print(f"     -> {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(IN_GLOB))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {IN_GLOB}")
    for f in files:
        try:
            process_one(f)
        except Exception as e:
            print(f"[FAIL] {f}: {e}")


if __name__ == "__main__":
    main()
