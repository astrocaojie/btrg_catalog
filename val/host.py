#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
host.py
WISE 灰度底图 + 射电等值线叠加 + 绿圈标记（用 demo 中心点）
要求：
- 低阈值 low=2.0σ 用来“连通”
- 只保留与主源（射电峰值）连通的那一块，去掉噪声岛
- 等值线仍从 3σ 开始（更干净），并额外画一条“主连通体边界”保证连起来
- 不改变像素比例：interpolation="nearest" + ax.set_aspect("equal")
- 终端环境保存 PNG，不用 plt.show()

依赖：
pip/conda: astropy, reproject, scipy, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize
from reproject import reproject_interp

from scipy.ndimage import label, binary_opening, binary_closing, generate_binary_structure


# =======================
# 配置区：改这里就行
# =======================
WISE_FITS  = "wise_demo_4.6.fits"
RADIO_FITS = "demo.fits"

# ✅ 用 demo 的中心点画绿圈（你给的）
HOST_WCS_STR = ("23h28m48.9s", "-35d08m18s")  # (RA, Dec)

# 低阈值：用于“连通”（你要 2.0）
LOW_SIGMA = 2.0

# 干净等值线起始阈值（保持论文风格）
CONTOUR_START_SIGMA = 3.0
CONTOUR_LEVELS_N = 10
CONTOUR_FACTOR = np.sqrt(2)

# 出图
OUT_PNG = "host_overlay.png"
DPI = 300
# =======================


def squeeze_to_2d(data: np.ndarray) -> np.ndarray:
    """把 (1,1,ny,nx) / (ny,nx,1,1) / (1,ny,nx) 等压到 2D"""
    d = data
    d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"Cannot squeeze to 2D, got shape={data.shape} -> {d.shape}")
    return d.astype(np.float32)


def robust_rms_mad(x: np.ndarray) -> float:
    """robust sigma via MAD"""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def estimate_sigma_edge(img: np.ndarray, border_frac: float = 0.25) -> float:
    """用边缘区域估 sigma（更稳）"""
    h, w = img.shape
    b = max(1, int(min(h, w) * border_frac))
    bg = np.zeros_like(img, dtype=bool)
    bg[:b, :] = True
    bg[-b:, :] = True
    bg[:, :b] = True
    bg[:, -b:] = True
    return robust_rms_mad(img[bg])


def main():
    # ====== 读 WISE ======
    with fits.open(WISE_FITS) as hdul:
        wise_data = hdul[0].data
        wise_wcs = WCS(hdul[0].header).celestial
    wise_img = squeeze_to_2d(wise_data)

    # ====== 读 RADIO（demo.fits 可能是 4D）======
    with fits.open(RADIO_FITS) as hdul:
        radio_data = hdul[0].data
        radio_wcs = WCS(hdul[0].header).celestial
    radio_img = squeeze_to_2d(radio_data)

    # ====== 把 RADIO reproject 到 WISE 网格 ======
    radio_on_wise, _ = reproject_interp(
        (radio_img, radio_wcs),
        wise_wcs,
        shape_out=wise_img.shape
    )

    # ====== 估 sigma ======
    sigma = estimate_sigma_edge(radio_on_wise, border_frac=0.25)
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("sigma 估计失败（得到 NaN/非正数）。请检查 radio_on_wise 数据。")

    print(f"[WISE] shape={wise_img.shape}")
    print(f"[RADIO] shape(2D)={radio_img.shape}")
    print(f"[RADIO] sigma ~ {sigma:.6e}")

    # ====== 低阈值连通体：只保留与峰值连通的主源 ======
    low = LOW_SIGMA * sigma
    finite = np.isfinite(radio_on_wise)
    m_low = finite & (radio_on_wise > low)

    # 形态学：去掉细碎噪声点（不建议太强，防止吃尾巴）
    st = generate_binary_structure(2, 2)  # 8连通
    m_low = binary_opening(m_low, structure=st, iterations=1)
    m_low = binary_closing(m_low, structure=st, iterations=1)

    # 用峰值像素确定主连通体
    py, px = np.unravel_index(np.nanargmax(radio_on_wise), radio_on_wise.shape)
    lab, nlab = label(m_low, structure=st)
    main_id = lab[py, px]
    if main_id == 0:
        # 峰值竟然不在 m_low 内：说明 low 设太高或数据异常
        raise RuntimeError(
            f"主连通体提取失败：峰值不在 low={LOW_SIGMA}σ 的mask内。\n"
            f"建议：检查 sigma 或把 LOW_SIGMA 适当降低（但你指定了 2.0）。"
        )
    m_main = (lab == main_id)

    # 用主连通体限制等值线：去掉所有噪声岛
    radio_main = np.where(m_main, radio_on_wise, np.nan)

    # ====== 干净等值线：从 3σ 起，√2 递增 ======
    start = CONTOUR_START_SIGMA * sigma
    levels = start * (CONTOUR_FACTOR ** np.arange(CONTOUR_LEVELS_N))

    # ====== 宿主/中心点（用 demo 中心点坐标画绿圈）======
    host = SkyCoord(HOST_WCS_STR[0], HOST_WCS_STR[1], frame=FK5(equinox="J2000"))

    # ====== 画图 ======
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(projection=wise_wcs)

    norm = ImageNormalize(wise_img, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(
        wise_img,
        origin="lower",
        cmap="gray",
        norm=norm,
        interpolation="nearest"  # ✅ 不糊、不“变形感”
    )

    # 1) 主源干净等值线（3σ 起）
    ax.contour(
        radio_main,
        levels=levels,
        colors="#ff00ff",
        linewidths=1.2,
        alpha=0.95
    )

    # 2) 额外加一条“主连通体边界”，保证视觉连通（不引入噪声岛）
    ax.contour(
        m_main.astype(float),
        levels=[0.5],
        colors="#ff00ff",
        linewidths=1.6,
        alpha=0.95
    )

    # 绿圈（demo中心点）
    ax.scatter(
        host.ra.deg, host.dec.deg,
        transform=ax.get_transform("world"),
        s=160,
        facecolors="none",
        edgecolors="lime",
        linewidths=2.6,
        zorder=10
    )

    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.grid(color="white", alpha=0.25, linestyle="--", linewidth=0.6)

    # ✅ 保持像素比例（看起来不被拉伸）
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=DPI)
    print(f"[OK] Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
