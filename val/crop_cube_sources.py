#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from tqdm import tqdm

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u


# ===================== 用户配置 =====================
CATALOG_FITS = "/home/caojie/work/Galaxy-Morphology/bent_catalog/BT_final_catalog.fits"
CUBE_GLOB = "/shared/main/meerkat/OTF/images/*.cube.int.restored.fits"
OUT_DIR = "/shared/main/caojie/meerkat/cube_bent_new"
CUTOUT_SIZE = 300  # 256x256
PAD_VALUE = np.nan  # 如果源靠边裁剪越界，是否补 pad（NaN）。不想 pad 可改为 None 并强制跳过越界源
# ====================================================


def pick_col(table, candidates):
    """从 astropy Table / recarray 的列名里，按候选列表挑一个存在的列名。"""
    names = [n for n in table.names]
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        if c in names:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def normalize_id(val, fallback_idx):
    """生成文件名友好的 source id。"""
    if val is None:
        return f"src_{fallback_idx:06d}"
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return f"src_{fallback_idx:06d}"
    # 文件名安全
    s = s.replace(" ", "_").replace("/", "_")
    return s


def get_spatial_wcs(hdr):
    """
    从 cube header 里取 WCS，并确保我们有可用于天球->像素的 celestial WCS。
    cube 通常是 (chan, y, x) 或 (stokes, chan, y, x) 等。
    这里使用 WCS(hdr).celestial 来专注 RA/DEC 两轴。
    """
    w = WCS(hdr)
    wc = w.celestial
    if wc.pixel_n_dim != 2:
        raise RuntimeError("Celestial WCS 不是 2 维，header 可能异常。")
    return w, wc


def cube_shape_info(data):
    """
    统一 cube 数据维度为至少 3D，并假定最后两维是 (y, x)。
    例如常见 shape:
      (nchan, ny, nx)
      (nstokes, nchan, ny, nx)
    这里不改变轴顺序，只裁剪最后两维。
    """
    if data.ndim < 3:
        raise ValueError(f"Cube data ndim={data.ndim} < 3，不像是 cube。shape={data.shape}")
    ny, nx = data.shape[-2], data.shape[-1]
    return ny, nx


def pixel_in_bounds(x, y, nx, ny):
    return (x >= 0) and (x < nx) and (y >= 0) and (y < ny)


def crop_last2d(arr, x0, x1, y0, y1, pad_value=np.nan):
    """
    对任意 ndim 数组裁剪最后两维 [y0:y1, x0:x1]。
    若越界并 pad_value!=None，则进行 pad 到目标大小。
    """
    target_h = y1 - y0
    target_w = x1 - x0

    ny, nx = arr.shape[-2], arr.shape[-1]
    iy0 = max(y0, 0)
    ix0 = max(x0, 0)
    iy1 = min(y1, ny)
    ix1 = min(x1, nx)

    cropped = arr[..., iy0:iy1, ix0:ix1]

    # 完全在界内
    if (iy0 == y0) and (ix0 == x0) and (iy1 == y1) and (ix1 == x1):
        return cropped

    if pad_value is None:
        # 不允许越界，返回 None 表示跳过
        return None

    # 需要 pad 到 (target_h, target_w)
    out_shape = list(arr.shape)
    out_shape[-2] = target_h
    out_shape[-1] = target_w
    out = np.full(out_shape, pad_value, dtype=arr.dtype)

    oy0 = iy0 - y0
    ox0 = ix0 - x0
    oy1 = oy0 + (iy1 - iy0)
    ox1 = ox0 + (ix1 - ix0)

    out[..., oy0:oy1, ox0:ox1] = cropped
    return out


def update_header_for_cutout(hdr, x0, y0):
    """
    更新 header 的 CRPIX1/CRPIX2，使裁剪后 WCS 仍正确。
    注意：FITS header 的 CRPIX 是 1-based。
    我们裁剪是对像素索引(0-based)做了平移：新图像的(0,0)对应原图的(x0,y0)。
    因此：CRPIX_new = CRPIX_old - x0/y0
    """
    new_hdr = hdr.copy()

    # 只对空间轴 CRPIX1/2 生效（通常 1->X, 2->Y）
    # 如果你的 cube header 空间轴不是 1/2（不常见），再按实际情况改。
    if "CRPIX1" in new_hdr:
        new_hdr["CRPIX1"] = new_hdr["CRPIX1"] - x0
    if "CRPIX2" in new_hdr:
        new_hdr["CRPIX2"] = new_hdr["CRPIX2"] - y0

    # 更新 NAXIS1/2
    new_hdr["NAXIS1"] = CUTOUT_SIZE
    new_hdr["NAXIS2"] = CUTOUT_SIZE

    # 记录裁剪信息（不影响你说的“保留信息”，只是附加 HISTORY）
    new_hdr.add_history(f"CUTOUT: spatial crop {CUTOUT_SIZE}x{CUTOUT_SIZE}")
    new_hdr.add_history(f"CUTOUT: origin x0={x0}, y0={y0} (0-based in original image)")
    return new_hdr


def cube_covers_coord(cel_wcs, coord, nx, ny):
    """判断该坐标是否落在 cube 的空间范围内（只做像素范围判断）。"""
    x, y = skycoord_to_pixel(coord, cel_wcs, origin=0)
    if np.isnan(x) or np.isnan(y):
        return False, None, None
    if pixel_in_bounds(x, y, nx, ny):
        return True, float(x), float(y)
    return False, float(x), float(y)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 读星表 ----------
    with fits.open(CATALOG_FITS, memmap=True) as hdul:
        tab = hdul[1].data

        # 自动识别坐标列
        ra_col = pick_col(hdul[1].columns, ["RA", "ra", "RAJ2000", "RA_ICRS", "ALPHA_J2000"])
        dec_col = pick_col(hdul[1].columns, ["DEC", "dec", "DEJ2000", "DEC_ICRS", "DELTA_J2000"])

        if ra_col is None or dec_col is None:
            raise RuntimeError(
                f"找不到 RA/DEC 列。请确认 final.fits 的列名。"
                f"当前列名示例：{hdul[1].columns.names[:30]}"
            )

        # source id 列（可选）
        id_col = pick_col(hdul[1].columns, ["source_id", "SOURCE_ID", "id", "ID", "Name", "NAME"])

        ra = np.array(tab[ra_col], dtype=float)
        dec = np.array(tab[dec_col], dtype=float)
        src_ids = tab[id_col] if id_col is not None else [None] * len(ra)

    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # ---------- 收集 cubes ----------
    cube_paths = sorted(glob.glob(CUBE_GLOB))
    if not cube_paths:
        raise FileNotFoundError(f"没有匹配到 cube：{CUBE_GLOB}")

    # 预读 cube 的 WCS/尺寸（加速：先建索引）
    cube_meta = []
    print(f"[INFO] Found {len(cube_paths)} cubes. Indexing WCS footprints...")
    for p in tqdm(cube_paths, desc="Index cubes"):
        with fits.open(p, memmap=True) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            if data is None:
                continue
            ny, nx = cube_shape_info(data)
            _, cel = get_spatial_wcs(hdr)
            cube_meta.append((p, nx, ny, cel))

    if not cube_meta:
        raise RuntimeError("没有可用的 cube（可能 fits 无 data）。")

    half = CUTOUT_SIZE // 2

    # ---------- 对每个源裁剪 ----------
    n_done = 0
    n_skip = 0

    for i, coord in tqdm(list(enumerate(coords)), total=len(coords), desc="Crop sources"):
        sid = normalize_id(src_ids[i] if src_ids is not None else None, i)

        # 找到第一个覆盖该源的 cube
        chosen = None
        for (p, nx, ny, cel_wcs) in cube_meta:
            ok, x, y = cube_covers_coord(cel_wcs, coord, nx, ny)
            if ok:
                chosen = (p, nx, ny, x, y)
                break

        if chosen is None:
            n_skip += 1
            continue

        cube_path, nx, ny, x, y = chosen

        # 以该像素为中心裁 256x256
        xc = int(np.round(x))
        yc = int(np.round(y))

        x0 = xc - half
        x1 = x0 + CUTOUT_SIZE
        y0 = yc - half
        y1 = y0 + CUTOUT_SIZE

        with fits.open(cube_path, memmap=True) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data

            cropped = crop_last2d(data, x0, x1, y0, y1, pad_value=PAD_VALUE)
            if cropped is None:
                n_skip += 1
                continue

            new_hdr = update_header_for_cutout(hdr, x0, y0)

            # 输出命名：源id + cube名
            cube_base = os.path.basename(cube_path).replace(".fits", "")
            out_name = f"{sid}__{cube_base}__cut{CUTOUT_SIZE}.fits"
            out_path = os.path.join(OUT_DIR, out_name)

            fits.writeto(out_path, cropped, header=new_hdr, overwrite=True)

        n_done += 1

    print(f"[DONE] Saved: {n_done} cutouts -> {OUT_DIR}")
    print(f"[DONE] Skipped (no cube / out of bounds): {n_skip}")


if __name__ == "__main__":
    main()
