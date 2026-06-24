#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
from typing import Tuple, Optional

import numpy as np
import requests
from astropy.table import Table
from astropy.io import fits

# ===================== 你给的路径 =====================
CAT_PATH = "/home/caojie/work/Galaxy-Morphology/bent_catalog/subcatalog.fits"
OUT_DIR  = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/wise_new"
# =====================================================

# ===================== SkyView / WISE 配置 =====================
SURVEY = "WISE 3.4"         # W1
PIXELS = 300               # 300x300

# WISE Atlas 图像常用像素尺度 ~ 1.375 arcsec/pix
PIX_SCALE_ARCSEC = 1.375
SIZE_DEG = (PIXELS * PIX_SCALE_ARCSEC) / 3600.0  # cutout 宽度(度)

# SkyView CGI
SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"

# 下载重试
TIMEOUT = 60
RETRIES = 4
SLEEP_BETWEEN = 2.0
# =====================================================


def _find_col(table: Table, candidates) -> Optional[str]:
    """在表里寻找可能的列名（大小写不敏感），返回实际列名。"""
    lower_map = {c.lower(): c for c in table.colnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _get_ra_dec_cols(tab: Table) -> Tuple[str, str]:
    """
    尽量自动识别 RA/DEC 列名。
    常见：RA, DEC, RAJ2000, DEJ2000, ra, dec, etc.
    """
    ra_col = _find_col(tab, ["ra", "raj2000", "ra_deg", "alpha_j2000", "ra_icrs"])
    dec_col = _find_col(tab, ["dec", "dej2000", "dec_deg", "delta_j2000", "dec_icrs"])
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"没找到 RA/DEC 列。现有列名：{tab.colnames}\n"
            "请确认你的星表里 RA/DEC 列名是什么，然后把 candidates 里加上。"
        )
    return ra_col, dec_col


def _get_id_col(tab: Table) -> str:
    """识别 source_id 列名。"""
    id_col = _find_col(tab, ["source_id", "sourceid", "id", "Source_id", "SOURCE_ID"])
    if id_col is None:
        raise ValueError(
            f"没找到 source_id 列。现有列名：{tab.colnames}\n"
            "如果你的 ID 列名不是 source_id，请把 _get_id_col 里 candidates 改成你的列名。"
        )
    return id_col


def _safe_filename(s: str) -> str:
    """确保文件名安全（只保留字母数字、下划线、短横线、点）。"""
    s = str(s).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s


def download_skyview_wise_fits(ra_deg: float, dec_deg: float, out_fits: str) -> bool:
    """
    通过 SkyView CGI 下载 WISE 3.4 FITS。
    成功返回 True，失败返回 False。
    """
    params = {
        "Position": f"{ra_deg},{dec_deg}",
        "Survey": SURVEY,
        "Return": "FITS",
        "Coordinates": "J2000",
        "pixels": str(PIXELS),
        "Size": f"{SIZE_DEG}",
        # 可选：指定投影/坐标系等
        # "Projection": "Tan",
    }

    # SkyView 有时返回的是一个包含 FITS 的 tar 或多文件结果；
    # 我们做：优先直接把响应当 FITS 打开验证；若不是 FITS，再尝试解析出 fits 文件链接。
    session = requests.Session()

    for k in range(1, RETRIES + 1):
        try:
            r = session.get(SKYVIEW_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            content = r.content

            # 1) 直接判断是否是 FITS（FITS 前 30 字节通常是 'SIMPLE  =                    T'）
            if content[:6] == b"SIMPLE":
                os.makedirs(os.path.dirname(out_fits), exist_ok=True)
                with open(out_fits, "wb") as f:
                    f.write(content)
                # 验证能否打开
                with fits.open(out_fits, memmap=False) as hdul:
                    _ = hdul[0].data
                return True

            # 2) 如果不是 FITS，尝试从 HTML 里找 .fits 链接（SkyView 常返回结果页）
            text = r.text
            # 匹配 href="...fits"
            m = re.findall(r'href="([^"]+\.fits[^"]*)"', text, flags=re.IGNORECASE)
            if m:
                fits_url = m[0]
                # 处理相对路径
                if fits_url.startswith("/"):
                    fits_url = "https://skyview.gsfc.nasa.gov" + fits_url
                elif not fits_url.lower().startswith("http"):
                    fits_url = "https://skyview.gsfc.nasa.gov/current/cgi/" + fits_url.lstrip("./")

                r2 = session.get(fits_url, timeout=TIMEOUT)
                r2.raise_for_status()
                content2 = r2.content
                if content2[:6] != b"SIMPLE":
                    raise RuntimeError("解析到的链接下载后仍不是 FITS。")

                os.makedirs(os.path.dirname(out_fits), exist_ok=True)
                with open(out_fits, "wb") as f:
                    f.write(content2)
                with fits.open(out_fits, memmap=False) as hdul:
                    _ = hdul[0].data
                return True

            raise RuntimeError("SkyView 返回内容不是 FITS，也没在结果页找到 FITS 链接。")

        except Exception as e:
            if k == RETRIES:
                print(f"[FAIL] {out_fits}  RA={ra_deg} DEC={dec_deg}  err={e}")
                return False
            time.sleep(SLEEP_BETWEEN * k)

    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tab = Table.read(CAT_PATH)
    ra_col, dec_col = _get_ra_dec_cols(tab)
    id_col = _get_id_col(tab)

    print(f"[INFO] catalog: {CAT_PATH}")
    print(f"[INFO] rows   : {len(tab)}")
    print(f"[INFO] RA col : {ra_col}")
    print(f"[INFO] DEC col: {dec_col}")
    print(f"[INFO] ID col : {id_col}")
    print(f"[INFO] survey : {SURVEY}")
    print(f"[INFO] pixels : {PIXELS}x{PIXELS}")
    print(f"[INFO] size   : {SIZE_DEG:.6f} deg (~{SIZE_DEG*60:.3f} arcmin)")

    ok = 0
    skip = 0
    fail = 0

    for i, row in enumerate(tab):
        sid = _safe_filename(row[id_col])
        ra = float(row[ra_col])
        dec = float(row[dec_col])

        out_fits = os.path.join(OUT_DIR, f"{sid}.fits")
        if os.path.exists(out_fits) and os.path.getsize(out_fits) > 0:
            skip += 1
            continue

        success = download_skyview_wise_fits(ra, dec, out_fits)
        if success:
            ok += 1
            if ok % 10 == 0:
                print(f"[INFO] downloaded {ok} / {len(tab)}")
        else:
            fail += 1

    print(f"[DONE] ok={ok}, skip={skip}, fail={fail}, total={len(tab)}")
    print(f"[OUT ] {OUT_DIR}")


if __name__ == "__main__":
    main()
