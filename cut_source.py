#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cut MeerKLASS FITS mosaics into 128x128 island-centered cutouts and save to HDF5.
- One FITS -> one H5
- Each island -> one group (data_0, data_1, ...) with dataset Img
- Save FITS header + island RA/DEC as attributes (like your example)

Catalog columns (as you provided):
  Source_Name (string)
  Source_id   (double)
  Isl_id      (double)
  RA          (double)
  DEC         (double)
"""

import os
import glob
import json
import warnings

import numpy as np
import h5py
from tqdm import tqdm

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Config
# =========================
FITS_GLOB = "/home/caojie/work/Galaxy-Morphology/bent_catalog/data/*.fits"
CATALOG_FITS = "/home/caojie/work/Galaxy-Morphology/bent_catalog/MeerKLASS_Lband_Catalogue_DR1_v24Jul2025.fits"
OUT_DIR = "/shared/main/caojie/meerkat/source_crop_all/"
CROP_SIZE = 128
FILL_VALUE = 0.0

# Fixed column names (per your screenshot)
COL_SOURCE_NAME = "Source_Name"
COL_SOURCE_ID = "Source_id"
COL_ISLAND_ID = "Isl_id"
COL_RA = "RA"
COL_DEC = "DEC"

os.makedirs(OUT_DIR, exist_ok=True)


def header_to_string(header: fits.Header) -> str:
    return header.tostring(sep="\n", endcard=True, padding=True)


def read_fits_2d(fits_path: str):
    """Return (data2d, header, wcs_celestial)."""
    with fits.open(fits_path, memmap=True) as hdul:
        hdu = hdul[0]
        header = hdu.header.copy()
        data = hdu.data

    if data is None:
        raise RuntimeError(f"No data in FITS: {fits_path}")

    data = np.asarray(data)
    data2d = np.squeeze(data)
    if data2d.ndim != 2:
        raise RuntimeError(f"Expected 2D after squeeze, got {data2d.shape} for {fits_path}")

    w = WCS(header).celestial
    return data2d, header, w


def catalog_rows_in_image(tab: Table, wcs_cel: WCS, nx: int, ny: int, margin: int):
    """Mask rows roughly inside image footprint (+margin for cutouts)."""
    ra = np.array(tab[COL_RA], dtype=float)
    dec = np.array(tab[COL_DEC], dtype=float)
    x, y = wcs_cel.world_to_pixel_values(ra, dec)

    x_min, y_min = -margin, -margin
    x_max, y_max = (nx - 1) + margin, (ny - 1) + margin
    ok = np.isfinite(x) & np.isfinite(y) & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return ok


def island_center(tab_island: Table):
    """
    Your requirement: use island id -> coordinate center.
    Catalog doesn't show island-center columns, so we use median of member RA/DEC.
    """
    ra = np.array(tab_island[COL_RA], dtype=float)
    dec = np.array(tab_island[COL_DEC], dtype=float)
    return float(np.nanmedian(ra)), float(np.nanmedian(dec))


def write_h5_for_fits(fits_path: str, out_h5_path: str, tab: Table):
    data2d, header, wcs_cel = read_fits_2d(fits_path)
    ny, nx = data2d.shape

    ok = catalog_rows_in_image(tab, wcs_cel, nx, ny, margin=CROP_SIZE // 2)
    tab_in = tab[ok]

    # Create h5 even if empty
    with h5py.File(out_h5_path, "w") as f:
        f.attrs["fits_path"] = fits_path
        f.attrs["fits_shape"] = json.dumps({"ny": int(ny), "nx": int(nx)})
        f.attrs["fits_header"] = header_to_string(header)

        if len(tab_in) == 0:
            f.attrs["num_groups"] = 0
            return

        island_ids = np.array(tab_in[COL_ISLAND_ID])
        # unique islands (stable)
        _, first_idx = np.unique(island_ids, return_index=True)
        unique_islands = island_ids[np.sort(first_idx)]

        group_count = 0

        for isl in tqdm(unique_islands, desc=f"Cropping {os.path.basename(fits_path)}", leave=False):
            tab_isl = tab_in[island_ids == isl]
            if len(tab_isl) == 0:
                continue

            ra_c, dec_c = island_center(tab_isl)
            if not (np.isfinite(ra_c) and np.isfinite(dec_c)):
                continue

            x_c, y_c = wcs_cel.world_to_pixel_values(ra_c, dec_c)
            if not (np.isfinite(x_c) and np.isfinite(y_c)):
                continue

            # cutout
            try:
                cut = Cutout2D(
                    data2d,
                    position=(x_c, y_c),
                    size=(CROP_SIZE, CROP_SIZE),
                    mode="partial",
                    fill_value=FILL_VALUE,
                )
                cut_img = np.array(cut.data, dtype=np.float32)
                if cut_img.shape != (CROP_SIZE, CROP_SIZE):
                    tmp = np.full((CROP_SIZE, CROP_SIZE), FILL_VALUE, dtype=np.float32)
                    sy, sx = cut_img.shape
                    tmp[:sy, :sx] = cut_img
                    cut_img = tmp
            except Exception:
                continue

            # group
            gname = f"data_{group_count}"
            g = f.create_group(gname)

            g.create_dataset(
                "Img",
                data=cut_img,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            # --- Attributes (match your example style) ---
            g.attrs["RA"] = float(ra_c)
            g.attrs["DEC"] = float(dec_c)

            # Isl_id / Source_id in catalog are double -> cast to int safely when possible
            # (if some are non-integer-like, we keep as float)
            try:
                isl_int = int(round(float(isl)))
                if abs(float(isl) - isl_int) < 1e-6:
                    g.attrs["Isl_id"] = isl_int
                else:
                    g.attrs["Isl_id"] = float(isl)
            except Exception:
                g.attrs["Isl_id"] = float(isl)

            # Member source ids
            src_ids = np.array(tab_isl[COL_SOURCE_ID], dtype=float)
            # store as int64 when they are effectively integers
            src_int = np.rint(src_ids).astype(np.int64)
            if np.all(np.isfinite(src_ids)) and np.all(np.abs(src_ids - src_int) < 1e-6):
                g.create_dataset("Source_id", data=src_int, compression="gzip", compression_opts=4, shuffle=True)
            else:
                g.create_dataset("Source_id", data=src_ids.astype(np.float64), compression="gzip", compression_opts=4, shuffle=True)

            # Source_Name (may be multiple within one island)
            if COL_SOURCE_NAME in tab_isl.colnames:
                names = np.array(tab_isl[COL_SOURCE_NAME]).astype("S")
                g.create_dataset("Source_Name", data=names, compression="gzip", compression_opts=4, shuffle=True)
                # also store first one as quick attr
                g.attrs["Source_Name_first"] = names[0].decode("utf-8", errors="ignore") if len(names) > 0 else ""

            g.attrs["num_sources_in_island"] = int(len(tab_isl))
            g.attrs["fits_path"] = fits_path

            # cutout WCS header (optional but very useful)
            try:
                cut_wcs_header = cut.wcs.to_header()
                g.attrs["cutout_wcs_header"] = cut_wcs_header.tostring(sep="\n", endcard=True, padding=True)
            except Exception:
                pass

            group_count += 1

        f.attrs["num_groups"] = int(group_count)


def main():
    fits_list = sorted(glob.glob(FITS_GLOB))
    if len(fits_list) == 0:
        raise RuntimeError(f"No FITS found: {FITS_GLOB}")

    print(f"[INFO] Found FITS: {len(fits_list)}")
    print(f"[INFO] Reading catalog: {CATALOG_FITS}")

    tab = Table.read(CATALOG_FITS)

    # hard check required columns
    for c in [COL_SOURCE_ID, COL_ISLAND_ID, COL_RA, COL_DEC]:
        if c not in tab.colnames:
            raise RuntimeError(f"Missing required column '{c}' in catalog. Available: {tab.colnames[:60]}")

    for fp in tqdm(fits_list, desc="Processing FITS"):
        base = os.path.splitext(os.path.basename(fp))[0]
        out_h5 = os.path.join(OUT_DIR, f"{base}.h5")

        try:
            write_h5_for_fits(fp, out_h5, tab)
        except Exception as e:
            print(f"[WARN] Failed {fp}: {e}")

    print("[DONE] Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
