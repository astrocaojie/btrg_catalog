#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


CATALOG_FITS = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog.fits"
DATA_GLOB    = "/home/caojie/work/Galaxy-Morphology/bent_catalog/data/*.fits"
OUT_DIR      = "/shared/main/caojie/meerkat/bent_crop_100/"
CUT_NX = 100
CUT_NY = 100


def pick_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def sanitize_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    return s


def ra_wrap_bounds(ra_deg_array):
    ra = np.asarray(ra_deg_array) % 360.0
    ra_min = ra.min()
    ra_max = ra.max()
    span = ra_max - ra_min
    if span > 180.0:
        ra2 = ((ra + 180.0) % 360.0) - 180.0
        return ra2.min(), ra2.max(), True
    return ra_min, ra_max, False


def ra_in_range(ra_deg, ra_min, ra_max, use_wrap):
    ra = (ra_deg % 360.0)
    if use_wrap:
        ra = ((ra + 180.0) % 360.0) - 180.0
    return (ra >= ra_min) & (ra <= ra_max)


def cut_last2_axes(data, xc, yc, nx=CUT_NX, ny=CUT_NY, fill_value=np.nan):
    """
    Cut a (ny,nx) window centered at (xc,yc) from the last two axes of `data`.
    Works for data with ndim >= 2. Keeps leading axes unchanged.
    Returns (cut_data, x0, y0) where x0,y0 are the lower-left corner in the original array.
    """
    if data.ndim < 2:
        raise ValueError("data.ndim < 2, cannot cut")

    H = data.shape[-2]
    W = data.shape[-1]

    halfx = nx // 2
    halfy = ny // 2

    # integer center
    xc_i = int(np.round(xc))
    yc_i = int(np.round(yc))

    x0 = xc_i - halfx
    x1 = x0 + nx
    y0 = yc_i - halfy
    y1 = y0 + ny

    # destination array filled
    out_shape = data.shape[:-2] + (ny, nx)
    out = np.full(out_shape, fill_value, dtype=data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32)

    # overlap region in original
    ox0 = max(x0, 0)
    ox1 = min(x1, W)
    oy0 = max(y0, 0)
    oy1 = min(y1, H)

    if (ox1 <= ox0) or (oy1 <= oy0):
        return out, x0, y0  # totally outside, keep fill

    # overlap region in output
    tx0 = ox0 - x0
    tx1 = tx0 + (ox1 - ox0)
    ty0 = oy0 - y0
    ty1 = ty0 + (oy1 - oy0)

    # slice copy
    src_slices = (Ellipsis, slice(oy0, oy1), slice(ox0, ox1))
    dst_slices = (Ellipsis, slice(ty0, ty1), slice(tx0, tx1))
    out[dst_slices] = data[src_slices]

    return out, x0, y0


def update_celestial_wcs_in_header(hdr, x0, y0):
    """
    Update header celestial WCS for a cutout that starts at (x0,y0) in original pixel coords
    on the last two axes.
    We keep everything else untouched, but adjust CRPIX1/2 and NAXIS1/2.
    """
    out_hdr = hdr.copy()

    # If there is celestial WCS, CRPIX1/2 refer to the last two axes in standard images.
    # For many radio images, celestial axes are indeed 1/2. This is usually true for 2D images.
    # For N-D, CRPIXn exist for each axis; celestial usually maps to two of them.
    # The safest: use astropy WCS to find which pixel axes correspond to celestial.
    w_full = WCS(hdr)
    w_cel = w_full.celestial

    # find mapping from celestial WCS to full WCS pixel axes
    # pixel axis numbers in FITS are 1-based
    # w_full.axis_type_names length = naxis
    # w_full.world_axis_physical_types gives mapping but can be None; we use w_full.celestial.pixel_axis_names?
    # Practical robust approach: assume the celestial are the last two axes of data AND correspond to CRPIX1/2
    # in these MeerKAT continuum images. If your header uses other axis ordering, tell me and we refine.
    if "CRPIX1" in out_hdr:
        out_hdr["CRPIX1"] = out_hdr["CRPIX1"] - x0
    if "CRPIX2" in out_hdr:
        out_hdr["CRPIX2"] = out_hdr["CRPIX2"] - y0

    return out_hdr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cat = Table.read(CATALOG_FITS)

    ra_col = pick_column(cat.colnames, ["RA", "RAJ2000", "RA_deg", "ALPHA_J2000", "RA_ICRS", "ra"])
    dec_col = pick_column(cat.colnames, ["DEC", "DEJ2000", "DEC_deg", "DELTA_J2000", "DEC_ICRS", "dec"])
    id_col = pick_column(cat.colnames, ["source_id", "SOURCE_ID", "id", "ID", "Name", "name"])

    if ra_col is None or dec_col is None:
        raise RuntimeError(f"Cannot find RA/DEC in catalog columns: {cat.colnames}")

    if id_col is None:
        print("[WARN] No source_id column found; use row index for filenames.")
        ids = np.array([f"row{idx:06d}" for idx in range(len(cat))], dtype=object)
    else:
        ids = np.array([sanitize_filename(x) for x in cat[id_col]], dtype=object)

    ra_deg  = np.array(cat[ra_col], dtype=float)
    dec_deg = np.array(cat[dec_col], dtype=float)

    written = np.zeros(len(cat), dtype=bool)

    fits_files = sorted(glob.glob(DATA_GLOB))
    if not fits_files:
        raise RuntimeError(f"No FITS matched: {DATA_GLOB}")

    print(f"[INFO] Catalog: {CATALOG_FITS}")
    print(f"[INFO] RA col: {ra_col}, DEC col: {dec_col}, ID col: {id_col or '(none)'}")
    print(f"[INFO] Input FITS files: {len(fits_files)}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Cutout: {CUT_NX}x{CUT_NY} on last 2 axes")

    for fi, fpath in enumerate(fits_files, start=1):
        try:
            with fits.open(fpath, memmap=True) as hdul:
                hdu = hdul[0]
                data = hdu.data
                hdr  = hdu.header

                if data is None:
                    print(f"[SKIP] {os.path.basename(fpath)} no data.")
                    continue

                # use celestial WCS to locate pixels
                w_cel = WCS(hdr).celestial

                # footprint for prefilter
                fp = w_cel.calc_footprint()
                fp_ra = fp[:, 0]
                fp_dec = fp[:, 1]
                ra_min, ra_max, use_wrap = ra_wrap_bounds(fp_ra)
                dec_min, dec_max = fp_dec.min(), fp_dec.max()

                mask = (~written) & ra_in_range(ra_deg, ra_min, ra_max, use_wrap) & (dec_deg >= dec_min) & (dec_deg <= dec_max)
                idxs = np.where(mask)[0]

                if idxs.size == 0:
                    print(f"[{fi}/{len(fits_files)}] {os.path.basename(fpath)} -> 0 candidates")
                    continue

                print(f"[{fi}/{len(fits_files)}] {os.path.basename(fpath)} -> {idxs.size} candidates (coarse)")

                H = data.shape[-2]
                W = data.shape[-1]

                for idx in idxs:
                    pos = SkyCoord(ra=ra_deg[idx]*u.deg, dec=dec_deg[idx]*u.deg, frame="icrs")
                    x, y = w_cel.world_to_pixel(pos)

                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    if (x < -CUT_NX) or (x > W + CUT_NX) or (y < -CUT_NY) or (y > H + CUT_NY):
                        continue

                    cut_data, x0, y0 = cut_last2_axes(data, x, y, nx=CUT_NX, ny=CUT_NY, fill_value=np.nan)

                    out_hdr = hdr.copy()
                    out_hdr = update_celestial_wcs_in_header(out_hdr, x0, y0)

                    # update NAXISn to match output
                    # Keep NAXIS (dim count) unchanged; only update last two sizes
                    naxis = out_hdr.get("NAXIS", data.ndim)
                    # FITS axis numbering: NAXIS1 = last axis (x), NAXIS2 = second-last (y)
                    out_hdr["NAXIS1"] = CUT_NX
                    out_hdr["NAXIS2"] = CUT_NY

                    # provenance
                    out_hdr["HISTORY"] = f"CUTOUT: {CUT_NX}x{CUT_NY} on last2 axes from {os.path.basename(fpath)}"
                    out_hdr["HISTORY"] = f"CENTER_RA  = {ra_deg[idx]:.8f} deg"
                    out_hdr["HISTORY"] = f"CENTER_DEC = {dec_deg[idx]:.8f} deg"
                    out_hdr["SRCID"] = (str(ids[idx])[:68], "Source identifier (from catalog if available)")
                    out_hdr["PARENT"] = (os.path.basename(fpath)[:68], "Parent FITS mosaic")

                    out_path = os.path.join(OUT_DIR, f"{ids[idx]}.fits")
                    fits.PrimaryHDU(data=cut_data, header=out_hdr).writeto(out_path, overwrite=True)
                    written[idx] = True

        except Exception as e:
            print(f"[ERROR] Failed processing {fpath}: {e}")
            continue

    n_written = int(written.sum())
    n_total = len(cat)
    n_missing = n_total - n_written

    print("\n[SUMMARY]")
    print(f"  Written: {n_written}/{n_total}")
    print(f"  Missing: {n_missing}/{n_total}")

    if n_missing > 0:
        miss_path = os.path.join(OUT_DIR, "missing_sources.txt")
        with open(miss_path, "w", encoding="utf-8") as f:
            for idx in np.where(~written)[0]:
                f.write(f"{ids[idx]}\tRA={ra_deg[idx]:.8f}\tDEC={dec_deg[idx]:.8f}\n")
        print(f"  Missing list saved to: {miss_path}")


if __name__ == "__main__":
    main()
