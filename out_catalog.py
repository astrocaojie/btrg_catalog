#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
from astropy.table import Table
from astropy.io import fits

# ====== CONFIG ======
IN_H5  = "/shared/main/caojie/meerkat/candidates/benttail_v2.h5"   # 改成你的实际路径
OUT_FITS = "benttail_catalog.fits"
# ====================

def _as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _as_int(x, default=-999999):
    try:
        return int(x)
    except Exception:
        return default

def _ds_to_str(ds):
    """Dataset -> comma-joined string (safe for FITS)."""
    try:
        arr = np.array(ds[()])
        # bytes -> str
        if arr.dtype.kind in ("S", "O"):
            arr = [a.decode("utf-8", errors="ignore") if isinstance(a, (bytes, np.bytes_)) else str(a) for a in arr]
        else:
            arr = arr.tolist()
        # flatten if needed
        if isinstance(arr, (list, tuple)):
            return ",".join([str(v) for v in arr])
        return str(arr)
    except Exception:
        return ""

def main():
    if not os.path.exists(IN_H5):
        raise FileNotFoundError(f"IN_H5 not found: {IN_H5}")

    rows = []
    with h5py.File(IN_H5, "r") as f:
        keys = sorted([k for k in f.keys() if isinstance(f[k], h5py.Group)])
        print("[INFO] groups:", len(keys))

        for gname in keys:
            g = f[gname]

            # attrs (common in your pipeline)
            ra  = _as_float(g.attrs.get("RA", np.nan))
            dec = _as_float(g.attrs.get("DEC", np.nan))

            isl = g.attrs.get("Isl_id", np.nan)
            try:
                isl = int(isl)
            except Exception:
                isl = _as_int(isl, -999999)

            fits_path = g.attrs.get("fits_path", "")
            if isinstance(fits_path, (bytes, np.bytes_)):
                fits_path = fits_path.decode("utf-8", errors="ignore")
            else:
                fits_path = str(fits_path) if fits_path is not None else ""

            sname = g.attrs.get("Source_Name_first", "")
            if isinstance(sname, (bytes, np.bytes_)):
                sname = sname.decode("utf-8", errors="ignore")
            else:
                sname = str(sname) if sname is not None else ""

            # optional labels / scores if present
            bt_label = g.attrs.get("bt_label", -999999)
            bt_label = _as_int(bt_label, -999999)

            p_bent = g.attrs.get("P_bent", np.nan)
            p_bent = _as_float(p_bent, np.nan)

            # datasets that may exist
            src_id_str = ""
            if "Source_id" in g:
                src_id_str = _ds_to_str(g["Source_id"])

            src_name_str = ""
            if "Source_Name" in g:
                src_name_str = _ds_to_str(g["Source_Name"])

            rows.append({
                "group": gname,
                "RA": ra,
                "DEC": dec,
                "Isl_id": isl,
                "Source_id": src_id_str,
                "Source_Name_first": sname,
                "Source_Name_all": src_name_str,
                "fits_path": fits_path,
                "bt_label": bt_label,
                "P_bent": p_bent,
            })

    tab = Table(rows)

    # 写 FITS（binary table）
    tab.write(OUT_FITS, format="fits", overwrite=True)
    print("[DONE] saved FITS catalog:", OUT_FITS)
    print("[INFO] columns:", tab.colnames)
    print("[INFO] nrows:", len(tab))

if __name__ == "__main__":
    main()
