#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import argparse
import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.irsa import Irsa
from tqdm import tqdm

# =========================
# 默认路径
# =========================
DEFAULT_INPUT_FITS = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog.fits"
DEFAULT_OUT_FITS   = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog_allwise_matched.fits"
DEFAULT_OUT_CSV    = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog_allwise_matched.csv"

# =========================
# 参数
# =========================
DEFAULT_RADIUS_ARCSEC = 2.0
DEFAULT_FALLBACK_RADIUS_ARCSEC = 5.0
DEFAULT_SLEEP_SEC = 0.05  # 你网络慢，建议先用 0.05；若稳定可设 0

Irsa.ROW_LIMIT = 50
ALLWISE_CATALOG = "allwise_p3as_psd"

WISE_COLS = [
    "designation", "ra", "dec",
    "w1mpro", "w1sigmpro", "w1snr",
    "w2mpro", "w2sigmpro", "w2snr",
    "w3mpro", "w3sigmpro", "w3snr",
    "w4mpro", "w4sigmpro", "w4snr",
    "cc_flags", "ph_qual", "ext_flg", "var_flg"
]

def extract_wise_token(host_name) -> str | None:
    if host_name is None:
        return None
    s = str(host_name).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.replace('"', ' ').replace("'", " ")
    s = re.sub(r"\s+", " ", s).strip()
    m = re.search(r"(J\d{6}\.\d{2}[+-]\d{6}\.\d)", s)
    return m.group(1) if m else None

def wise_name_to_coord(jtoken: str) -> SkyCoord | None:
    if not jtoken:
        return None
    try:
        return SkyCoord(jtoken, unit=(u.hourangle, u.deg), frame="icrs")
    except Exception:
        return None

def query_allwise_nearest(coord: SkyCoord, radius_arcsec: float):
    out = {f"allwise_{k}": None for k in WISE_COLS}
    out["allwise_sep_arcsec"] = None
    out["allwise_error"] = None
    out["allwise_radius_used"] = None
    out["allwise_fallback5"] = 0

    if coord is None:
        return out

    try:
        tab = Irsa.query_region(
            coord, catalog=ALLWISE_CATALOG, spatial="Cone",
            radius=radius_arcsec * u.arcsec
        )
        if tab is None or len(tab) == 0:
            return out

        coords2 = SkyCoord(tab["ra"], tab["dec"], unit=(u.deg, u.deg), frame="icrs")
        sep = coord.separation(coords2).arcsec
        i = int(np.argmin(sep))

        row = tab[i]
        for k in WISE_COLS:
            if k in row.colnames:
                val = row[k]
                try:
                    if getattr(val, "mask", False):
                        val = None
                except Exception:
                    pass
                out[f"allwise_{k}"] = None if val is None else (val.item() if hasattr(val, "item") else val)

        out["allwise_sep_arcsec"] = float(sep[i])
        out["allwise_radius_used"] = float(radius_arcsec)
        return out

    except Exception as e:
        out["allwise_error"] = str(e)
        out["allwise_radius_used"] = float(radius_arcsec)
        return out

def fits_safe_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _to_fixed_str_col(s: pd.Series) -> np.ndarray:
        arr = s.fillna("").astype(str).to_numpy()
        maxlen = max([len(x) for x in arr] + [1])
        return arr.astype(f"<U{maxlen}")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = _to_fixed_str_col(df[col])

    numeric_like = [
        "host_ra_deg", "host_dec_deg",
        "allwise_ra", "allwise_dec",
        "allwise_w1mpro", "allwise_w1sigmpro", "allwise_w1snr",
        "allwise_w2mpro", "allwise_w2sigmpro", "allwise_w2snr",
        "allwise_w3mpro", "allwise_w3sigmpro", "allwise_w3snr",
        "allwise_w4mpro", "allwise_w4sigmpro", "allwise_w4snr",
        "allwise_sep_arcsec", "allwise_radius_used",
        "allwise_fallback5"
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def main():
    ap = argparse.ArgumentParser(description="Cross-match HOST_NAME (WISE J-name) to AllWISE with fallback radius.")
    ap.add_argument("input_fits", nargs="?", default=DEFAULT_INPUT_FITS)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_ARCSEC, help="Primary radius arcsec (default 2.0)")
    ap.add_argument("--fallback_radius", type=float, default=DEFAULT_FALLBACK_RADIUS_ARCSEC, help="Fallback radius arcsec (default 5.0)")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SEC, help="Sleep seconds between queries")
    ap.add_argument("--out_fits", default=DEFAULT_OUT_FITS)
    ap.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    args = ap.parse_args()

    print("========== AllWISE Cross-match (2\" + fallback) ==========")
    print(f"Input FITS       : {args.input_fits}")
    print(f"Primary radius   : {args.radius} arcsec")
    print(f"Fallback radius  : {args.fallback_radius} arcsec (only if primary fails)")
    print(f"Sleep            : {args.sleep} s/query")
    print(f"Output FITS      : {args.out_fits}")
    print(f"Output CSV       : {args.out_csv}")
    print("=========================================================\n")

    tab = Table.read(args.input_fits)
    df = tab.to_pandas()

    if "HOST_NAME" not in df.columns:
        raise ValueError("Input FITS must contain column: HOST_NAME")

    df["host_wise_token"] = df["HOST_NAME"].apply(extract_wise_token)

    coords = []
    df["host_ra_deg"] = np.nan
    df["host_dec_deg"] = np.nan

    for i, tok in enumerate(df["host_wise_token"].tolist()):
        c = wise_name_to_coord(tok) if tok else None
        coords.append(c)
        if c is not None:
            df.at[i, "host_ra_deg"] = float(c.ra.deg)
            df.at[i, "host_dec_deg"] = float(c.dec.deg)

    n_total = len(df)
    print(f"Rows total: {n_total}, with HOST token: {df['host_wise_token'].notna().sum()}")

    results = []
    pbar = tqdm(range(n_total), desc="Querying AllWISE", ncols=110)
    for i in pbar:
        c = coords[i]

        # 1) primary
        res = query_allwise_nearest(c, args.radius)

        # 2) fallback only if no designation AND no hard error
        if (res.get("allwise_designation") is None) and (res.get("allwise_error") is None) and (c is not None):
            res2 = query_allwise_nearest(c, args.fallback_radius)
            # 如果 fallback 成功，就用 fallback 结果，并打标
            if res2.get("allwise_designation") is not None:
                res2["allwise_fallback5"] = 1
                res = res2

        results.append(res)

        if res.get("allwise_designation") is not None:
            pbar.set_postfix_str(f"sep={res['allwise_sep_arcsec']:.2f}\" r={res['allwise_radius_used']:.0f}\"")
        else:
            pbar.set_postfix_str("no match")

        if args.sleep > 0:
            time.sleep(args.sleep)

    df_wise = pd.DataFrame(results)
    df_out = pd.concat([df, df_wise], axis=1)

    # 输出 CSV
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nCSV saved: {args.out_csv}")

    # 输出 FITS（先做fits安全处理）
    df_fits = fits_safe_cast(df_out)
    Table.from_pandas(df_fits).write(args.out_fits, overwrite=True)
    print(f"FITS saved: {args.out_fits}")

    matched = df_out["allwise_designation"].notna().sum() if "allwise_designation" in df_out.columns else 0
    fb = df_out["allwise_fallback5"].fillna(0).astype(int).sum() if "allwise_fallback5" in df_out.columns else 0
    miss = len(df_out) - matched
    print("\n========== Summary ==========")
    print(f"Matched: {matched}/{len(df_out)}")
    print(f"  - via fallback radius: {fb}")
    print(f"Missing: {miss}")
    print("=============================\n")

if __name__ == "__main__":
    main()