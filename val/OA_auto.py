#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OA_auto.py

Automatic opening-angle measurement for bent-tail candidates.

Pipeline:
1) Read radio cutouts from RADIO_DIR
2) Build a robust connected-component mask around the main source
3) Skeletonize the mask and extract the main spine
4) Choose T points from the two ends of the longest skeleton path
5) Choose C point:
   - prefer host position (from host CSV or catalog HOST_NAME), snapped to the spine
   - fallback to the strongest bend (maximum curvature) on the spine
6) Compute OA and optionally write results back to the catalog

Outputs:
- CSV summary with OA/C/T info
- optional FITS catalog update with columns:
    OA_auto, type_auto,
    C_auto_ra, C_auto_dec,
    T1_auto_ra, T1_auto_dec,
    T2_auto_ra, T2_auto_dec,
    OA_auto_status
"""

from __future__ import annotations

import os
import re
import math
import glob
import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, skeletonize


# =======================
# Config
# =======================
CATALOG_FITS = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog.fits"
RADIO_DIR = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/bent_crop_300/"
HOST_CSV = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/host_candidates_fig2_bylabel_updated.csv"

OUT_CSV = "/home/caojie/work/Galaxy-Morphology/bent_catalog/val/oa_auto_results.csv"
WRITE_BACK = True
WRITE_BACK_PATH = "/home/caojie/work/Galaxy-Morphology/bent_catalog/final_catalog.fits"

TYPE_THRESHOLD_DEG = 90.0
MIN_MASK_PIX = 30
LOW_SIGMA = 2.2
HIGH_SIGMA = 3.8
HOST_SNAP_PIX = 18.0
CURVATURE_WINDOW = 7
MAX_CURVATURE_END_FRAC = 0.18


@dataclass
class OAResult:
    row_idx: int
    source_id: str
    fits_path: str
    status: str
    oa_deg: float = np.nan
    typ: str = "---"
    c_ra: float = np.nan
    c_dec: float = np.nan
    t1_ra: float = np.nan
    t1_dec: float = np.nan
    t2_ra: float = np.nan
    t2_dec: float = np.nan
    host_ra: float = np.nan
    host_dec: float = np.nan
    n_endpoints: int = 0
    mask_area: int = 0
    path_npix: int = 0


def robust_sigma_border(data: np.ndarray, border_frac: float = 0.18) -> tuple[float, float]:
    ny, nx = data.shape
    bx = max(1, int(nx * border_frac))
    by = max(1, int(ny * border_frac))
    m = np.zeros_like(data, dtype=bool)
    m[:by, :] = True
    m[-by:, :] = True
    m[:, :bx] = True
    m[:, -bx:] = True
    v = data[m & np.isfinite(data)]
    if v.size < 50:
        v = data[np.isfinite(data)]
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(v))
    return med, sig


def extract_wise_token(host_name) -> str | None:
    if host_name is None:
        return None
    s = str(host_name).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.replace("\n", " ").replace('"', " ").replace("'", " ")
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


def parse_host_csv(path: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path)
    if "source_id" not in df.columns or "host_ra_deg" not in df.columns or "host_dec_deg" not in df.columns:
        return out

    for _, row in df.iterrows():
        sid = normalize_id(row.get("source_id"))
        try:
            ra = float(row["host_ra_deg"])
            dec = float(row["host_dec_deg"])
        except Exception:
            continue
        if np.isfinite(ra) and np.isfinite(dec):
            out[sid] = (ra, dec)
    return out


def normalize_id(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if np.isfinite(f):
            if abs(f - round(f)) < 1e-6:
                return str(int(round(f)))
            return f"{f:.6f}".rstrip("0").rstrip(".")
    except Exception:
        pass
    return s


def read_radio_cutout(path: str) -> tuple[np.ndarray, WCS]:
    with fits.open(path, memmap=False) as hdul:
        data = np.squeeze(hdul[0].data).astype(np.float32)
        wcs = WCS(hdul[0].header).celestial
    if data.ndim != 2:
        raise ValueError(f"Expected 2D radio cutout, got {data.shape}")
    return data, wcs


def build_connected_mask(data: np.ndarray, anchor_xy: tuple[float, float] | None = None) -> np.ndarray:
    med, sig = robust_sigma_border(data)
    if not np.isfinite(sig) or sig <= 0:
        return np.zeros_like(data, dtype=bool)

    high = np.isfinite(data) & (data >= med + HIGH_SIGMA * sig)
    low = np.isfinite(data) & (data >= med + LOW_SIGMA * sig)

    high = remove_small_objects(high, MIN_MASK_PIX)
    low = remove_small_objects(low, max(8, MIN_MASK_PIX // 2))

    st = ndi.generate_binary_structure(2, 2)
    low = ndi.binary_closing(low, structure=st, iterations=2)
    low = ndi.binary_fill_holes(low)

    lab, nlab = ndi.label(low, structure=st)
    if nlab == 0:
        return np.zeros_like(data, dtype=bool)

    chosen = 0
    if anchor_xy is not None and np.all(np.isfinite(anchor_xy)):
        ax, ay = anchor_xy
        ix, iy = int(round(ax)), int(round(ay))
        if 0 <= ix < data.shape[1] and 0 <= iy < data.shape[0]:
            chosen = int(lab[iy, ix])

    if chosen == 0 and np.any(high):
        hit = np.unique(lab[high])
        hit = hit[hit > 0]
        if hit.size > 0:
            best, best_peak = None, -np.inf
            for lid in hit:
                peak = float(np.nanmax(data[lab == lid]))
                if peak > best_peak:
                    best_peak = peak
                    best = int(lid)
            chosen = int(best)

    if chosen == 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        chosen = int(np.argmax(sizes))

    mask = (lab == chosen)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=st, iterations=1)
    mask = remove_small_objects(mask, MIN_MASK_PIX)
    return mask.astype(bool)


def skeleton_neighbors(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ker = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = ndi.convolve(skel.astype(np.uint8), ker, mode="constant", cval=0)
    nnb = conv - 10 * skel.astype(np.uint8)
    return nnb, conv


def get_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    nnb, _ = skeleton_neighbors(skel)
    ys, xs = np.where(skel & (nnb == 1))
    return [(int(x), int(y)) for y, x in zip(ys, xs)]


def nearest_true_pixel(mask: np.ndarray, x: float, y: float, r: int = 30) -> tuple[int, int] | None:
    iy, ix = int(round(y)), int(round(x))
    y0 = max(0, iy - r)
    y1 = min(mask.shape[0] - 1, iy + r)
    x0 = max(0, ix - r)
    x1 = min(mask.shape[1] - 1, ix + r)
    cut = mask[y0:y1 + 1, x0:x1 + 1]
    ys, xs = np.where(cut)
    if xs.size == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    k = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[k]), int(ys[k])


def dijkstra_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    sx, sy = start
    gx, gy = goal
    if not mask[sy, sx] or not mask[gy, gx]:
        return None

    h, w = mask.shape
    dist = np.full((h, w), np.inf, dtype=float)
    prev = np.full((h, w, 2), -1, dtype=int)
    pq: list[tuple[float, int, int]] = []

    dist[sy, sx] = 0.0
    heapq.heappush(pq, (0.0, sx, sy))
    nbrs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    while pq:
        d, x, y = heapq.heappop(pq)
        if d != dist[y, x]:
            continue
        if x == gx and y == gy:
            break
        for dx, dy in nbrs:
            xx, yy = x + dx, y + dy
            if xx < 0 or xx >= w or yy < 0 or yy >= h or not mask[yy, xx]:
                continue
            step = 1.4142 if dx != 0 and dy != 0 else 1.0
            nd = d + step
            if nd < dist[yy, xx]:
                dist[yy, xx] = nd
                prev[yy, xx] = [x, y]
                heapq.heappush(pq, (nd, xx, yy))

    if not np.isfinite(dist[gy, gx]):
        return None

    path: list[tuple[int, int]] = []
    x, y = gx, gy
    while not (x == sx and y == sy):
        path.append((x, y))
        px, py = prev[y, x]
        if px < 0:
            return None
        x, y = int(px), int(py)
    path.append((sx, sy))
    path.reverse()
    return path


def longest_endpoint_path(skel: np.ndarray, endpoints: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(endpoints) < 2:
        return None

    best_path = None
    best_len = -1.0
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            path = dijkstra_path(skel, endpoints[i], endpoints[j])
            if path is None:
                continue
            plen = 0.0
            for k in range(1, len(path)):
                x0, y0 = path[k - 1]
                x1, y1 = path[k]
                plen += math.hypot(x1 - x0, y1 - y0)
            if plen > best_len:
                best_len = plen
                best_path = path
    return best_path


def curvature_index(path: list[tuple[int, int]], window: int = CURVATURE_WINDOW) -> int:
    n = len(path)
    if n < 2 * window + 3:
        return n // 2

    lo = max(1, int(n * MAX_CURVATURE_END_FRAC))
    hi = min(n - 2, int(n * (1.0 - MAX_CURVATURE_END_FRAC)))
    if hi <= lo:
        return n // 2

    best_i = n // 2
    best_bend = -np.inf
    pts = np.asarray(path, dtype=float)

    for i in range(max(window, lo), min(n - window, hi)):
        v1 = pts[i - window] - pts[i]
        v2 = pts[i + window] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 0 or n2 <= 0:
            continue
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = math.degrees(math.acos(cosv))
        bend = 180.0 - ang
        if bend > best_bend:
            best_bend = bend
            best_i = i
    return int(best_i)


def snap_host_to_path(path: list[tuple[int, int]], host_xy: tuple[float, float] | None) -> int | None:
    if host_xy is None or not np.all(np.isfinite(host_xy)):
        return None
    pts = np.asarray(path, dtype=float)
    hx, hy = host_xy
    d2 = (pts[:, 0] - hx) ** 2 + (pts[:, 1] - hy) ** 2
    i = int(np.argmin(d2))
    if float(np.sqrt(d2[i])) <= HOST_SNAP_PIX:
        return i
    return None


def opening_angle_deg(c_world: SkyCoord, t1_world: SkyCoord, t2_world: SkyCoord) -> float:
    pa1 = c_world.position_angle(t1_world).to_value(u.deg)
    pa2 = c_world.position_angle(t2_world).to_value(u.deg)
    d = abs(pa1 - pa2) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return float(d)


def build_cutout_index(radio_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for fp in sorted(glob.glob(os.path.join(radio_dir, "*.fits"))):
        stem = os.path.splitext(os.path.basename(fp))[0]
        toks = re.split(r"__|_", stem)
        cands = [stem] + toks
        for c in cands:
            sid = normalize_id(c)
            if sid and sid not in out:
                out[sid] = fp
    return out


def pick_source_id_column(tab: Table) -> str:
    for cand in ("Source_id", "source_id", "ID", "id"):
        if cand in tab.colnames:
            return cand
    raise RuntimeError("Cannot find source id column in catalog")


def get_host_coord_for_row(row, host_map: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    sid = normalize_id(row.get("Source_id", None))
    if sid in host_map:
        return host_map[sid]

    if "HOST_RA" in row.colnames and "HOST_DEC" in row.colnames:
        try:
            ra = float(row["HOST_RA"])
            dec = float(row["HOST_DEC"])
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
        except Exception:
            pass

    if "host_ra_deg" in row.colnames and "host_dec_deg" in row.colnames:
        try:
            ra = float(row["host_ra_deg"])
            dec = float(row["host_dec_deg"])
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
        except Exception:
            pass

    if "HOST_NAME" in row.colnames:
        coord = wise_name_to_coord(extract_wise_token(row["HOST_NAME"]))
        if coord is not None:
            return float(coord.ra.deg), float(coord.dec.deg)

    return None


def process_one(row_idx: int, row, cutout_path: str, host_map: dict[str, tuple[float, float]]) -> OAResult:
    sid = normalize_id(row["Source_id"]) if "Source_id" in row.colnames else str(row_idx)
    result = OAResult(row_idx=row_idx, source_id=sid, fits_path=cutout_path, status="init")

    try:
        data, wcs = read_radio_cutout(cutout_path)
    except Exception as e:
        result.status = f"read_error:{e}"
        return result

    host_world = None
    host_xy = None
    host_coord = get_host_coord_for_row(row, host_map)
    if host_coord is not None:
        result.host_ra, result.host_dec = host_coord
        host_world = SkyCoord(host_coord[0] * u.deg, host_coord[1] * u.deg, frame="icrs")
        try:
            hx, hy = wcs.world_to_pixel(host_world)
            if np.all(np.isfinite([hx, hy])):
                host_xy = (float(hx), float(hy))
        except Exception:
            host_xy = None

    anchor_xy = host_xy
    if anchor_xy is None:
        anchor_xy = ((data.shape[1] - 1) / 2.0, (data.shape[0] - 1) / 2.0)

    mask = build_connected_mask(data, anchor_xy=anchor_xy)
    result.mask_area = int(mask.sum())
    if not np.any(mask):
        result.status = "no_mask"
        return result

    skel = skeletonize(mask)
    if not np.any(skel):
        result.status = "no_skeleton"
        return result

    endpoints = get_endpoints(skel)
    result.n_endpoints = len(endpoints)
    if len(endpoints) < 2:
        result.status = "too_few_endpoints"
        return result

    path = longest_endpoint_path(skel, endpoints)
    if path is None or len(path) < 3:
        result.status = "no_main_path"
        return result
    result.path_npix = len(path)

    c_idx = snap_host_to_path(path, host_xy)
    c_mode = "host"
    if c_idx is None:
        c_idx = curvature_index(path)
        c_mode = "bend"

    # Avoid placing C too close to an end.
    c_idx = int(np.clip(c_idx, 1, len(path) - 2))

    c_pix = path[c_idx]
    t1_pix = path[0]
    t2_pix = path[-1]

    c_world = SkyCoord.from_pixel(float(c_pix[0]), float(c_pix[1]), wcs)
    t1_world = SkyCoord.from_pixel(float(t1_pix[0]), float(t1_pix[1]), wcs)
    t2_world = SkyCoord.from_pixel(float(t2_pix[0]), float(t2_pix[1]), wcs)

    oa_deg = opening_angle_deg(c_world, t1_world, t2_world)
    typ = "NAT" if oa_deg < TYPE_THRESHOLD_DEG else "WAT"

    result.status = f"ok:{c_mode}"
    result.oa_deg = oa_deg
    result.typ = typ
    result.c_ra = float(c_world.ra.deg)
    result.c_dec = float(c_world.dec.deg)
    result.t1_ra = float(t1_world.ra.deg)
    result.t1_dec = float(t1_world.dec.deg)
    result.t2_ra = float(t2_world.ra.deg)
    result.t2_dec = float(t2_world.dec.deg)
    return result


def main():
    if not os.path.exists(CATALOG_FITS):
        raise FileNotFoundError(f"catalog not found: {CATALOG_FITS}")

    tab = Table.read(CATALOG_FITS)
    sid_col = pick_source_id_column(tab)
    if sid_col != "Source_id":
        tab["Source_id"] = tab[sid_col]

    host_map = parse_host_csv(HOST_CSV)
    cutout_index = build_cutout_index(RADIO_DIR)

    rows: list[dict] = []
    results: list[OAResult] = []

    for i, row in enumerate(tab):
        sid = normalize_id(row["Source_id"])
        cutout_path = cutout_index.get(sid, "")
        if not cutout_path:
            res = OAResult(row_idx=i, source_id=sid, fits_path="", status="no_cutout")
        else:
            res = process_one(i, row, cutout_path, host_map)
        results.append(res)
        rows.append({
            "row_idx": res.row_idx,
            "Source_id": res.source_id,
            "fits_path": res.fits_path,
            "OA_auto": res.oa_deg,
            "type_auto": res.typ,
            "C_auto_ra": res.c_ra,
            "C_auto_dec": res.c_dec,
            "T1_auto_ra": res.t1_ra,
            "T1_auto_dec": res.t1_dec,
            "T2_auto_ra": res.t2_ra,
            "T2_auto_dec": res.t2_dec,
            "host_ra_deg": res.host_ra,
            "host_dec_deg": res.host_dec,
            "n_endpoints": res.n_endpoints,
            "mask_area": res.mask_area,
            "path_npix": res.path_npix,
            "OA_auto_status": res.status,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    if WRITE_BACK:
        oa = np.full(len(tab), np.nan, dtype=float)
        typ = np.full(len(tab), "---", dtype="U16")
        cra = np.full(len(tab), np.nan, dtype=float)
        cdec = np.full(len(tab), np.nan, dtype=float)
        t1ra = np.full(len(tab), np.nan, dtype=float)
        t1dec = np.full(len(tab), np.nan, dtype=float)
        t2ra = np.full(len(tab), np.nan, dtype=float)
        t2dec = np.full(len(tab), np.nan, dtype=float)
        stat = np.full(len(tab), "", dtype="U32")

        for r in results:
            oa[r.row_idx] = r.oa_deg
            typ[r.row_idx] = r.typ
            cra[r.row_idx] = r.c_ra
            cdec[r.row_idx] = r.c_dec
            t1ra[r.row_idx] = r.t1_ra
            t1dec[r.row_idx] = r.t1_dec
            t2ra[r.row_idx] = r.t2_ra
            t2dec[r.row_idx] = r.t2_dec
            stat[r.row_idx] = r.status

        tab["OA_auto"] = oa
        tab["type_auto"] = typ
        tab["C_auto_ra"] = cra
        tab["C_auto_dec"] = cdec
        tab["T1_auto_ra"] = t1ra
        tab["T1_auto_dec"] = t1dec
        tab["T2_auto_ra"] = t2ra
        tab["T2_auto_dec"] = t2dec
        tab["OA_auto_status"] = stat
        tab.write(WRITE_BACK_PATH, overwrite=True)

    n_ok = int(df["OA_auto"].notna().sum())
    print("[DONE] CSV:", OUT_CSV)
    if WRITE_BACK:
        print("[DONE] FITS updated:", WRITE_BACK_PATH)
    print(f"[INFO] measured OA_auto: {n_ok}/{len(df)}")
    print("[INFO] status counts:")
    print(df["OA_auto_status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
