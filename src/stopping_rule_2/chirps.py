"""CHIRPS loading and area-weighted aggregation to woredas."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse
from shapely.geometry import box

from . import config as C

EQUAL_AREA = "ESRI:102022"  # Africa Albers Equal Area Conic


def load_woredas() -> gpd.GeoDataFrame:
    g = gpd.read_file(C.SHAPEFILE)
    return g[[C.PCODE_FIELD, "adm3_name", "adm2_name", "adm1_name", "geometry"]].rename(
        columns={C.PCODE_FIELD: "pcode"}
    )


def open_chirps(years=C.CHIRPS_YEARS) -> xr.DataArray:
    parts = []
    for y in years:
        with xr.open_dataset(C.CHIRPS_DIR / f"{y}.nc", engine="h5netcdf") as ds:
            parts.append(ds["precip"].load())
    return xr.concat(parts, dim="time")


def weight_matrix(lat: np.ndarray, lon: np.ndarray, woredas: gpd.GeoDataFrame):
    """Sparse (n_woreda, n_lat*n_lon) matrix of cell/woreda intersection areas,
    row-normalised so each woreda's weights sum to 1."""
    dlat = float(np.median(np.diff(lat)))
    dlon = float(np.median(np.diff(lon)))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    cells = gpd.GeoDataFrame(
        {"cell": np.arange(LA.size)},
        geometry=[
            box(x - dlon / 2, y - dlat / 2, x + dlon / 2, y + dlat / 2)
            for y, x in zip(LA.ravel(), LO.ravel())
        ],
        crs="EPSG:4326",
    )
    xmin, ymin, xmax, ymax = woredas.total_bounds
    cells = cells.cx[xmin - 1 : xmax + 1, ymin - 1 : ymax + 1]
    w = woredas[["pcode", "geometry"]].reset_index(drop=True)
    w["row"] = np.arange(len(w))
    inter = gpd.overlay(w.to_crs(EQUAL_AREA), cells.to_crs(EQUAL_AREA), how="intersection")
    inter["a"] = inter.area
    M = sparse.csr_matrix(
        (inter["a"].to_numpy(), (inter["row"].to_numpy(), inter["cell"].to_numpy())),
        shape=(len(w), LA.size),
    )
    rs = np.asarray(M.sum(axis=1)).ravel()
    M = sparse.diags(1.0 / rs) @ M
    return M.tocsr()


def aggregate(out=C.CHIRPS_WOREDA) -> pd.DataFrame:
    """Aggregate CHIRPS to woredas; returns (date x pcode) DataFrame and caches it."""
    woredas = load_woredas()
    da = open_chirps().load()
    lat, lon = da.latitude.values, da.longitude.values
    M = weight_matrix(lat, lon, woredas)
    vals = da.values.reshape(da.shape[0], -1)
    if np.isnan(vals).any():
        # Renormalise weights over valid cells per day (CHIRPS is ocean-masked).
        valid = ~np.isnan(vals)
        num = (M @ np.nan_to_num(vals).T).T
        den = (M @ valid.T.astype(float)).T
        agg = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    else:
        agg = (M @ vals.T).T
    df = pd.DataFrame(agg.astype("float32"), index=pd.DatetimeIndex(da.time.values, name="date"),
                      columns=woredas["pcode"].to_numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def load_woreda_series(path=C.CHIRPS_WOREDA) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Parquet yields one block per column; consolidate into a single float
    # block so row selections don't rebuild (and pin) a full copy each time.
    return pd.DataFrame(df.to_numpy(dtype="float32"), index=df.index, columns=df.columns)


def season_cube(df: pd.DataFrame, years, pcodes) -> np.ndarray:
    """Return array (year, day, pcode) covering ARRAY_START..ARRAY_END each year.

    Day index i corresponds to offset i + ARRAY_START_OFFSET from May 1.
    """
    sub = df[list(pcodes)]
    out = []
    for y in years:
        s = pd.Timestamp(y, *C.ARRAY_START)
        e = pd.Timestamp(y, *C.ARRAY_END)
        block = sub.loc[s:e].to_numpy(dtype=float)
        n = (e - s).days + 1
        if block.shape[0] != n:
            raise ValueError(f"CHIRPS {y}: expected {n} days, got {block.shape[0]}")
        out.append(block)
    return np.stack(out)
