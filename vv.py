#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

# =========================
# USER SETTINGS (macOS)
# =========================
RAW_ROOT  = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/raw_data")
GRID_ROOT = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/grid")
OUT_ROOT  = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/results_new")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2018, 2025))  # 2018..2024
LON_CONVENTION = "[-180,180)"
VALID_VV_CODES = set(range(90, 100))

# =========================
# HELPERS
# =========================
def normalize_lon(arr, convention: str):
    """Normalize longitudes to a given convention."""
    arr = np.asarray(arr, dtype=float)
    if convention == "[-180,180)":
        return ((arr + 180) % 360) - 180
    elif convention == "[0,360)":
        return arr % 360
    else:
        raise ValueError("Bad LON_CONVENTION")

# =========================
# PROCESS YEAR BY YEAR
# =========================
for year in YEARS:
    print(f"\n=== Processing {year} ===")

    grid_csv = GRID_ROOT / f"{year}.csv"
    year_dir = RAW_ROOT / str(year)

    if not grid_csv.exists() or not year_dir.exists():
        print(f"Skip {year} (missing grid or data)")
        continue

    # ---- Load grid ----
    grid = pd.read_csv(grid_csv)
    grid = grid.rename(columns={"longitude": "grid_lon", "latitude": "grid_lat"})
    grid["date"] = pd.to_datetime(grid["date"]).dt.date
    grid["grid_lon"] = normalize_lon(grid["grid_lon"].values, LON_CONVENTION)
    grid["grid_lat"] = grid["grid_lat"].astype(float)

    # ---- Build KDTree dictionary by date ----
    print("Building KDTree index for each date...")
    unique_dates = grid["date"].unique()
    tree_dict = {}
    grid_dict = {}

    for date in unique_dates:
        date_grid = grid[grid["date"] == date].copy()
        grid_coords = np.column_stack([date_grid["grid_lon"].values, date_grid["grid_lat"].values])
        tree_dict[date] = cKDTree(grid_coords)
        grid_dict[date] = date_grid.reset_index(drop=True)

    print(f"Built KDTree for {len(unique_dates)} unique dates")

    # ---- Find all .nc files ----
    nc_files = sorted(glob.glob(str(year_dir / "*" / "*.nc")))
    print(f"Found {len(nc_files)} nc files")

    #  store per-file category counts:
    # (grid_lon, grid_lat, date, vis_cat) -> n_cat
    all_counts = []

    for i, fpath in enumerate(nc_files, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i}/{len(nc_files)}] {os.path.basename(fpath)}")

        try:
            ds = xr.open_dataset(fpath, decode_timedelta=False)

            if "VV" not in ds:
                ds.close()
                continue

            df = pd.DataFrame({
                "time": pd.to_datetime(ds["time"].values),
                "lon": ds["lon"].values.astype(float),
                "lat": ds["lat"].values.astype(float),
                "VV": ds["VV"].values.astype(float),
            })

            ds.close()

            # ---- Clean ----
            df = df.dropna()
            df = df[df["VV"] != -99]
            df["VV"] = df["VV"].astype(int)
            df = df[df["VV"].isin(VALID_VV_CODES)]
            if df.empty:
                continue

            df["date"] = df["time"].dt.date
            df["lon"] = normalize_lon(df["lon"].values, LON_CONVENTION)

            # ---- KDTree match by date ----
            df["grid_lon"] = np.nan
            df["grid_lat"] = np.nan

            for date in df["date"].unique():
                if date not in tree_dict:
                    # Skip observations on dates not in grid
                    continue

                date_mask = df["date"] == date
                date_obs = df.loc[date_mask]

                obs_coords = np.column_stack([date_obs["lon"].values, date_obs["lat"].values])
                _, idx = tree_dict[date].query(obs_coords, k=1)

                df.loc[date_mask, "grid_lon"] = grid_dict[date]["grid_lon"].values[idx]
                df.loc[date_mask, "grid_lat"] = grid_dict[date]["grid_lat"].values[idx]

            # Remove observations that couldn't be matched
            df = df.dropna(subset=["grid_lon", "grid_lat"])
            if df.empty:
                continue

            # ---- aggregate counts per category in THIS file ----
            df["vis_cat"] = df["VV"]

            counts = (
                df.groupby(["grid_lon", "grid_lat", "date", "vis_cat"])
                  .size()
                  .reset_index(name="n_cat")
            )

            all_counts.append(counts)

        except Exception as e:
            print(f"  ERROR processing {os.path.basename(fpath)}: {e}")
            continue

    if not all_counts:
        print(f"No visibility data for {year}")
        continue

    # ---- Combine counts across ALL files ----
    counts_all = pd.concat(all_counts, ignore_index=True)

    # Sum counts per category per grid-day across files
    counts_all = (
        counts_all.groupby(["grid_lon", "grid_lat", "date", "vis_cat"], as_index=False)
                  .agg(n_cat=("n_cat", "sum"))
    )

    # Total obs per grid-day
    nobs = (
        counts_all.groupby(["grid_lon", "grid_lat", "date"], as_index=False)
                  .agg(n_obs=("n_cat", "sum"))
    )

    # Dominant category per grid-day:
    # choose highest n_cat; tie-break by smallest vis_cat
    counts_all_sorted = counts_all.sort_values(
        ["grid_lon", "grid_lat", "date", "n_cat", "vis_cat"],
        ascending=[True, True, True, False, True]
    )

    dominant = (
        counts_all_sorted.groupby(["grid_lon", "grid_lat", "date"], as_index=False)
                         .first()
                         .rename(columns={"vis_cat": "dominant_vis_cat"})
                         [["grid_lon", "grid_lat", "date", "dominant_vis_cat"]]
    )

    vis_all = dominant.merge(nobs, on=["grid_lon", "grid_lat", "date"], how="left")
    # ---- FINAL CLEAN OUTPUT ----
    out = vis_all.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    # ---- Add 0.5° grid aggregation ----
    # Map each 0.1° grid point to the nearest 0.5° cell
    out["longitude_interpolated"] = np.round(out["grid_lon"] / 0.5) * 0.5
    out["latitude_interpolated"] = np.round(out["grid_lat"] / 0.5) * 0.5

    # Aggregate dominant visibility across 0.5° cells using n_obs as weights
    coarse_counts = (
        out.groupby(
            ["longitude_interpolated", "latitude_interpolated", "date", "dominant_vis_cat"],
            as_index=False
        )
        .agg(n_obs=("n_obs", "sum"))
    )

    coarse_sorted = coarse_counts.sort_values(
        ["longitude_interpolated", "latitude_interpolated", "date", "n_obs", "dominant_vis_cat"],
        ascending=[True, True, True, False, True]
    )

    coarse_dom = (
        coarse_sorted.groupby(
            ["longitude_interpolated", "latitude_interpolated", "date"],
            as_index=False
        )
        .first()
        .rename(columns={"dominant_vis_cat": "visibility_0.5degree"})
        [["longitude_interpolated", "latitude_interpolated", "date", "visibility_0.5degree"]]
    )

    out = out.merge(
        coarse_dom,
        on=["longitude_interpolated", "latitude_interpolated", "date"],
        how="left"
    )

    out = out[[
        "grid_lon",
        "grid_lat",
        "date",
        "dominant_vis_cat",
        "n_obs",
        "longitude_interpolated",
        "latitude_interpolated",
        "visibility_0.5degree"
    ]]

    # New filename so you keep old outputs
    out_path = OUT_ROOT / f"new_visibility_grid_day_{year}_v2_counts_mode.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved clean output: {out_path}")
