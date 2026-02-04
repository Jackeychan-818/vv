#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary statistics on missing rates for visibility grid.

Missing rate = (Total - Observed) / Total

Where:
- Total = number of unique 0.1 degree grids × number of days
- Observed (0.1°) = number of 0.1° grid-days with non-missing visibility
- Observed (0.5°) = number of 0.1° grid-days with non-missing interpolated 0.5° visibility
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
GRID_ROOT = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/grid")
RESULTS_ROOT = Path("/Users/jackey/NUS Dropbox/Zeqi Chen/vv/results_vv")

YEARS = list(range(2018, 2025))

# =========================
# COMPUTE MISSING RATES
# =========================
summary_rows = []

for year in YEARS:
    print(f"\n=== {year} ===")

    grid_csv = GRID_ROOT / f"{year}.csv"
    results_csv = RESULTS_ROOT / f"visibility_{year}.csv"

    if not grid_csv.exists():
        print(f"  Missing grid file")
        continue
    if not results_csv.exists():
        print(f"  Missing results file")
        continue

    # ---- Load chlorophyll grid ----
    grid = pd.read_csv(grid_csv, usecols=["longitude", "latitude", "date"])
    grid["date"] = pd.to_datetime(grid["date"]).dt.date

    # Unique 0.1 degree grids (lon, lat pairs)
    unique_grids_01 = grid[["longitude", "latitude"]].drop_duplicates()
    n_unique_grids = len(unique_grids_01)

    # Number of unique days
    unique_days = grid["date"].unique()
    n_days = len(unique_days)

    # Total = unique grids × days
    n_total = n_unique_grids * n_days
    print(f"  Unique 0.1° grids: {n_unique_grids:,}, Days: {n_days}, Total: {n_total:,}")

    # ---- Load visibility results ----
    results = pd.read_csv(results_csv)
    results["date"] = pd.to_datetime(results["date"]).dt.date

    # Observed (0.1°): 0.1° grid-days with non-missing visibility
    # (dominant_vis_cat is the 0.1° visibility)
    obs_01 = results[results["dominant_vis_cat"].notna()]
    n_obs_01 = len(obs_01[["grid_lon", "grid_lat", "date"]].drop_duplicates())

    # Observed (0.5°): 0.1° grid-days with non-missing interpolated 0.5° visibility
    # A 0.1° cell has 0.5° visibility if its parent 0.5° cell has any visibility data

    # Get unique 0.5° grid-days that have visibility
    vis_05_cells = results[results["visibility_0.5degree"].notna()][
        ["longitude_interpolated", "latitude_interpolated", "date"]
    ].drop_duplicates()
    vis_05_cells = vis_05_cells.rename(columns={
        "longitude_interpolated": "lon_05",
        "latitude_interpolated": "lat_05"
    })

    # Create full 0.1° panel with 0.5° cell mapping
    full_panel = grid[["longitude", "latitude", "date"]].drop_duplicates().copy()
    full_panel["lon_05"] = np.round(full_panel["longitude"] / 0.5) * 0.5
    full_panel["lat_05"] = np.round(full_panel["latitude"] / 0.5) * 0.5

    # Count 0.1° cells whose parent 0.5° cell has visibility
    merged = full_panel.merge(
        vis_05_cells,
        on=["lon_05", "lat_05", "date"],
        how="left",
        indicator=True
    )
    n_obs_05 = (merged["_merge"] == "both").sum()

    # ---- Calculate missing rates ----
    missing_rate_01 = (n_total - n_obs_01) / n_total if n_total > 0 else np.nan
    missing_rate_05 = (n_total - n_obs_05) / n_total if n_total > 0 else np.nan

    print(f"  0.1° visibility: {n_obs_01:,} observed, missing rate: {missing_rate_01*100:.2f}%")
    print(f"  0.5° visibility: {n_obs_05:,} observed, missing rate: {missing_rate_05*100:.2f}%")

    summary_rows.append({
        "year": year,
        "n_unique_grids": n_unique_grids,
        "n_days": n_days,
        "n_total": n_total,
        "n_obs_01": n_obs_01,
        "missing_rate_01": missing_rate_01,
        "n_obs_05": n_obs_05,
        "missing_rate_05": missing_rate_05,
    })

# =========================
# SUMMARY TABLE
# =========================
summary = pd.DataFrame(summary_rows)

print("\n" + "="*80)
print("SUMMARY: Missing Rates by Year")
print("="*80)
print(summary.to_string(index=False))

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
total = summary["n_total"].sum()
total_obs_01 = summary["n_obs_01"].sum()
total_obs_05 = summary["n_obs_05"].sum()

print(f"Total (unique grids × days): {total:,}")

print(f"\n0.1° visibility:")
print(f"  Observed: {total_obs_01:,}")
print(f"  Overall missing rate: {(total - total_obs_01)/total*100:.2f}%")

print(f"\n0.5° visibility (interpolated):")
print(f"  Observed: {total_obs_05:,}")
print(f"  Overall missing rate: {(total - total_obs_05)/total*100:.2f}%")

# Save summary
out_path = RESULTS_ROOT / "missing_rate_summary_ndays.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved summary to: {out_path}")
