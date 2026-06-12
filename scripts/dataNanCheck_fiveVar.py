import numpy as np
import rasterio
import os

OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\SRTM\SouthAsia_DEM_0p05deg.tif"
ELEVATION_PATH = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\SouthAsia_DEM_0p05deg.tif"

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

VARIABLES = [
    'ERA5_PRECIP',
    'MSWEP_PRECIP',
    'ERA5_U_WIND_10M',
    'ERA5_V_WIND_10M',
    'ERA5_WIND_SPEED_10M',
    'CHIRPS_PRECIP',
]

# Precipitation variables where negative values are physically invalid
PRECIP_VARIABLES = {'ERA5_PRECIP', 'MSWEP_PRECIP', 'CHIRPS_PRECIP'}

# Build the full list of 36 files (6 variables × 6 years)
files = {}
for var in VARIABLES:
    for year in YEARS:
        key  = f"{var}_{year}"
        path = os.path.join(OUTPUT_DIR, f"{key}_aligned.tif")
        files[key] = path

# ── HELPER: check a single data array ────────────────────────────────────────
def check_array(name, data, check_negatives=False):
    """
    Prints NaN / negative / stats report for a numpy array (bands/days, H, W).
    Returns a list of issue strings (empty if clean).
    """
    total_pixels = data.size
    nan_mask     = np.isnan(data)
    nan_total    = nan_mask.sum()
    nan_days     = np.where(nan_mask.any(axis=(1, 2)))[0]
    zero_total   = (data == 0).sum()
    clean_data   = data[~nan_mask]

    print(f"  Shape          : {data.shape}  →  ({'days' if data.shape[0] > 1 else 'bands'}, H, W)")
    print(f"  Total pixels   : {total_pixels:,}")

    print(f"\n  ── NaN ──────────────────────────────────────────")
    print(f"  NaN count      : {nan_total}")
    print(f"  NaN slices     : {len(nan_days)}  →  "
          f"{nan_days.tolist() if len(nan_days) <= 10 else str(nan_days[:10].tolist()) + '...'}")

    issues = []

    if check_negatives:
        neg_mask  = data < 0
        neg_total = neg_mask.sum()
        neg_days  = np.where(neg_mask.any(axis=(1, 2)))[0]
        print(f"\n  ── Negatives (precip — physically invalid) ──────")
        print(f"  Negative count : {neg_total}")
        print(f"  Negative days  : {len(neg_days)}  →  "
              f"{neg_days.tolist() if len(neg_days) <= 10 else str(neg_days[:10].tolist()) + '...'}")
        if neg_total > 0:
            print(f"  Min negative   : {data[neg_mask].min():.6f}")
            issues.append(f"❌ {neg_total} negative values")
    else:
        print(f"\n  ── Negatives ────────────────────────────────────")
        print(f"  Skipped (wind U/V can be legitimately negative)")

    print(f"\n  ── Value Stats (NaN excluded) ────────────────────")
    if clean_data.size > 0:
        print(f"  Min            : {clean_data.min():.6f}")
        print(f"  Max            : {clean_data.max():.6f}")
        print(f"  Mean           : {clean_data.mean():.6f}")
    print(f"  Zeros          : {zero_total:,}  ({100 * zero_total / total_pixels:.1f}%)")

    print(f"\n  ── Verdict ───────────────────────────────────────")
    if nan_total > 0:
        issues.append(f"❌ {nan_total} NaN values")

    if not issues:
        print(f"  ✅ CLEAN")
    else:
        for i in issues:
            print(f"  {i}")

    return issues


# ── MAIN CHECK: 36 aligned variable files ────────────────────────────────────
overall_issues = []

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  ✗  FILE NOT FOUND: {path}")
        overall_issues.append(f"MISSING: {name}")
        continue

    # Determine the variable name (strip the year suffix)
    var_name = '_'.join(name.split('_')[:-1])  # e.g. "ERA5_U_WIND_10M_2015" → "ERA5_U_WIND_10M"

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # (bands/days, H, W)

    issues = check_array(name, data, check_negatives=(var_name in PRECIP_VARIABLES))
    if issues:
        overall_issues.append(f"{name}: {', '.join(issues)}")


# ── ELEVATION CHECK ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ELEVATION  —  SouthAsia_DEM_0p05deg.tif")
print(f"{'='*60}")

if not os.path.exists(ELEVATION_PATH):
    print(f"  ✗  FILE NOT FOUND: {ELEVATION_PATH}")
    overall_issues.append("MISSING: ELEVATION")
else:
    with rasterio.open(ELEVATION_PATH) as src:
        elev_data = src.read().astype(np.float32)   # (1, H, W) — single band DEM
        print(f"  CRS            : {src.crs}")
        print(f"  Transform      : {src.transform}")

    # For elevation, negative values are valid (below sea level), so no negative check
    issues = check_array("ELEVATION", elev_data, check_negatives=False)
    if issues:
        overall_issues.append(f"ELEVATION: {', '.join(issues)}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY  (36 variable files + 1 elevation file checked)")
print(f"{'='*60}")
if not overall_issues:
    print("  ✅ All 37 files CLEAN — no NaNs, no invalid negatives.")
else:
    print(f"  ⚠  {len(overall_issues)} file(s) with issues:\n")
    for issue in overall_issues:
        print(f"    • {issue}")