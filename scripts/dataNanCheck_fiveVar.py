import numpy as np
import rasterio
import os

OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

VARIABLES = [
    'ERA5_PRECIP',
    'MSWEP_PRECIP',
    'ERA5_U_WIND_10M',
    'ERA5_V_WIND_10M',
    'ERA5_WIND_SPEED_10M',
    'CHIRPS_PRECIP',
]

# Build the full list of 30 files
files = {}
for var in VARIABLES:
    for year in YEARS:
        key  = f"{var}_{year}"
        path = os.path.join(OUTPUT_DIR, f"{key}_aligned.tif")
        files[key] = path

# ── NaN CHECK ─────────────────────────────────────────────────────────────────
overall_issues = []

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  ✗  FILE NOT FOUND: {path}")
        overall_issues.append(f"MISSING: {name}")
        continue

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # shape: (bands, H, W)

    total_pixels = data.size

    # ── NaN CHECK ─────────────────────────────────────────────
    nan_mask  = np.isnan(data)
    nan_total = nan_mask.sum()
    nan_days  = np.where(nan_mask.any(axis=(1, 2)))[0]

    # ── NEGATIVE CHECK ────────────────────────────────────────
    neg_mask  = data < 0
    neg_total = neg_mask.sum()
    neg_days  = np.where(neg_mask.any(axis=(1, 2)))[0]

    # ── ZERO CHECK (informational) ────────────────────────────
    zero_total = (data == 0).sum()

    # ── STATS (NaN excluded) ──────────────────────────────────
    clean_data = data[~nan_mask]

    print(f"  Shape          : {data.shape}  →  (days, H, W)")
    print(f"  Total pixels   : {total_pixels:,}")

    print(f"\n  ── NaN ──────────────────────────────────────────")
    print(f"  NaN count      : {nan_total}")
    print(f"  NaN days       : {len(nan_days)}  →  {nan_days.tolist() if len(nan_days) <= 10 else str(nan_days[:10].tolist()) + '...'}")

    print(f"\n  ── Negatives ────────────────────────────────────")
    print(f"  Negative count : {neg_total}")
    print(f"  Negative days  : {len(neg_days)}  →  {neg_days.tolist() if len(neg_days) <= 10 else str(neg_days[:10].tolist()) + '...'}")
    if neg_total > 0:
        print(f"  Min negative   : {data[neg_mask].min():.6f}")

    print(f"\n  ── Value Stats (NaN excluded) ────────────────────")
    if clean_data.size > 0:
        print(f"  Min            : {clean_data.min():.6f}")
        print(f"  Max            : {clean_data.max():.6f}")
        print(f"  Mean           : {clean_data.mean():.6f}")
    print(f"  Zeros          : {zero_total:,}  ({100*zero_total/total_pixels:.1f}%)")

    print(f"\n  ── Verdict ───────────────────────────────────────")
    issues = []
    if nan_total > 0:  issues.append(f"❌ {nan_total} NaN values")
    if neg_total > 0:  issues.append(f"❌ {neg_total} negative values")
    if not issues:
        print(f"  ✅ CLEAN — no NaNs, no negatives")
    else:
        for i in issues:
            print(f"  {i}")
        overall_issues.append(f"{name}: {', '.join(issues)}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SUMMARY  ({len(files)} files checked)")
print(f"{'='*60}")
if not overall_issues:
    print("  ✅ All 36 files CLEAN — no NaNs, no negatives.")
else:
    print(f"  ⚠  {len(overall_issues)} file(s) with issues:\n")
    for issue in overall_issues:
        print(f"    • {issue}")