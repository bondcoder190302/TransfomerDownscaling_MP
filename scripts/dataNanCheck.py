import numpy as np
import rasterio
import os

OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_from_raw"

files = {
    'ERA5':   os.path.join(OUTPUT_DIR, 'ERA5_64x64_2020.tif'),
    'CHIRPS': os.path.join(OUTPUT_DIR, 'CHIRPS_128x128_2020.tif'),
}

for name, path in files.items():
    print(f"\n{'='*55}")
    print(f"  {name}: {os.path.basename(path)}")
    print(f"{'='*55}")

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # shape: (366, H, W)

    total_pixels = data.size

    # ── NaN CHECK ─────────────────────────────────────────────
    nan_mask        = np.isnan(data)
    nan_total       = nan_mask.sum()
    nan_days        = np.where(nan_mask.any(axis=(1, 2)))[0]

    # ── NEGATIVE CHECK ────────────────────────────────────────
    neg_mask        = data < 0
    neg_total       = neg_mask.sum()
    neg_days        = np.where(neg_mask.any(axis=(1, 2)))[0]

    # ── ZERO CHECK (informational, not necessarily bad) ───────
    zero_total      = (data == 0).sum()

    # ── STATS ─────────────────────────────────────────────────
    clean_data      = data[~nan_mask]  # exclude NaNs for stats

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
    print(f"  Min            : {clean_data.min():.6f}")
    print(f"  Max            : {clean_data.max():.6f}")
    print(f"  Mean           : {clean_data.mean():.6f}")
    print(f"  Zeros          : {zero_total:,}  ({100*zero_total/total_pixels:.1f}%)  ← dry days, expected")

    print(f"\n  ── Verdict ───────────────────────────────────────")
    issues = []
    if nan_total > 0:   issues.append(f"❌ {nan_total} NaN values found")
    if neg_total > 0:   issues.append(f"❌ {neg_total} negative values found")
    if not issues:
        print(f"  ✅ CLEAN — no NaNs, no negatives")
    else:
        for i in issues:
            print(f"  {i}")