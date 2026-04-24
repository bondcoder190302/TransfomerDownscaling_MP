import numpy as np
import rasterio
import os
from datetime import date, timedelta

# ── PATHS ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_from_raw"
DEM_PATH   = os.path.join(INPUT_DIR, "DEM_128x128.tif")
OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"

CHIRPS_OUT = os.path.join(OUTPUT_DIR, "CHIRPS_precip_cut_obs")
ERA5_OUT   = os.path.join(OUTPUT_DIR, "ERA5_precip_cut")
HGT_OUT    = os.path.join(OUTPUT_DIR, "HGT_fix_cut_obs")

for folder in [CHIRPS_OUT, ERA5_OUT, HGT_OUT]:
    os.makedirs(folder, exist_ok=True)

# ── DATE LIST FOR 2020 (leap year → 366 days) ─────────────────────────────────
start = date(2020, 1, 1)
dates = [start + timedelta(days=i) for i in range(366)]  # 2020-01-01 … 2020-12-31
date_strings = [d.strftime('%Y%m%d') for d in dates]     # '20200101', '20200102', …


def tif_to_daily_npy(tif_path, output_dir, date_strings, dataset_name):
    """
    Read a multi-band .tif (bands = days) and save each band
    as a separate .npy file named by date string.
    """
    print(f"\n{'='*55}")
    print(f"  Converting: {os.path.basename(tif_path)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*55}")

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        assert n_bands == len(date_strings), \
            f"Band count ({n_bands}) != date count ({len(date_strings)}). Check your tif."

        print(f"  Bands      : {n_bands}")
        print(f"  Shape/band : ({src.height}, {src.width})")
        print(f"  Dtype      : {src.dtypes[0]}")
        print(f"  Saving...")

        for i, date_str in enumerate(date_strings):
            band_data = src.read(i + 1).astype(np.float32)  # shape: (H, W)
            out_path  = os.path.join(output_dir, f"{date_str}.npy")
            np.save(out_path, band_data)

            if (i + 1) % 50 == 0 or (i + 1) == n_bands:
                print(f"  [{i+1:3d}/{n_bands}] saved {date_str}.npy  "
                      f"shape={band_data.shape}  "
                      f"min={np.nanmin(band_data):.4f}  "
                      f"max={np.nanmax(band_data):.4f}")

    print(f"  ✓ {dataset_name} done — {n_bands} .npy files saved.")


def tif_to_single_npy(tif_path, output_dir, filename):
    """
    Read a single-band .tif (DEM) and save as one .npy file.
    """
    print(f"\n{'='*55}")
    print(f"  Converting: {os.path.basename(tif_path)}")
    print(f"  Output    : {filename}")
    print(f"{'='*55}")

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)  # shape: (H, W)

    out_path = os.path.join(output_dir, filename)
    np.save(out_path, data)

    print(f"  Shape : {data.shape}")
    print(f"  Min   : {np.nanmin(data):.2f} m")
    print(f"  Max   : {np.nanmax(data):.2f} m")
    print(f"  Mean  : {np.nanmean(data):.2f} m")
    print(f"  ✓ Saved → {out_path}")


# ── RUN CONVERSIONS ───────────────────────────────────────────────────────────
tif_to_daily_npy(
    tif_path    = os.path.join(INPUT_DIR, 'ERA5_64x64_2020.tif'),
    output_dir  = ERA5_OUT,
    date_strings= date_strings,
    dataset_name= 'ERA5'
)

tif_to_daily_npy(
    tif_path    = os.path.join(INPUT_DIR, 'CHIRPS_128x128_2020.tif'),
    output_dir  = CHIRPS_OUT,
    date_strings= date_strings,
    dataset_name= 'CHIRPS'
)

tif_to_single_npy(
    tif_path   = DEM_PATH,
    output_dir = HGT_OUT,
    filename   = 'HGT_fix.npy'
)


# ── FINAL VERIFICATION ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  FINAL CHECK")
print(f"{'='*55}")

era5_files   = sorted(os.listdir(ERA5_OUT))
chirps_files = sorted(os.listdir(CHIRPS_OUT))
hgt_files    = os.listdir(HGT_OUT)

print(f"\n  ERA5   files : {len(era5_files)}   (expected 366)")
print(f"  CHIRPS files : {len(chirps_files)}  (expected 366)")
print(f"  HGT    files : {hgt_files}")

# Spot check: first, middle, last day
for label, folder, fname in [
    ('ERA5   first', ERA5_OUT,   era5_files[0]),
    ('ERA5   last ',  ERA5_OUT,  era5_files[-1]),
    ('CHIRPS first', CHIRPS_OUT, chirps_files[0]),
    ('CHIRPS last ', CHIRPS_OUT, chirps_files[-1]),
]:
    arr = np.load(os.path.join(folder, fname))
    print(f"  {label} → {fname}  shape={arr.shape}  dtype={arr.dtype}  "
          f"min={np.nanmin(arr):.4f}  max={np.nanmax(arr):.4f}")

# HGT check
hgt = np.load(os.path.join(HGT_OUT, 'HGT_fix.npy'))
print(f"  HGT          → HGT_fix.npy  shape={hgt.shape}  dtype={hgt.dtype}  "
      f"min={np.nanmin(hgt):.2f}  max={np.nanmax(hgt):.2f}")

print(f"\n✓ All conversions complete.")
print(f"\n  Output structure:")
print(f"  NPY_new_128/")
print(f"  ├── ERA5_precip_cut/         → 366 × (64,64)   float32")
print(f"  ├── CHIRPS_precip_cut_obs/   → 366 × (128,128) float32")
print(f"  └── HGT_fix_cut_obs/         → 1   × (128,128) float32")