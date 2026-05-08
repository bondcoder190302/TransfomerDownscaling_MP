import numpy as np
import rasterio
import os

# ── PATHS ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"

# ── OUTPUT FOLDER MAPPING ──────────────────────────────────────────────────────
# (tif_prefix, output_subfolder)
# DEM is intentionally excluded — already present in output directory.
VARIABLE_MAP = [
    ('ERA5_PRECIP',         'ERA5_precip_cut'),
    ('CHIRPS_PRECIP',       'CHIRPS_precip_cut_obs'),
    ('ERA5_U_WIND_10M',     'ERA5_u_wind_10m_cut'),
    ('ERA5_V_WIND_10M',     'ERA5_v_wind_10m_cut'),
    ('ERA5_WIND_SPEED_10M', 'ERA5_wind_speed_10m_cut'),
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# Create all output subfolders upfront
for _, subfolder in VARIABLE_MAP:
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)


# ── CONVERSION FUNCTION ────────────────────────────────────────────────────────
def tif_to_daily_npy(tif_path, output_dir):
    """
    Read a multi-band aligned .tif and save each band as a separate .npy file.

    Band descriptions stored in the TIF (format: YYYY_MM_DD) are used directly
    as filenames (converted to YYYYMMDD.npy) — no external date list needed.

    Values are saved as-is in float32 — no rounding, no scaling.
    """
    with rasterio.open(tif_path) as src:
        n_bands      = src.count
        descriptions = src.descriptions   # tuple of 'YYYY_MM_DD' strings

        print(f"  Bands : {n_bands}  |  Shape/band : ({src.height} × {src.width})"
              f"  |  Dtype : {src.dtypes[0]}")

        # Verify band descriptions are present
        if any(d is None for d in descriptions):
            raise ValueError(
                f"Some band descriptions are missing in {tif_path}.\n"
                f"Expected YYYY_MM_DD strings in all band descriptions."
            )

        for i, desc in enumerate(descriptions):
            # desc is 'YYYY_MM_DD' → convert to 'YYYYMMDD' for filename
            date_str = desc.replace('_', '')          # '2020_01_01' → '20200101'

            band_data = src.read(i + 1).astype(np.float32)   # (H, W), full precision

            out_path = os.path.join(output_dir, f"{date_str}.npy")
            np.save(out_path, band_data)

            if (i + 1) % 60 == 0 or (i + 1) == n_bands:
                print(f"    [{i+1:3d}/{n_bands}]  {date_str}.npy  "
                      f"min={np.nanmin(band_data):.6f}  "
                      f"max={np.nanmax(band_data):.6f}")


# ── MAIN RUN ──────────────────────────────────────────────────────────────────
print(f"\nConverting {len(VARIABLE_MAP)} variables × {len(YEARS)} years "
      f"= {len(VARIABLE_MAP) * len(YEARS)} TIF files\n")

overall_errors = []

for prefix, subfolder in VARIABLE_MAP:
    out_dir = os.path.join(OUTPUT_DIR, subfolder)

    print(f"\n{'='*65}")
    print(f"  Variable : {prefix}")
    print(f"  Output   : {subfolder}")
    print(f"{'='*65}")

    for year in YEARS:
        tif_name = f"{prefix}_{year}_aligned.tif"
        tif_path = os.path.join(INPUT_DIR, tif_name)

        print(f"\n  ── {year} ── {tif_name}")

        if not os.path.exists(tif_path):
            msg = f"MISSING: {tif_name}"
            print(f"  ✗  {msg}")
            overall_errors.append(msg)
            continue

        try:
            tif_to_daily_npy(tif_path, out_dir)
            print(f"  ✓  {year} done")
        except Exception as e:
            msg = f"ERROR in {tif_name}: {e}"
            print(f"  ✗  {msg}")
            overall_errors.append(msg)


# ── FINAL VERIFICATION ────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print(f"  FINAL VERIFICATION")
print(f"{'='*65}\n")

# Expected counts per variable (6 years: 2015–2020)
# Leap years: 2016, 2020 → 366 days each; rest → 365
EXPECTED_COUNT = 365*4 + 366*2   # 2192 total .npy files per variable

for prefix, subfolder in VARIABLE_MAP:
    out_dir   = os.path.join(OUTPUT_DIR, subfolder)
    npy_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npy')])
    count     = len(npy_files)
    status    = "✓" if count == EXPECTED_COUNT else "✗"

    print(f"  {status}  {subfolder:<30}  {count} files  (expected {EXPECTED_COUNT})")

    if count > 0:
        # Spot check first and last file
        for fname in [npy_files[0], npy_files[-1]]:
            arr = np.load(os.path.join(out_dir, fname))
            print(f"       {fname}  shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={np.nanmin(arr):.6f}  max={np.nanmax(arr):.6f}")

if overall_errors:
    print(f"\n⚠  {len(overall_errors)} error(s):")
    for e in overall_errors:
        print(f"    • {e}")
else:
    print(f"\n✓  All conversions completed with no errors.")

print(f"""
  Output structure:
  NPY_new_128/
  ├── ERA5_precip_cut/              →  2192 × (64,  64)  float32
  ├── CHIRPS_precip_cut_obs/        →  2192 × (128, 128) float32
  ├── ERA5_u_wind_10m_cut/          →  2192 × (64,  64)  float32
  ├── ERA5_v_wind_10m_cut/          →  2192 × (64,  64)  float32
  ├── ERA5_wind_speed_10m_cut/      →  2192 × (64,  64)  float32
  └── HGT_fix_cut_obs/              →  (already present, untouched)
""")
