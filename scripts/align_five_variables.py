import os
import numpy as np

# ── FIX: Point PROJ to rasterio's own database ────────────────────────────────
import rasterio
rasterio_dir   = os.path.dirname(rasterio.__file__)
proj_data_path = os.path.join(rasterio_dir, 'proj_data')

if not os.path.exists(proj_data_path):
    try:
        import pyproj
        proj_data_path = pyproj.datadir.get_data_dir()
    except Exception:
        pass

os.environ['PROJ_DATA'] = proj_data_path
os.environ['PROJ_LIB']  = proj_data_path
print(f"PROJ data dir set to: {proj_data_path}")

from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

# ── PATHS ──────────────────────────────────────────────────────────────────────
INPUT_DIR_FIVEVAR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\rawFiveVar"
INPUT_DIR_MSWEP   = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\MSWEP_data"
OUTPUT_DIR        = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TARGET GRID SPEC ───────────────────────────────────────────────────────────
XMIN, YMIN, XMAX, YMAX = 76.0, 21.0, 82.4, 27.4

ERA5_W,   ERA5_H   = 64,  64   # ~0.1° resolution
CHIRPS_W, CHIRPS_H = 128, 128  # ~0.05° resolution

era5_transform   = from_bounds(XMIN, YMIN, XMAX, YMAX, ERA5_W,   ERA5_H)
chirps_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, CHIRPS_W, CHIRPS_H)

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# ── VARIABLE CATALOGUE ────────────────────────────────────────────────────────
#  All variables aligned every run — precipitation choice happens later at NPY stage.
#
#  Variable              │ Grid    │ Resampling │ Source dir
#  ──────────────────────┼─────────┼────────────┼─────────────
#  ERA5 Precipitation    │ 64×64   │ Bilinear   │ rawFiveVar
#  MSWEP Precipitation   │ 64×64   │ Bilinear   │ MSWEP_data   (filename: {year}_cropped.tif)
#  ERA5 U-wind at 10 m   │ 64×64   │ Bilinear   │ rawFiveVar
#  ERA5 V-wind at 10 m   │ 64×64   │ Bilinear   │ rawFiveVar
#  ERA5 Wind Speed 10 m  │ 64×64   │ Bilinear   │ rawFiveVar
#  CHIRPS Precipitation  │ 128×128 │ Nearest    │ rawFiveVar

# Standard variables: input filename is  {prefix}_{year}.tif
STANDARD_VARIABLES = [
    # (prefix,                 input_dir,          target_w,  target_h,  transform,        resampling)
    ('ERA5_PRECIP',         INPUT_DIR_FIVEVAR,  ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_U_WIND_10M',     INPUT_DIR_FIVEVAR,  ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_V_WIND_10M',     INPUT_DIR_FIVEVAR,  ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_WIND_SPEED_10M', INPUT_DIR_FIVEVAR,  ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('CHIRPS_PRECIP',       INPUT_DIR_FIVEVAR,  CHIRPS_W, CHIRPS_H, chirps_transform, Resampling.nearest),
]


# ── ALIGNMENT FUNCTION ────────────────────────────────────────────────────────
def align_to_exact_grid(input_path, output_path, target_w, target_h,
                        target_transform, resampling=Resampling.bilinear):
    with rasterio.open(input_path) as src:
        n_bands = src.count
        src_crs = src.crs

        dst_data = np.zeros((n_bands, target_h, target_w), dtype=np.float32)

        reproject(
            source        = rasterio.band(src, list(range(1, n_bands + 1))),
            destination   = dst_data,
            src_transform = src.transform,
            src_crs       = src_crs,
            dst_transform = target_transform,
            dst_crs       = src_crs,
            resampling    = resampling,
            src_nodata    = src.nodata,
            dst_nodata    = float('nan')
        )

        profile = {
            'driver'   : 'GTiff',
            'dtype'    : 'float32',
            'width'    : target_w,
            'height'   : target_h,
            'count'    : n_bands,
            'crs'      : src_crs,
            'transform': target_transform,
            'compress' : 'lzw',
            'nodata'   : float('nan')
        }

        band_descriptions = src.descriptions

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_data)
            dst.descriptions = band_descriptions

    print(f"  ✓  {os.path.basename(output_path)}  →  {target_w}×{target_h}, {n_bands} bands")


errors = []

# ── RUN: standard variables ───────────────────────────────────────────────────
print("\nAligning standard variables (6 years × 5 variables)...\n")

for prefix, input_dir, tw, th, transform, resamp in STANDARD_VARIABLES:
    print(f"── {prefix} ({'64×64' if tw == ERA5_W else '128×128'}, {resamp.name}) ──")
    for year in YEARS:
        in_name  = f"{prefix}_{year}.tif"
        out_name = f"{prefix}_{year}_aligned.tif"
        in_path  = os.path.join(input_dir, in_name)
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if not os.path.exists(in_path):
            print(f"  ✗  MISSING: {in_name}")
            errors.append(in_name)
            continue

        try:
            align_to_exact_grid(in_path, out_path, tw, th, transform, resamp)
        except Exception as e:
            print(f"  ✗  ERROR on {in_name}: {e}")
            errors.append(in_name)
    print()

# ── RUN: MSWEP (different directory and filename pattern) ─────────────────────
print("── MSWEP_PRECIP (64×64, bilinear) ──")

for year in YEARS:
    in_name  = f"{year}_cropped.tif"
    out_name = f"MSWEP_PRECIP_{year}_aligned.tif"
    in_path  = os.path.join(INPUT_DIR_MSWEP, in_name)
    out_path = os.path.join(OUTPUT_DIR, out_name)

    if not os.path.exists(in_path):
        print(f"  ✗  MISSING: {in_name}")
        errors.append(in_name)
        continue

    try:
        align_to_exact_grid(in_path, out_path, ERA5_W, ERA5_H, era5_transform, Resampling.bilinear)
    except Exception as e:
        print(f"  ✗  ERROR on {in_name}: {e}")
        errors.append(in_name)

print()

# ── VERIFICATION ──────────────────────────────────────────────────────────────
print("=" * 65)
print("VERIFICATION  (36 files: 6 variables × 6 years)")
print("=" * 65)

ALL_EXPECTED = (
    [(f"{prefix}_{year}_aligned.tif", tw, th)
     for prefix, _, tw, th, _, _ in STANDARD_VARIABLES
     for year in YEARS]
  + [(f"MSWEP_PRECIP_{year}_aligned.tif", ERA5_W, ERA5_H)
     for year in YEARS]
)

failed = []
for out_name, exp_w, exp_h in ALL_EXPECTED:
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if not os.path.exists(out_path):
        print(f"  ✗  NOT FOUND: {out_name}")
        failed.append(out_name)
        continue
    with rasterio.open(out_path) as src:
        ok = src.width == exp_w and src.height == exp_h and src.dtypes[0] == 'float32'
        status = "✓" if ok else "✗"
        print(f"  {status}  {out_name}  {src.width}×{src.height}  {src.count} bands")
        if not ok:
            failed.append(out_name)

print()
if errors:
    print(f"⚠  Missing / errored inputs ({len(errors)}): {errors}")
if failed:
    print(f"⚠  Verification failed ({len(failed)}): {failed}")
if not errors and not failed:
    print("✓  All 36 files aligned and verified successfully.")
    print("   ERA5_PRECIP, MSWEP_PRECIP, U-wind, V-wind, Wind Speed → 64×64")
    print("   CHIRPS_PRECIP → 128×128")