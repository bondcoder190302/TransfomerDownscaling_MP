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
INPUT_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\rawFiveVar"
OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TARGET GRID SPEC ───────────────────────────────────────────────────────────
XMIN, YMIN, XMAX, YMAX = 76.0, 21.0, 82.4, 27.4

ERA5_W,   ERA5_H   = 64,  64   # ~0.1° resolution
CHIRPS_W, CHIRPS_H = 128, 128  # ~0.05° resolution

era5_transform   = from_bounds(XMIN, YMIN, XMAX, YMAX, ERA5_W,   ERA5_H)
chirps_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, CHIRPS_W, CHIRPS_H)

# ── VARIABLE CATALOGUE ────────────────────────────────────────────────────────
#
#  Resampling method rationale:
#
#  All five variables are spatially continuous scalar fields, so bilinear
#  interpolation (weighted average of the 4 nearest source pixels) is
#  appropriate for all of them.  It preserves smooth spatial gradients
#  and avoids the blocky artefacts of nearest-neighbour.
#
#  Nearest-neighbour is only preferred for:
#    • Categorical / classified data (land-use, soil type, etc.)
#    • Data where pixel-centre values must not be mixed (e.g. integer IDs)
#  None of our five variables fall into those categories.
#
#  Variable              │ Grid   │ Resampling  │ Reason
#  ──────────────────────┼────────┼─────────────┼───────────────────────────────
#  ERA5 Precipitation    │ 64×64  │ Bilinear    │ Continuous flux field
#  ERA5 U-wind at 10 m   │ 64×64  │ Bilinear    │ Continuous signed wind field
#  ERA5 V-wind at 10 m   │ 64×64  │ Bilinear    │ Continuous signed wind field
#  ERA5 Wind Speed 10 m  │ 64×64  │ Bilinear    │ Continuous positive scalar
#  CHIRPS Precipitation  │ 128×128│ Nearest     │ Preserve original pixel values (user preference)
#
VARIABLES = [
    # (filename_prefix,          target_w,  target_h,  target_transform,  resampling)
    ('ERA5_PRECIP',          ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_U_WIND_10M',      ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_V_WIND_10M',      ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_WIND_SPEED_10M',  ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('CHIRPS_PRECIP',        CHIRPS_W, CHIRPS_H, chirps_transform, Resampling.nearest),
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# ── ALIGNMENT FUNCTION ────────────────────────────────────────────────────────
def align_to_exact_grid(input_path, output_path, target_w, target_h,
                         target_transform, resampling=Resampling.bilinear):
    with rasterio.open(input_path) as src:
        n_bands = src.count
        src_crs = src.crs

        dst_data = np.zeros((n_bands, target_h, target_w), dtype=np.float32)

        reproject(
            source      = rasterio.band(src, list(range(1, n_bands + 1))),
            destination = dst_data,
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

        # ── copy band descriptions (YYYY_MM_DD) from source to destination ──────
        # src.descriptions is a tuple of the GEE-exported band name strings.
        # Assigning to dst.descriptions writes them as GDAL band descriptions,
        # which is the same field rasterio reads back via src.descriptions.
        # update_tags() writes to metadata tags only and does NOT affect this field.
        band_descriptions = src.descriptions   # tuple of 'YYYY_MM_DD' strings

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_data)
            dst.descriptions = band_descriptions   # ← correct way to preserve band names

    print(f"  ✓  {os.path.basename(output_path)}  →  {target_w}×{target_h}, {n_bands} bands")


# ── MAIN RUN ──────────────────────────────────────────────────────────────────
print("\nStarting alignment of 30 files (6 years × 5 variables)...\n")

errors = []

for prefix, tw, th, transform, resamp in VARIABLES:
    print(f"── {prefix} ({'ERA5 64×64' if tw == ERA5_W else 'CHIRPS 128×128'}, {resamp.name}) ──")
    for year in YEARS:
        in_name  = f"{prefix}_{year}.tif"
        out_name = f"{prefix}_{year}_aligned.tif"
        in_path  = os.path.join(INPUT_DIR,  in_name)
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

# ── VERIFICATION ──────────────────────────────────────────────────────────────
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

failed = []
for prefix, tw, th, _, _ in VARIABLES:
    for year in YEARS:
        out_name = f"{prefix}_{year}_aligned.tif"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if not os.path.exists(out_path):
            continue
        with rasterio.open(out_path) as src:
            ok = src.width == tw and src.height == th and src.dtypes[0] == 'float32'
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
    print("✓  All 30 files aligned and verified successfully.")