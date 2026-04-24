import os
import sys

# ── FIX: Point PROJ to rasterio's own database, not PostgreSQL's ──────────────
# Find rasterio's PROJ data directory and force it before any imports
import rasterio
rasterio_dir = os.path.dirname(rasterio.__file__)
proj_data_path = os.path.join(rasterio_dir, 'proj_data')

# Fallback: find it via pyproj if above doesn't exist
if not os.path.exists(proj_data_path):
    try:
        import pyproj
        proj_data_path = pyproj.datadir.get_data_dir()
    except Exception:
        pass

os.environ['PROJ_DATA'] = proj_data_path
os.environ['PROJ_LIB']  = proj_data_path  # older rasterio versions use this key
print(f"PROJ data dir set to: {proj_data_path}")

# ── NOW safe to import rasterio submodules ─────────────────────────────────────
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

# ── PATHS ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64"
OUTPUT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_from_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TARGET SPEC ────────────────────────────────────────────────────────────────
XMIN, YMIN, XMAX, YMAX = 76.0, 21.0, 82.4, 27.4

ERA5_W,   ERA5_H   = 64,  64
CHIRPS_W, CHIRPS_H = 128, 128

era5_transform   = from_bounds(XMIN, YMIN, XMAX, YMAX, ERA5_W,   ERA5_H)
chirps_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, CHIRPS_W, CHIRPS_H)


def align_to_exact_grid(input_path, output_path, target_w, target_h,
                        target_transform, resampling=Resampling.bilinear):
    with rasterio.open(input_path) as src:
        n_bands = src.count
        src_crs = src.crs

        dst_data = np.zeros((n_bands, target_h, target_w), dtype=np.float32)

        reproject(
            source=rasterio.band(src, list(range(1, n_bands + 1))),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=src_crs,          # ← use src_crs object directly, avoids EPSG string lookup
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=float('nan')
        )

        profile = {
            'driver':    'GTiff',
            'dtype':     'float32',
            'width':     target_w,
            'height':    target_h,
            'count':     n_bands,
            'crs':       src_crs,     # ← same: reuse CRS object, no string parsing
            'transform': target_transform,
            'compress':  'lzw',
            'nodata':    float('nan')
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_data)

    print(f"✓ Saved {output_path}  →  {target_w}×{target_h}, {n_bands} bands")


# ── RUN ────────────────────────────────────────────────────────────────────────
align_to_exact_grid(
    input_path       = os.path.join(INPUT_DIR, 'ERA5_RAW_2020.tif'),
    output_path      = os.path.join(OUTPUT_DIR, 'ERA5_64x64_2020.tif'),
    target_w         = ERA5_W,
    target_h         = ERA5_H,
    target_transform = era5_transform,
    resampling       = Resampling.bilinear
)

align_to_exact_grid(
    input_path       = os.path.join(INPUT_DIR, 'CHIRPS_RAW_2020.tif'),
    output_path      = os.path.join(OUTPUT_DIR, 'CHIRPS_128x128_2020.tif'),
    target_w         = CHIRPS_W,
    target_h         = CHIRPS_H,
    target_transform = chirps_transform,
    resampling       = Resampling.nearest
)
# After downloading COP30_DEM_RAW.tif, align to 128x128 (CHIRPS grid)
align_to_exact_grid(
    input_path       = os.path.join(INPUT_DIR, 'COP30_DEM_RAW.tif'),
    output_path      = os.path.join(OUTPUT_DIR, 'DEM_128x128.tif'),
    target_w         = 128,
    target_h         = 128,
    target_transform = chirps_transform,   # reuse from previous script
    resampling       = Resampling.bilinear # bilinear fine for elevation
)

# ── VERIFICATION ───────────────────────────────────────────────────────────────
for path, expected_w, expected_h in [
    (os.path.join(OUTPUT_DIR, 'ERA5_64x64_2020.tif'),     64,  64),
    (os.path.join(OUTPUT_DIR, 'CHIRPS_128x128_2020.tif'), 128, 128)
]:
    with rasterio.open(path) as src:
        print(f"\n{os.path.basename(path)}")
        print(f"  Size     : {src.width} × {src.height}  (expected {expected_w}×{expected_h})")
        print(f"  Extent   : {src.bounds}")
        print(f"  Dtype    : {src.dtypes[0]}")
        print(f"  Transform: {src.transform}")
        assert src.width  == expected_w, "WIDTH MISMATCH!"
        assert src.height == expected_h, "HEIGHT MISMATCH!"
        assert src.dtypes[0] == 'float32', "DTYPE MISMATCH!"

print("\n✓ All checks passed — datasets aligned and saved.")