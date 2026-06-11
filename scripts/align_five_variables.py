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

# ── PATHS ─────────────────────────────────────────────────────────────────────
INPUT_DIR_FIVEVAR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\rawFiveVar"
INPUT_DIR_MSWEP   = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\MSWEP_data"
OUTPUT_DIR        = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TARGET GRID SPEC — anchored to Elevation reference layer ──────────────────
#
#  Elevation (reference):
#    Extent : (62.2998717527286487, 1.3000558348497222)
#           → (100.6998717527286544, 39.7000558348497279)
#    Size   : 768 × 768  →  pixel ≈ 0.05°
#
#  Derived grids:
#    ERA5 / MSWEP  →  384 × 384  (pixel ≈ 0.10°, ERA5-Land native resolution)
#    CHIRPS        →  768 × 768  (pixel ≈ 0.05°, matches elevation exactly)

XMIN = 62.2998717527286487
YMIN =  1.3000558348497222
XMAX = 100.6998717527286544
YMAX = 39.7000558348497279

ERA5_W,   ERA5_H   = 384, 384
CHIRPS_W, CHIRPS_H = 768, 768

era5_transform   = from_bounds(XMIN, YMIN, XMAX, YMAX, ERA5_W,   ERA5_H)
chirps_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, CHIRPS_W, CHIRPS_H)

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# ── VARIABLE CATALOGUE ────────────────────────────────────────────────────────
#
#  Columns:
#   prefix       – output filename prefix; also used as {prefix} in in_pattern
#   input_dir    – folder containing the source .tif
#   in_pattern   – format string for the INPUT filename
#                  placeholders available: {prefix}  {year}
#                  (MSWEP uses a different pattern: {year}_cropped.tif)
#   tw / th      – output pixel dimensions
#   transform    – affine transform for the target grid
#   resampling   – rasterio Resampling method
#
#  Variable               │ Output grid │ Resampling │ Source dir
#  ───────────────────────┼─────────────┼────────────┼──────────────────
#  ERA5 Precipitation     │  384 × 384  │ Bilinear   │ rawFiveVar
#  ERA5 U-wind at 10 m    │  384 × 384  │ Bilinear   │ rawFiveVar
#  ERA5 V-wind at 10 m    │  384 × 384  │ Bilinear   │ rawFiveVar
#  ERA5 Wind Speed 10 m   │  384 × 384  │ Bilinear   │ rawFiveVar
#  MSWEP Precipitation    │  384 × 384  │ Bilinear   │ MSWEP_data
#  CHIRPS Precipitation   │  768 × 768  │ Nearest    │ rawFiveVar

VARIABLE_CATALOGUE = [
    # (prefix,                  input_dir,          in_pattern,              tw,        th,        transform,        resampling          )
    ('ERA5_PRECIP',          INPUT_DIR_FIVEVAR,  '{prefix}_{year}.tif',   ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_U_WIND_10M',      INPUT_DIR_FIVEVAR,  '{prefix}_{year}.tif',   ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_V_WIND_10M',      INPUT_DIR_FIVEVAR,  '{prefix}_{year}.tif',   ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('ERA5_WIND_SPEED_10M',  INPUT_DIR_FIVEVAR,  '{prefix}_{year}.tif',   ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('MSWEP_PRECIP',         INPUT_DIR_MSWEP,    '{year}_cropped.tif',    ERA5_W,   ERA5_H,   era5_transform,   Resampling.bilinear),
    ('CHIRPS_PRECIP',        INPUT_DIR_FIVEVAR,  '{prefix}_{year}.tif',   CHIRPS_W, CHIRPS_H, chirps_transform, Resampling.bilinear),
]

TOTAL_EXPECTED = len(VARIABLE_CATALOGUE) * len(YEARS)   # 6 × 6 = 36


# ── ALIGNMENT FUNCTION ────────────────────────────────────────────────────────
def align_to_exact_grid(input_path, output_path,
                        target_w, target_h, target_transform,
                        resampling=Resampling.bilinear):
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
            dst_nodata    = float('nan'),
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
            'nodata'   : float('nan'),
        }

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dst_data)
            dst.descriptions = src.descriptions

    print(f"  ✓  {os.path.basename(output_path)}  →  {target_w}×{target_h}, {n_bands} band(s)")


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
errors = []

print("\n" + "=" * 65)
print("ALIGNMENT RUN")
print("=" * 65)
print(f"Reference extent  :  ({XMIN}, {YMIN})  →  ({XMAX}, {YMAX})")
print(f"ERA5 / MSWEP grid :  {ERA5_W} × {ERA5_H}  (~0.10° pixel)")
print(f"CHIRPS grid       :  {CHIRPS_W} × {CHIRPS_H}  (~0.05° pixel, matches elevation)")
print(f"Total files       :  {len(VARIABLE_CATALOGUE)} variables × {len(YEARS)} years = {TOTAL_EXPECTED}\n")

for prefix, input_dir, in_pattern, tw, th, transform, resamp in VARIABLE_CATALOGUE:
    print(f"── {prefix:<25s}  ({tw}×{th}, {resamp.name}) ──")
    for year in YEARS:
        in_name  = in_pattern.format(prefix=prefix, year=year)
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


# ── VERIFICATION ──────────────────────────────────────────────────────────────
print("=" * 65)
print(f"VERIFICATION  ({TOTAL_EXPECTED} files: {len(VARIABLE_CATALOGUE)} variables × {len(YEARS)} years)")
print("=" * 65)

ALL_EXPECTED = [
    (f"{prefix}_{year}_aligned.tif", tw, th)
    for prefix, _, _, tw, th, _, _ in VARIABLE_CATALOGUE
    for year in YEARS
]

failed = []
for out_name, exp_w, exp_h in ALL_EXPECTED:
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if not os.path.exists(out_path):
        print(f"  ✗  NOT FOUND : {out_name}")
        failed.append(out_name)
        continue
    with rasterio.open(out_path) as src:
        size_ok  = (src.width == exp_w) and (src.height == exp_h)
        dtype_ok = src.dtypes[0] == 'float32'
        ok       = size_ok and dtype_ok
        tag      = "✓" if ok else "✗  ← size/dtype mismatch"
        print(f"  {tag}  {out_name:<48s}  {src.width}×{src.height}  {src.count} band(s)")
        if not ok:
            failed.append(out_name)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"⚠  Missing / errored inputs  ({len(errors)}):")
    for e in errors:
        print(f"     {e}")
if failed:
    print(f"⚠  Verification failures  ({len(failed)}):")
    for f in failed:
        print(f"     {f}")
if not errors and not failed:
    print(f"✓  All {TOTAL_EXPECTED} files aligned and verified successfully.\n")
    print("   ERA5_PRECIP         →  384×384   (input,  ~0.10°)")
    print("   ERA5_U_WIND_10M     →  384×384   (input,  ~0.10°)")
    print("   ERA5_V_WIND_10M     →  384×384   (input,  ~0.10°)")
    print("   ERA5_WIND_SPEED_10M →  384×384   (input,  ~0.10°)")
    print("   MSWEP_PRECIP        →  384×384   (input,  ~0.10°)")
    print("   CHIRPS_PRECIP       →  768×768   (target, ~0.05°, matches elevation)")