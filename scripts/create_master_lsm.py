import os
import rasterio
import numpy as np

# ── LOCAL PATHS ─────────────────────────────────────────────────────────────
ALIGNED_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
OUTPUT_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\SRTM"

# We use 2015 as the structural anchor. 
# Make sure these filenames match exactly what your alignment script produced!
ERA5_PATH   = os.path.join(ALIGNED_DIR, "ERA5_PRECIP_2015_aligned.tif")      # 384 x 384
CHIRPS_PATH = os.path.join(ALIGNED_DIR, "CHIRPS_PRECIP_2015_aligned.tif")    # 768 x 768

print("="*70)
print("🏗️ GENERATING RESEARCH-BACKED MASTER LAND-SEA MASK")
print("="*70)

# ── 1. LOAD NATIVE GRID VALIDITY TEMPLATES ──────────────────────────────────
with rasterio.open(ERA5_PATH) as src_lr:
    era5_band = src_lr.read(1)
    # True where ERA5-Land has real data (Not NaN)
    era5_valid_lr = ~np.isnan(era5_band)

with rasterio.open(CHIRPS_PATH) as src_hr:
    chirps_band = src_hr.read(1)
    # True where native high-res CHIRPS targets are valid (Not NaN)
    chirps_valid_hr = ~np.isnan(chirps_band)

# ── 2. GENERATE THE LOW-RES MASK INTERSECTION ───────────────────────────────
# Downsample the fine CHIRPS validation grid to 0.1° using stride checking
chirps_valid_lr = chirps_valid_hr[::2, ::2]

# A coarse cell is land ONLY if both datasets contain valid data
lsm_coarse_lr = (era5_valid_lr & chirps_valid_lr).astype(np.float32)

print(f"1. Coarse 0.1° Mask Grid Computed: {lsm_coarse_lr.shape}")
print(f"   -> Total Coarse Land Cells: {int(np.sum(lsm_coarse_lr))}")

# ── 3. UPSAMPLE COARSE MASK TO 0.05° (2x2 PIXEL REPLICATION) ────────────────
# np.kron expands each pixel into an identical 2x2 sub-grid block
lsm_upsampled_hr = np.kron(lsm_coarse_lr, np.ones((2, 2), dtype=np.float32))
print(f"2. Upsampled 0.05° Block Mask Shaped: {lsm_upsampled_hr.shape}")

# ── 4. APPLY THE NATIVE CHIRPS CORRECTION RULE ──────────────────────────────
# Cleanly mask out any sub-pixels where native CHIRPS targets are NaN
master_lsm_hr = lsm_upsampled_hr * chirps_valid_hr.astype(np.float32)

# Derive the final matching low-res mask directly from our corrected master
master_lsm_lr = master_lsm_hr[::2, ::2]

print(f"3. Final Shoreline Correction Applied via Native CHIRPS 0.05°")
print(f"   -> Final Master HR Land Pixels (768x768): {int(np.sum(master_lsm_hr))}")
print(f"   -> Final Master LR Land Pixels (384x384): {int(np.sum(master_lsm_lr))}")

# ── 5. SAVE AS STATIC MASTER ASSETS ─────────────────────────────────────────
hr_out = os.path.join(OUTPUT_DIR, 'india_lsm_HR.npy')
lr_out = os.path.join(OUTPUT_DIR, 'india_lsm_LR.npy')

np.save(hr_out, master_lsm_hr)
np.save(lr_out, master_lsm_lr)

print(f"\n✅ Success! Master masks saved in: {OUTPUT_DIR}")
print("="*70)