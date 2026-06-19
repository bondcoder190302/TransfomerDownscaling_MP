import os
import rasterio
import numpy as np
import torch
import pickle
from datetime import date, timedelta
from tqdm import tqdm

# ── 1. FOLDER PATHS ───────────────────────────────────────────────────────────
ALIGNED_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\aligned_fiveVar"
DEM_PATH    = r"C:\Users\HP\Downloads\MTP Phase 2\All set dataset\here128and64\SRTM\SouthAsia_DEM_0p05deg.tif"

OUTPUT_NPY_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"
OUTPUT_PKL_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\param_stat"

os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)
os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018, 2019, 2020]

# ── 2. VARIABLE CONFIGURATION ─────────────────────────────────────────────────
# Maps the dataset prefix to its target folder, log1p requirement, and stat name
VARIABLE_CONFIG = {
    # 'ERA5_PRECIP': {
    #     'out_folder': 'ERA5_precip_cut_log1p',
    #     'log1p': True,
    #     'stat_name': 'ERA5_precip_log1p_stat.pkl'
    # },
    'MSWEP_PRECIP': {
        'out_folder': 'MSWEP_precip_cut_log1p',
        'log1p': True,
        'stat_name': 'MSWEP_precip_log1p_stat.pkl'
    },
    'CHIRPS_PRECIP': {
        'out_folder': 'CHIRPS_precip_cut_obs_log1p',
        'log1p': True,
        'stat_name': 'CHIRPS_precip_obs_log1p_stat.pkl'
    },
    'ERA5_U_WIND_10M': {
        'out_folder': 'ERA5_u_wind_10m_cut',
        'log1p': False,
        'stat_name': 'ERA5_u_wind_10m_stat.pkl'
    },
    'ERA5_V_WIND_10M': {
        'out_folder': 'ERA5_v_wind_10m_cut',
        'log1p': False,
        'stat_name': 'ERA5_v_wind_10m_stat.pkl'
    },
    'ERA5_WIND_SPEED_10M': {
        'out_folder': 'ERA5_wind_speed_10m_cut',
        'log1p': False,
        'stat_name': 'ERA5_wind_speed_10m_stat.pkl'
    }
}

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────
def get_date_strings(year):
    """Generates a list of YYYYMMDD strings for a given year (handles leap years)."""
    start = date(year, 1, 1)
    days_in_year = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(days_in_year)]

def save_stats_pkl(mean, var, min_val, max_val, out_path):
    """Packages and saves the statistics into PyTorch tensors within a PKL file."""
    stats = {
        "mean": torch.tensor(float(mean), dtype=torch.float32),
        "var":  torch.tensor(float(var),  dtype=torch.float32),
        "min":  torch.tensor(float(min_val), dtype=torch.float32),
        "max":  torch.tensor(float(max_val), dtype=torch.float32),
    }
    with open(out_path, "wb") as f:
        pickle.dump(stats, f)

# ── 4. MAIN PROCESSING LOOP (DYNAMIC VARIABLES) ───────────────────────────────
print("="*70)
print("🚀 STARTING UNIFIED DATA PIPELINE (NPY + LOG1P + STATS + ZEROING)")
print("="*70)

for prefix, config in VARIABLE_CONFIG.items():
    print(f"\nProcessing Variable: {prefix}")
    
    out_dir = os.path.join(OUTPUT_NPY_DIR, config['out_folder'])
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize robust float64 variables for safe chunked statistical accumulation
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    data_min = float('inf')
    data_max = float('-inf')
    
    for year in YEARS:
        tif_filename = f"{prefix}_{year}_aligned.tif"
        tif_path = os.path.join(ALIGNED_DIR, tif_filename)
        
        if not os.path.exists(tif_path):
            print(f"  ❌ WARNING: Missing file {tif_filename}")
            continue
            
        date_strings = get_date_strings(year)
        
        with rasterio.open(tif_path) as src:
            for i, date_str in enumerate(tqdm(date_strings, desc=f"  Extracting {year}", leave=False)):
                # 1. Read raw data
                data = src.read(i + 1).astype(np.float32)
                
                # 2. Apply Log1p Transform (if configured for this variable)
                if config['log1p']:
                    # We ensure no strict negatives exist before logging due to interpolation smoothing
                    data = np.log1p(np.maximum(data, 0.0))
                
                # 3. Accumulate Statistics (STRICTLY ON VALID LAND PIXELS, IGNORING NANS)
                valid_mask = ~np.isnan(data)
                if np.any(valid_mask):
                    valid_data = data[valid_mask].astype(np.float64) # float64 prevents overflow
                    total_sum += np.sum(valid_data)
                    total_sq_sum += np.sum(valid_data ** 2)
                    total_count += valid_data.size
                    data_min = min(data_min, np.min(valid_data))
                    data_max = max(data_max, np.max(valid_data))
                
                # 4. Zero-Fill the Ocean NaNs for Safe Neural Network Input
                clean_data = np.nan_to_num(data, nan=0.0)
                
                # 5. Save the final processed daily .npy
                out_npy_path = os.path.join(out_dir, f"{date_str}.npy")
                np.save(out_npy_path, clean_data)

    # 6. Calculate Final Global Variance and Mean
    if total_count > 0:
        mean = total_sum / total_count
        var = (total_sq_sum / total_count) - (mean ** 2)
        
        pkl_path = os.path.join(OUTPUT_PKL_DIR, config['stat_name'])
        save_stats_pkl(mean, var, data_min, data_max, pkl_path)
        print(f"  ✅ Stats Saved: Mean={mean:.4f}, Min={data_min:.4f}, Max={data_max:.4f}")
    else:
        print("  ❌ ERROR: No valid data found to calculate stats!")

# ── 5. PROCESSING ELEVATION (DEM) ─────────────────────────────────────────────
print("\nProcessing Static Elevation (SRTM DEM)")
dem_out_dir = os.path.join(OUTPUT_NPY_DIR, 'HGT_fix_cut_obs')
os.makedirs(dem_out_dir, exist_ok=True)

with rasterio.open(DEM_PATH) as src:
    dem_data = src.read(1).astype(np.float32)
    
    # Calculate stats on valid land pixels (Ignoring NaNs)
    valid_mask = ~np.isnan(dem_data)
    valid_dem = dem_data[valid_mask].astype(np.float64)
    
    dem_mean = np.mean(valid_dem)
    dem_var = np.var(valid_dem)
    dem_min = np.min(valid_dem)
    dem_max = np.max(valid_dem)
    
    # Zero-fill NaNs
    clean_dem = np.nan_to_num(dem_data, nan=0.0)
    
    # Save NPY
    dem_npy_path = os.path.join(dem_out_dir, "HGT_fix.npy")
    np.save(dem_npy_path, clean_dem)
    
    # Save PKL
    dem_pkl_path = os.path.join(OUTPUT_PKL_DIR, "HGT_obs_stat.pkl")
    save_stats_pkl(dem_mean, dem_var, dem_min, dem_max, dem_pkl_path)
    print(f"  ✅ Elevation Saved: Mean={dem_mean:.2f}m, Min={dem_min:.2f}m, Max={dem_max:.2f}m")

print("\n🎉 PIPELINE COMPLETE. ALL ASSETS ARE READY FOR TRAINING.")
print("="*70)