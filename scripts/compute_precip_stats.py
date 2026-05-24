import glob
import os
import pickle
import numpy as np
import torch

# ── Configuration ─────────────────────────────────────────────────────────────

# Your requested base and output directories
NPY_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"
STAT_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\pkl_files"

# Data is already log1p-transformed on disk for precip → no need to apply again here
LOG1P_TRANSFORM = False

# Dictionary mapping variable keys to their folder paths
VAR_DIRS = {
    "ERA5_precip_cut":       os.path.join(NPY_DIR, "ERA5_precip_cut_log1p"),
    "MSWEP_precip_cut":      os.path.join(NPY_DIR, "MSWEP_precip_cut_log1p"),
    "CHIRPS_precip_cut_obs": os.path.join(NPY_DIR, "CHIRPS_precip_cut_obs_log1p"),
    "HGT_fix_cut_obs":       os.path.join(NPY_DIR, "HGT_fix_cut_obs"),
    "ERA5_u_wind":           os.path.join(NPY_DIR, "ERA5_u_wind_10m_cut"),
    "ERA5_v_wind":           os.path.join(NPY_DIR, "ERA5_v_wind_10m_cut"),
    "ERA5_wind_speed":       os.path.join(NPY_DIR, "ERA5_wind_speed_10m_cut"),
}

# ── Stat filename map ──────────────────────────────────────────────────────────
# Dictionary mapping keys to the specific .pkl filenames used in your YAML
STAT_FILENAMES = {
    "ERA5_precip_cut":       "ERA5_precip_log1p_stat.pkl",
    "MSWEP_precip_cut":      "MSWEP_precip_log1p_stat.pkl",
    "CHIRPS_precip_cut_obs": "CHIRPS_precip_obs_log1p_stat.pkl",
    "HGT_fix_cut_obs":       "HGT_obs_stat.pkl",
    "ERA5_u_wind":           "ERA5_u_wind_10m_stat.pkl",
    "ERA5_v_wind":           "ERA5_v_wind_10m_stat.pkl",
    "ERA5_wind_speed":       "ERA5_wind_speed_10m_stat.pkl",
}

def compute_stats(npy_dir: str, log1p: bool = False) -> dict:
    """Load all .npy files in *npy_dir* and return a stats dict with
    mean, var, min, max as float32 torch tensors."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")
    print(f"  Found {len(files)} file(s) in {npy_dir}")

    # Process files one by one to avoid huge memory spikes with 6 years of data
    total_count = 0
    total_sum   = 0.0
    total_sq_sum = 0.0
    data_min    = float('inf')
    data_max    = float('-inf')

    for f in files:
        array = np.load(f).astype(np.float64) # Use float64 for sum stability
        if log1p:
            array = np.log1p(array)
        
        total_sum += array.sum()
        total_sq_sum += (array**2).sum()
        total_count += array.size
        data_min = min(data_min, array.min())
        data_max = max(data_max, array.max())

    mean = total_sum / total_count
    var  = (total_sq_sum / total_count) - (mean**2)

    stats = {
        "mean": torch.tensor(float(mean), dtype=torch.float32),
        "var":  torch.tensor(float(var),  dtype=torch.float32),
        "min":  torch.tensor(float(data_min),  dtype=torch.float32),
        "max":  torch.tensor(float(data_max),  dtype=torch.float32),
    }
    return stats

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(STAT_DIR, exist_ok=True)

    for var_name, npy_dir in VAR_DIRS.items():
        print(f"\nProcessing '{var_name}':")
        try:
            stats    = compute_stats(npy_dir, log1p=LOG1P_TRANSFORM)
            fname    = STAT_FILENAMES[var_name]
            out_path = os.path.join(STAT_DIR, fname)

            with open(out_path, "wb") as f:
                pickle.dump(stats, f)

            print(f"  mean = {stats['mean'].item():.4f}")
            print(f"  std  = {stats['var'].item()**0.5:.4f}")
            print(f"  min  = {stats['min'].item():.4f}")
            print(f"  max  = {stats['max'].item():.4f}")
            print(f"  Saved → {out_path}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")

    print("\n✓ All statistics saved successfully.")