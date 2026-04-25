import glob
import os
import pickle
import numpy as np
import torch

# ── Configuration ─────────────────────────────────────────────────────────────

NPY_DIR  = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"
STAT_DIR = r"C:\Users\HP\Downloads\MTP_repo\TransfomerDownscaling_MP\DownScale_Paper\param_stat_12_36"

# Data is already log1p-transformed on disk → no need to apply again here
LOG1P_TRANSFORM = False

VAR_DIRS = {
    "ERA5_precip_cut":       os.path.join(NPY_DIR, "ERA5_precip_cut_log1p"),
    "CHIRPS_precip_cut_obs": os.path.join(NPY_DIR, "CHIRPS_precip_cut_obs_log1p"),
    "HGT_fix_cut_obs":       os.path.join(NPY_DIR, "HGT_fix_cut_obs"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
# ── Stat filename map ──────────────────────────────────────────────────────────
STAT_FILENAMES = {
    "ERA5_precip_cut":       "ERA5_precip_log1p_stat.pkl",
    "CHIRPS_precip_cut_obs": "CHIRPS_precip_obs_log1p_stat.pkl",
    "HGT_fix_cut_obs":       "HGT_obs_stat.pkl",
}


def compute_stats(npy_dir: str, log1p: bool = False) -> dict:
    """Load all .npy files in *npy_dir* and return a stats dict with
    mean, var, min, max as float32 torch tensors."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")
    print(f"  Found {len(files)} file(s) in {npy_dir}")

    arrays = [np.load(f).astype(np.float32) for f in files]
    data   = np.concatenate([a.ravel() for a in arrays])

    if log1p:
        data = np.log1p(data)

    stats = {
        "mean": torch.tensor(float(data.mean()), dtype=torch.float32),
        "var":  torch.tensor(float(data.var()),  dtype=torch.float32),
        "min":  torch.tensor(float(data.min()),  dtype=torch.float32),
        "max":  torch.tensor(float(data.max()),  dtype=torch.float32),
    }
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(STAT_DIR, exist_ok=True)

    for var_name, npy_dir in VAR_DIRS.items():
        print(f"\nProcessing '{var_name}':")
        stats    = compute_stats(npy_dir, log1p=LOG1P_TRANSFORM)
        fname    = STAT_FILENAMES[var_name]          # ← use dict instead of function
        out_path = os.path.join(STAT_DIR, fname)

        with open(out_path, "wb") as f:
            pickle.dump(stats, f)

        print(f"  mean = {stats['mean'].item():.4f}")
        print(f"  std  = {stats['var'].item()**0.5:.4f}")
        print(f"  min  = {stats['min'].item():.4f}")
        print(f"  max  = {stats['max'].item():.4f}")
        print(f"  Saved → {out_path}")

    print("\n✓ All statistics saved successfully.")
    print(f"\n  Output files:")
    for var_name in VAR_DIRS:
        print(f"  {STAT_DIR}\\{STAT_FILENAMES[var_name]}")