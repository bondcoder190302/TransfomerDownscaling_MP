"""Compute normalization statistics for ERA5 precipitation, CHIRPS precipitation,
and the DEM elevation variable, then save them as .pkl files in the format expected
by MergeDataset.init_stat().

Stat file naming convention (matches the dataset code):
  varName.replace("_cut","").replace("_fix","") + "_stat.pkl"

  ERA5_precip_cut      → ERA5_precip_stat.pkl
  CHIRPS_precip_cut_obs → CHIRPS_precip_obs_stat.pkl
  HGT_fix_cut_obs      → HGT_obs_stat.pkl

Usage (run from the repository root):
    python scripts/compute_precip_stats.py

Note on precipitation skewness:
  Precipitation distributions are heavily right-skewed (many zeros, rare large
  values). A log1p transform (log(1+x)) before fitting z-score statistics can
  improve normalisation. Set LOG1P_TRANSFORM = True below to enable it; the same
  transform must then be applied to your data before loading (i.e. pre-process the
  .npy files with log1p before saving them to disk).
"""

import glob
import os
import pickle
import numpy as np
import torch

# ── Configuration ─────────────────────────────────────────────────────────────

DATAROOT = "./DownScale_Paper"
STAT_DIR = os.path.join(DATAROOT, "param_stat_12_36")

# Set to True if your .npy files have already been log1p-transformed.
LOG1P_TRANSFORM = False

# Map from variable name (as used in the YAML config) to the directory that
# holds the .npy files for that variable.
VAR_DIRS = {
    "ERA5_precip_cut":       os.path.join(DATAROOT, "ERA5_precip_cut"),
    "CHIRPS_precip_cut_obs": os.path.join(DATAROOT, "CHIRPS_precip_cut_obs"),
    # For the static elevation layer there is only one file; point directly to it.
    "HGT_fix_cut_obs":       os.path.join(DATAROOT, "HGT_fix_cut_obs"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def stat_filename(var_name: str) -> str:
    """Return the .pkl filename that MergeDataset expects for *var_name*."""
    base = var_name.replace("_cut", "").replace("_fix", "")
    return base + "_stat.pkl"


def compute_stats(npy_dir: str, log1p: bool = False) -> dict:
    """Load all .npy files in *npy_dir* and return a stats dict with
    mean, var, min, max as float32 torch tensors."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")
    print(f"  Found {len(files)} file(s) in {npy_dir}")

    arrays = [np.load(f).astype(np.float32) for f in files]
    data = np.concatenate([a.ravel() for a in arrays])

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
        stats = compute_stats(npy_dir, log1p=LOG1P_TRANSFORM)
        fname = stat_filename(var_name)
        out_path = os.path.join(STAT_DIR, fname)
        with open(out_path, "wb") as f:
            pickle.dump(stats, f)
        print(f"  mean={stats['mean'].item():.4f}  "
              f"std={stats['var'].item()**0.5:.4f}  "
              f"min={stats['min'].item():.4f}  "
              f"max={stats['max'].item():.4f}")
        print(f"  Saved → {out_path}")

    print("\nAll statistics saved successfully.")
