"""
precompute_percentile_maps.py
─────────────────────────────
Computes PER-PIXEL wet-day P90 and P10 maps from the CHIRPS log1p daily files,
in z-score space, ready to be cropped alongside the GT in MergeDataset and
consumed by MaskedExtremeWeightedCharbonnierLoss as per-grid thresholds
(faithful to Chandel et al. 2025, eq. 3, where y10/y90 are "for every grid").

WHAT IT PRODUCES (both 768x768 float32 .npy, in z-score space):
    p90_hr_z.npy  – per-pixel 90th percentile of WET-day rainfall (heavy threshold)
    p10_hr_z.npy  – per-pixel 10th percentile of WET-day rainfall (light threshold)
  plus *_mm.npy copies in mm/day purely for your own inspection / figures.

KEY DESIGN CHOICES (and why):
  • WET DAYS ONLY (rain >= WET_DAY_MM). Including dry days makes P90 collapse to
    ~0 in arid grids (zero-inflation). Wet-day percentiles are the climate-science
    standard (R95p / ETCCDI) and what makes "extreme" regionally meaningful.
  • PER-PIXEL over the TEMPORAL record. Each grid cell gets its own P90/P10.
  • DEGENERATE-PIXEL FLOOR. Grids with fewer than MIN_WET_DAYS wet samples can't
    give a stable percentile, so they fall back to the spatial median of all
    well-sampled grids. (Memory-safe: no giant all-wet-values array needed.)
  • OCEAN SENTINELS. Ocean pixels (LSM==0) get P90z=+10, P10z=-10 so the loss
    weighting can never trigger there. (They are zeroed by the LSM anyway, this
    is just belt-and-braces.)

LEAKAGE NOTE (read this):
  These maps feed the TRAINING loss. To be strictly leakage-free, compute them
  from the TRAINING split only. Set TRAIN_LIST_FILE to your train split list and
  the script will use only those dates. If you leave it None it uses ALL files in
  CHIRPS_DIR and prints a warning — acceptable for a quick test (these are
  climatological weights, not labels) but recompute on train-only for thesis-final
  numbers.

MEMORY:
  Loads the full (N_days, 768, 768) cube as float32. For 1754 days that's ~4 GB,
  for 2192 days ~5 GB. Percentiles are computed tile-by-tile so the peak stays
  near cube-size. If you OOM, lower the number of input days or raise TILE_ROWS
  granularity is already minimal; the cube itself is the cost.
"""

import os
import glob
import math
import numpy as np

# ── CONFIG — EDIT THESE ──────────────────────────────────────────────────────
CHIRPS_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128\DownScale_Paper\CHIRPS_precip_cut_obs_log1p"   # log1p daily .npy
LSM_PATH   = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128\DownScale_Paper\LSM_fix\india_lsm_HR.npy"  # 768x768
OUT_DIR    = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128\DownScale_Paper\percentile_maps"               # output folder

# Set to your training-split list file to avoid leakage; None = use all files.
# IMPORTANT: train_12_36.txt holds LINE INDICES into data.txt, not dates. The
# script resolves index -> data.txt line -> date token (mirrors RadarDataset:
# idx = int(seqs[ind]); img_f = radar_map[idx]; date = img_f[:-4]).
# So if you set TRAIN_LIST_FILE you must ALSO set RADAR_FILE to your data.txt.
TRAIN_LIST_FILE = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128\DownScale_Paper\DownScale_Correction_split\train_12_36.txt"   # e.g. "./DownScale_Paper/DownScale_Correction_split/train_12_36.txt"
   # e.g. r"...\DownScale_Correction_split\train_12_36.txt"
RADAR_FILE      = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128\DownScale_Paper\DownScale_Correction_split\data.txt"   # e.g. r"...\DownScale_Correction_split\data.txt"  (required if TRAIN_LIST_FILE set)

# CHIRPS log1p z-score stats (your values)
MU    = 0.4180
VAR   = 0.8968
SIGMA = math.sqrt(VAR)

# Percentile / wet-day settings
WET_DAY_MM   = 1.0    # IMD wet-day boundary (mm/day). 1.0 standard; 2.5 also defensible.
P_HIGH       = 90     # upper-tail percentile (Chandel y90)
P_LOW        = 10     # lower-tail percentile (Chandel y10)
MIN_WET_DAYS = 30     # grids with fewer wet days fall back to the spatial-median floor
TILE_ROWS    = 64     # rows processed per percentile pass (memory control)
# ─────────────────────────────────────────────────────────────────────────────


def load_file_list():
    files = sorted(glob.glob(os.path.join(CHIRPS_DIR, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files in {CHIRPS_DIR}")
    if TRAIN_LIST_FILE:
        if not RADAR_FILE:
            raise ValueError(
                "TRAIN_LIST_FILE is set but RADAR_FILE (your data.txt) is None. "
                "train_12_36.txt holds line indices into data.txt, so data.txt "
                "is needed to resolve indices -> dates."
            )
        # 1) read data.txt into {line_index: path} (mirrors read_file in the repo)
        data_map = {}
        i = 0
        with open(RADAR_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    data_map[i] = line
                    i += 1
        # 2) resolve each training index to its date token (filename without .npy)
        train_dates = set()
        with open(TRAIN_LIST_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                idx = int(line)
                img_f = os.path.basename(data_map[idx])      # e.g. 20200101.npy
                train_dates.add(os.path.splitext(img_f)[0])  # e.g. 20200101
        # 3) keep only CHIRPS files whose name contains a training date token
        kept = [p for p in files if os.path.splitext(os.path.basename(p))[0] in train_dates]
        if not kept:
            # fall back to substring match in case CHIRPS filenames carry a prefix/suffix
            kept = [p for p in files if any(d in os.path.basename(p) for d in train_dates)]
        if not kept:
            raise RuntimeError(
                f"Resolved {len(train_dates)} training dates from the split, but matched "
                f"0 CHIRPS files. Example training date: {next(iter(train_dates))}. "
                f"Example CHIRPS filename: {os.path.basename(files[0])}. "
                f"Check that they share the same date token."
            )
        print(f"Using TRAIN split only: {len(kept)} / {len(files)} files "
              f"({len(train_dates)} train dates resolved from data.txt).")
        return kept
    print(f"⚠️  TRAIN_LIST_FILE is None — using ALL {len(files)} files "
          f"(mild leakage; fine for a test, recompute on train-only for final).")
    return files


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = load_file_list()
    N = len(files)

    lsm = np.load(LSM_PATH).astype(np.float32)
    H, W = lsm.shape
    land = lsm > 0
    print(f"Grid {H}x{W}, land pixels: {int(land.sum()):,}, days: {N}")

    wet_log = math.log1p(WET_DAY_MM)

    # ── load cube (N, H, W) float32 ──────────────────────────────────────────
    print("Loading daily arrays into memory (this is the memory-heavy step)...")
    cube = np.empty((N, H, W), dtype=np.float32)
    for i, f in enumerate(files):
        arr = np.load(f).astype(np.float32)
        if arr.shape != (H, W):
            raise ValueError(f"{f} has shape {arr.shape}, expected {(H, W)}")
        cube[i] = arr
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{N}")

    # ── tiled per-pixel wet-day percentiles ──────────────────────────────────
    print("Computing per-pixel wet-day percentiles (tiled)...")
    p90 = np.zeros((H, W), np.float32)
    p10 = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.int32)
    for r0 in range(0, H, TILE_ROWS):
        r1 = min(r0 + TILE_ROWS, H)
        blk = cube[:, r0:r1, :]                 # (N, tr, W)
        wm = blk >= wet_log                     # wet-day mask
        cnt[r0:r1, :] = wm.sum(axis=0)
        b = np.where(wm, blk, np.nan)
        with np.errstate(all="ignore"):
            p90[r0:r1, :] = np.nan_to_num(np.nanpercentile(b, P_HIGH, axis=0), nan=0.0)
            p10[r0:r1, :] = np.nan_to_num(np.nanpercentile(b, P_LOW,  axis=0), nan=0.0)

    # ── degenerate-pixel floor (spatial median of well-sampled land grids) ───
    enough = (cnt >= MIN_WET_DAYS) & land
    if enough.sum() == 0:
        raise RuntimeError("No grid has >= MIN_WET_DAYS wet days. Lower MIN_WET_DAYS.")
    floor90 = float(np.median(p90[enough]))
    floor10 = float(np.median(p10[enough]))
    fill = land & (~enough)
    p90[fill] = floor90
    p10[fill] = floor10
    print(f"Well-sampled land grids: {int(enough.sum()):,} / {int(land.sum()):,}")
    print(f"Floor (log1p)  P90={floor90:.4f} ({math.expm1(floor90):.1f} mm)  "
          f"P10={floor10:.4f} ({math.expm1(floor10):.1f} mm)")

    # ── to z-score space + ocean sentinels ───────────────────────────────────
    p90z = ((p90 - MU) / SIGMA).astype(np.float32)
    p10z = ((p10 - MU) / SIGMA).astype(np.float32)
    p90z[~land] = 10.0    # target can never exceed -> no extreme weight on ocean
    p10z[~land] = -10.0   # target can never fall below -> no low weight on ocean

    # ── save ─────────────────────────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, "p90_hr_z.npy"), p90z)
    np.save(os.path.join(OUT_DIR, "p10_hr_z.npy"), p10z)
    np.save(os.path.join(OUT_DIR, "p90_hr_mm.npy"), np.expm1(p90).astype(np.float32))
    np.save(os.path.join(OUT_DIR, "p10_hr_mm.npy"), np.expm1(p10).astype(np.float32))

    # ── report ───────────────────────────────────────────────────────────────
    lp90z = p90z[land]; lp90mm = np.expm1(p90[land])
    print("\n================ SUMMARY (land pixels) ================")
    print(f"P90 z-score : min {lp90z.min():.3f}  median {np.median(lp90z):.3f}  max {lp90z.max():.3f}")
    print(f"P90 mm/day  : min {lp90mm.min():.1f}  median {np.median(lp90mm):.1f}  max {lp90mm.max():.1f}")
    print(f"  (spatially varying -> arid grids low, Western Ghats / NE high)")
    print(f"Saved to {OUT_DIR}:  p90_hr_z.npy  p10_hr_z.npy  (+ *_mm.npy for inspection)")
    print("=======================================================")


if __name__ == "__main__":
    main()