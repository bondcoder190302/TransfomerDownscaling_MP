import numpy as np
import glob
import os

def apply_log1p_transform(input_dir, output_dir):
    """Apply log1p transform: log(1+x) to all .npy files"""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"\nProcessing {len(files)} files from {input_dir}")
    print(f"Output to: {output_dir}\n")

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        data = np.load(filepath)
        data_log1p = np.log1p(data)  # log(1+x)

        output_path = os.path.join(output_dir, filename)
        np.save(output_path, data_log1p)

        if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(files):
            print(f"  [{i+1:3d}/{len(files)}] {filename}  "
                  f"orig_max={data.max():.4f}  →  log1p_max={data_log1p.max():.4f}")

    print(f"✓ Completed: {len(files)} files saved to {output_dir}")


# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\HP\Downloads\MTP Phase 2\NPY_new_128"

ERA5_IN    = os.path.join(BASE_DIR, "ERA5_precip_cut")
CHIRPS_IN  = os.path.join(BASE_DIR, "CHIRPS_precip_cut_obs")

ERA5_OUT   = os.path.join(BASE_DIR, "ERA5_precip_cut_log1p")
CHIRPS_OUT = os.path.join(BASE_DIR, "CHIRPS_precip_cut_obs_log1p")

# ── RUN ────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("APPLYING LOG1P TRANSFORM TO ALL DATASETS")
print("=" * 70)

apply_log1p_transform(ERA5_IN,   ERA5_OUT)
apply_log1p_transform(CHIRPS_IN, CHIRPS_OUT)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)