#!/usr/bin/env python3
import os
import glob
import numpy as np

ROOT = "./DownScale_Paper"

TARGETS = {
    "ERA5_precip_cut_log1p": os.path.join(ROOT, "ERA5_precip_cut_log1p"),
    "CHIRPS_precip_cut_obs_log1p": os.path.join(ROOT, "CHIRPS_precip_cut_obs_log1p"),
    "HGT_fix_cut_obs": os.path.join(ROOT, "HGT_fix_cut_obs"),
}

# Change if you want stricter/looser flags
ABS_EXTREME_THRESHOLD = 1e6


def scan_file(fp):
    arr = np.load(fp)
    finite = np.isfinite(arr)
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    finite_ratio = finite.mean()

    if finite.any():
        vals = arr[finite]
        mn = float(vals.min())
        mx = float(vals.max())
        mean = float(vals.mean())
        std = float(vals.std())
        p001 = float(np.percentile(vals, 0.1))
        p01 = float(np.percentile(vals, 1))
        p99 = float(np.percentile(vals, 99))
        p999 = float(np.percentile(vals, 99.9))
    else:
        mn = mx = mean = std = p001 = p01 = p99 = p999 = np.nan

    extreme = False
    if finite.any():
        extreme = (abs(mn) > ABS_EXTREME_THRESHOLD) or (abs(mx) > ABS_EXTREME_THRESHOLD)

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "finite_ratio": float(finite_ratio),
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "p0.1": p001,
        "p1": p01,
        "p99": p99,
        "p99.9": p999,
        "extreme": bool(extreme),
    }


def scan_dir(name, d):
    print(f"\n=== Scanning {name} ===")
    if not os.path.isdir(d):
        print(f"[ERROR] Missing directory: {d}")
        return

    files = sorted(glob.glob(os.path.join(d, "*.npy")))
    if not files:
        print(f"[WARN] No .npy files found in {d}")
        return

    bad = []
    global_min = np.inf
    global_max = -np.inf

    for i, fp in enumerate(files, 1):
        s = scan_file(fp)

        if np.isfinite(s["min"]):
            global_min = min(global_min, s["min"])
            global_max = max(global_max, s["max"])

        if s["has_nan"] or s["has_inf"] or s["extreme"]:
            bad.append((fp, s))

        # light progress
        if i % 50 == 0 or i == 1 or i == len(files):
            print(f"  [{i:4d}/{len(files)}] {os.path.basename(fp)}")

    print(f"Total files: {len(files)}")
    print(f"Global min/max (finite): {global_min:.6g} / {global_max:.6g}")

    if not bad:
        print("[OK] No NaN/Inf/extreme flagged files.")
        return

    print(f"[ALERT] Flagged files: {len(bad)}")
    for fp, s in bad[:30]:
        print(
            f"- {fp}\n"
            f"    shape={s['shape']} dtype={s['dtype']} "
            f"nan={s['has_nan']} inf={s['has_inf']} extreme={s['extreme']} "
            f"min={s['min']:.6g} max={s['max']:.6g} std={s['std']:.6g}"
        )
    if len(bad) > 30:
        print(f"... and {len(bad)-30} more")


if __name__ == "__main__":
    for name, d in TARGETS.items():
        scan_dir(name, d)