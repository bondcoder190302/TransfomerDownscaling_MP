"""Generate data split files required by MergeDataset.
Creates the following files under DownScale_Paper/DownScale_Correction_split/:
  data.txt           – one filename (YYYYMMDD.npy) per line, all 366 days
  train_12_36.txt    – 0-based indices of the training days  (≈80% → 293 days)
  val_12_36.txt      – 0-based indices of the validation days (≈10% →  37 days)
  test_12_36.txt     – 0-based indices of the test days       (≈10% →  36 days)
Dataset: Full Year 2020 (366 days, leap year).
Split: RANDOM with fixed seed for reproducibility.
Usage (run from the repository root):
    python scripts/prepare_data_splits.py
"""

import os
import datetime
import random

# ── Configuration ─────────────────────────────────────────────────────────────
SPLIT_DIR  = r"C:\Users\HP\Downloads\MTP_repo\TransfomerDownscaling_MP\DownScale_Paper\DownScale_Correction_split"

START_DATE = datetime.date(2020, 1, 1)
END_DATE   = datetime.date(2020, 12, 31)

RANDOM_SEED = 42   # fix seed → same split every run

# 80 / 10 / 10 split over 366 days
TRAIN_SIZE = 293
VAL_SIZE   = 37
# TEST_SIZE  = 366 - 293 - 37 = 36

# ── Helpers ───────────────────────────────────────────────────────────────────
def date_range(start: datetime.date, end: datetime.date):
    """Yield each date from *start* to *end* inclusive."""
    delta = end - start
    for i in range(delta.days + 1):
        yield start + datetime.timedelta(days=i)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(SPLIT_DIR, exist_ok=True)

    dates     = list(date_range(START_DATE, END_DATE))
    n         = len(dates)
    filenames = [d.strftime("%Y%m%d") + ".npy" for d in dates]
    indices   = list(range(n))

    print(f"Total days : {n}  ({START_DATE} – {END_DATE})")
    assert n == 366, f"Expected 366 days for 2020, got {n}"

    # ── Random shuffle ────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    shuffled = indices.copy()
    random.shuffle(shuffled)

    train_idx = sorted(shuffled[:TRAIN_SIZE])
    val_idx   = sorted(shuffled[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE])
    test_idx  = sorted(shuffled[TRAIN_SIZE + VAL_SIZE :])

    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "Split sizes don't add up!"
    assert len(set(train_idx) & set(val_idx))  == 0, "Train/Val overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/Test overlap!"
    assert len(set(val_idx)   & set(test_idx)) == 0, "Val/Test overlap!"

    print(f"Train : {len(train_idx)} days  ({len(train_idx)/n*100:.1f}%)")
    print(f"Val   : {len(val_idx)}  days  ({len(val_idx)/n*100:.1f}%)")
    print(f"Test  : {len(test_idx)}  days  ({len(test_idx)/n*100:.1f}%)")

    # ── Write data.txt ────────────────────────────────────────────────────────
    data_path = os.path.join(SPLIT_DIR, "data.txt")
    with open(data_path, "w") as f:
        f.write("\n".join(filenames) + "\n")
    print(f"\nWrote {data_path}  ({n} filenames)")

    # ── Write split index files ───────────────────────────────────────────────
    for split_name, split_indices in [
        ("train_12_36", train_idx),
        ("val_12_36",   val_idx),
        ("test_12_36",  test_idx),
    ]:
        path = os.path.join(SPLIT_DIR, split_name + ".txt")
        with open(path, "w") as f:
            f.write("\n".join(str(i) for i in split_indices) + "\n")
        print(f"Wrote {path}  ({len(split_indices)} indices)")

    # ── Sanity print: show sample dates from each split ───────────────────────
    print("\n  Sample train dates :", [filenames[i] for i in train_idx[:3]], "...")
    print("  Sample val   dates :", [filenames[i] for i in val_idx[:3]],   "...")
    print("  Sample test  dates :", [filenames[i] for i in test_idx[:3]],  "...")

    print("\nDone.")