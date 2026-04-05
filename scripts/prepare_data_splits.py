"""Generate data split files required by MergeDataset.

Creates the following files under DownScale_Paper/DownScale_Correction_split/:

  data.txt           – one filename (YYYYMMDD.npy) per line, all 92 days
  train_12_36.txt    – 0-based indices of the training days  (≈80 %)
  val_12_36.txt      – 0-based indices of the validation days (≈10 %)
  test_12_36.txt     – 0-based indices of the test days       (≈10 %)

Dataset: June–August 2020 (92 days total).
  Jun 2020: 30 days → indices 0-29
  Jul 2020: 31 days → indices 30-60
  Aug 2020: 31 days → indices 61-91

Default split (chronological, no shuffling):
  Train : indices 0-73  (74 days)
  Val   : indices 74-82  (9 days)
  Test  : indices 83-91  (9 days)

Usage (run from the repository root):
    python scripts/prepare_data_splits.py
"""

import os
import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

SPLIT_DIR = "./DownScale_Paper/DownScale_Correction_split"

# Period covered by the dataset
START_DATE = datetime.date(2020, 6, 1)
END_DATE   = datetime.date(2020, 8, 31)

# Split boundary indices (0-based, exclusive upper bound)
TRAIN_END = 74   # training   = indices [0, 74)  → 74 days
VAL_END   = 83   # validation = indices [74, 83) →  9 days
# test                        = indices [83, 92) →  9 days

# ── Helpers ───────────────────────────────────────────────────────────────────


def date_range(start: datetime.date, end: datetime.date):
    """Yield each date from *start* to *end* inclusive."""
    delta = end - start
    for i in range(delta.days + 1):
        yield start + datetime.timedelta(days=i)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SPLIT_DIR, exist_ok=True)

    dates = list(date_range(START_DATE, END_DATE))
    n = len(dates)
    print(f"Total days: {n}  ({START_DATE} – {END_DATE})")

    filenames = [d.strftime("%Y%m%d") + ".npy" for d in dates]
    indices   = list(range(n))

    train_idx = indices[:TRAIN_END]
    val_idx   = indices[TRAIN_END:VAL_END]
    test_idx  = indices[VAL_END:]

    print(f"Train: {len(train_idx)} days  (indices {train_idx[0]}–{train_idx[-1]})")
    print(f"Val  : {len(val_idx)}  days  (indices {val_idx[0]}–{val_idx[-1]})")
    print(f"Test : {len(test_idx)}  days  (indices {test_idx[0]}–{test_idx[-1]})")

    data_path = os.path.join(SPLIT_DIR, "data.txt")
    with open(data_path, "w") as f:
        f.write("\n".join(filenames) + "\n")
    print(f"\nWrote {data_path}")

    for split_name, split_indices in [
        ("train_12_36", train_idx),
        ("val_12_36",   val_idx),
        ("test_12_36",  test_idx),
    ]:
        path = os.path.join(SPLIT_DIR, split_name + ".txt")
        with open(path, "w") as f:
            f.write("\n".join(str(i) for i in split_indices) + "\n")
        print(f"Wrote {path}")

    print("\nDone.")
