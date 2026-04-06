"""
fix_labels.py
-------------
Fast one-time fix: re-saves y_train / y_dev / y_test .npy files from the CSV split files,
WITHOUT re-running the expensive feature extraction in build_dataset.py.

Run from project root:
    python src/fix_labels.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- Paths ---
project_root = Path(__file__).parent.parent
raw_dir  = project_root / "data" / "raw" / "DAIC_WOZ"
save_dir = project_root / "data" / "processed_features"

split_csvs = {
    "train": raw_dir / "train_split_Depression_AVEC2017.csv",
    "dev":   raw_dir / "dev_split_Depression_AVEC2017.csv",
    "test":  raw_dir / "full_test_split.csv",
}

def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['Participant_ID'].astype(str), df['PHQ8_Binary']))

# Build a full label lookup across all splits
all_labels = {}
for split, csv_path in split_csvs.items():
    if csv_path.exists():
        all_labels.update(load_labels(csv_path))
        print(f"Loaded {len(load_labels(csv_path))} labels from {csv_path.name}")
    else:
        print(f"[WARN] Missing: {csv_path}")

# For each split, load already-saved IDs and re-save matching labels
for split in ["train", "dev", "test"]:
    ids_path  = save_dir / f"ids_{split}.npy"
    x_path    = save_dir / f"X_{split}_scaled.npy"
    y_path    = save_dir / f"y_{split}.npy"
    csv_path  = split_csvs.get(split)

    # Determine participant IDs
    if ids_path.exists():
        ids = np.load(ids_path).astype(int)
        print(f"[{split}] Loaded {len(ids)} IDs from ids_{split}.npy")
    elif csv_path and csv_path.exists() and x_path.exists():
        # Fall back: derive from CSV, verify count matches X array rows
        df_csv = pd.read_csv(csv_path)
        csv_ids = df_csv['Participant_ID'].astype(int).tolist()
        x_rows = np.load(x_path).shape[0]
        if len(csv_ids) != x_rows:
            print(f"[{split}] Row mismatch: CSV has {len(csv_ids)} IDs but X has {x_rows} rows.")
            print(f"         This can happen when some participants were skipped during extraction.")
            print(f"         Cannot reliably re-save labels. Please re-run build_dataset.py for this split.")
            continue
        ids = np.array(csv_ids)
        print(f"[{split}] Derived {len(ids)} IDs from {csv_path.name} (matched X shape)")
    else:
        print(f"[SKIP] {split}: No IDs available — run build_dataset.py first.")
        continue

    labels = []
    missing = []
    for pid in ids:
        lbl = all_labels.get(str(pid))
        if lbl is not None:
            labels.append(float(lbl))
        else:
            labels.append(float('nan'))
            missing.append(pid)

    valid = [l for l in labels if not np.isnan(l)]
    if missing:
        print(f"  [{split}] {len(missing)} IDs had no label in CSVs: {missing}")

    if valid:
        np.save(y_path, np.array(labels, dtype=np.float32))
        print(f"  [{split}] Saved y_{split}.npy  ({len(valid)} valid labels)")
    else:
        print(f"  [{split}] No valid labels found — y_{split}.npy NOT saved.")

print("\nDone! Re-run train_traditional.py now.")
