# experiments/test_dataset.py

print("[test_dataset] Script started")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("[test_dataset] Project root:", ROOT)

from offline_rl.kuairec_dataset import KuaiRecOfflineDataset

print("[test_dataset] Creating dataset...")

ds = KuaiRecOfflineDataset(
    csv_path="data/raw/kuairec/data/big_matrix.csv",
    history_len=5,
)

print("[test_dataset] Dataset size:", len(ds))

sample = ds[0]
print("[test_dataset] Sample contents:")
for k, v in sample.items():
    print(f"  {k:10s} shape={tuple(v.shape)} dtype={v.dtype}")