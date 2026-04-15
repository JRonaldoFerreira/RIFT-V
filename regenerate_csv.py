"""Regenerate results.csv with campaign-specific columns."""
import sys
from pathlib import Path

import pandas as pd

# Set up project path
sys.path.insert(0, str(Path(__file__).parent))

# Import from tflmfi.uart directly
from tflmfi.uart import build_csv

# Path to the run
run_dir = Path("runs/multi_memoryfi_lenet5_mnist_20260414_091416")
jsonl_path = run_dir / "results.jsonl"
csv_path = run_dir / "results.csv"
groups_csv = run_dir / "memory_fault_groups.csv"

print(f"Regenerating {csv_path}...")
print(f"  Source: {jsonl_path}")
print(f"  Groups: {groups_csv}")

# Build with campaign-specific columns
build_csv(
    jsonl_path,
    csv_path,
    campaign_type="memory",
    is_multi=True,
    groups_csv=groups_csv,
)

# Verify
print("\nVerifying new CSV columns:")
df = pd.read_csv(csv_path)
print(df.columns.tolist())
print(f"\nTotal rows: {len(df)}")

# Show sample
print("\nSample fault rows (with group enrichment):")
faults = df[df["phase"] == "fault"].head(3)
if "group_id" in df.columns:
    cols = ["group_id", "fault_count", "image_id", "label", "pred_class"]
    if "pool_index_0" in df.columns:
        cols.extend(["pool_index_0", "pool_index_1", "pool_index_2"])
    print(faults[cols].to_string())

print("\n✓ Done!")
