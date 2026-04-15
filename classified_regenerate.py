"""Regenerate classified_results.csv from results.csv."""
import sys
from pathlib import Path
 
import pandas as pd
 
# Set up project path
sys.path.insert(0, str(Path(__file__).parent))
 
# Import from tflmfi.classify directly
from tflmfi.classify import classify_results
 
# Path to the run (change this to target different runs)
run_dir = Path("runs/multi_memoryfi_lenet5_mnist_20260414_064054")
results_csv = run_dir / "results.csv"
classified_csv = run_dir / "classified_results.csv"
 
print(f"Regenerating {classified_csv}...")
print(f"  Source: {results_csv}")
 
# Check if source exists
if not results_csv.exists():
    print(f"[ERROR] {results_csv} not found. Run regenerate_csv.py first.")
    sys.exit(1)
 
# Classify
classify_results(results_csv, classified_csv)
 
print(f"\n✓ Done! Classified results saved to {classified_csv}")
 