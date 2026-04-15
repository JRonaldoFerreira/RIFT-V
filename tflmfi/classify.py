"""Classify fault injection results against the golden baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

NON_INJECTION_STATUSES = {
    "timer_not_fired",
    "timing_window_too_short",
    "invalid_frame",
    "invalid_register",
    "idle",
    "armed",
    "panic_non_json",
}


# ---------------------------------------------------------------------------
# Row classifier
# ---------------------------------------------------------------------------

def classify_row(row: pd.Series) -> str:
    """Classify a single fault-injection result row."""
    status = str(row.get("status", "ok"))
    if status != "ok":
        return "Crash"

    fault_model = str(row.get("fault_model", ""))
    if fault_model == "register":
        injection_status = str(row.get("injection_status", "injected"))
        if injection_status in NON_INJECTION_STATUSES:
            return "NoInjection"

    pred = int(row["pred_class"])
    golden_pred = int(row["golden_pred_class_ref"])
    label = int(row["label"])
    prob = float(row["top1_prob"])
    golden_prob = float(row["golden_top1_prob_ref"])

    if pred == golden_pred and np.isclose(prob, golden_prob, atol=1e-6):
        return "Masked"
    if pred == label and golden_pred != label:
        return "Good"
    if pred == golden_pred:
        if golden_prob <= 0:
            return "Acceptable"
        drop = (golden_prob - prob) / golden_prob
        return "Acceptable" if drop < 0.05 else "Warning"
    if pred != label or prob < 0.80:
        return "Critical"
    return "Observed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_results(results_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Classify fault results in a campaign CSV.

    The CSV must contain both golden (phase=="golden") and fault rows.
    Returns the classified DataFrame (fault rows only).
    """
    df = pd.read_csv(results_csv)

    golden_df = df[df["phase"] == "golden"].copy()
    fault_df = df[(df["event"] == "result") & (df["phase"] == "fault")].copy()

    if golden_df.empty:
        raise ValueError("No golden rows found in results CSV.")
    if fault_df.empty:
        raise ValueError("No fault rows found in results CSV.")

    # Build golden reference lookup
    golden_ref = golden_df[["image_id", "pred_class", "top1_prob"]].rename(columns={
        "pred_class": "golden_pred_class_ref",
        "top1_prob": "golden_top1_prob_ref",
    })

    merged = fault_df.merge(golden_ref, on="image_id", how="left")
    merged["fault_class"] = merged.apply(classify_row, axis=1)
    merged.to_csv(output_csv, index=False)

    # Print summary
    vc = merged["fault_class"].value_counts()
    total = len(merged)
    print(f"\n{'='*50}")
    print(f"Classification Summary  ({total} fault results)")
    print(f"{'='*50}")
    for cls in ["Masked", "Acceptable", "Observed", "Warning", "Critical", "Crash", "Good", "NoInjection"]:
        n = vc.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"  {cls:<15s}: {n:>6d}  ({pct:5.1f}%)")
    print(f"{'='*50}\n")

    return merged
