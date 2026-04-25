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
        # The host scheduler emits ``fatal_hang`` (in either ``status`` or
        # ``injection_status``) when all hardware-reset strategies plus the
        # firmware-watchdog wait fail to recover the device.  Surface that
        # as a distinct ``Hang`` class so it can be reported separately
        # from ordinary crashes (which do reboot the chip).
        injection_status = str(row.get("injection_status", ""))
        if status == "fatal_hang" or injection_status == "fatal_hang":
            return "Hang"
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

    # Enrich operator_name from op_profile rows when firmware didn't include it
    if "operator_name" in merged.columns:
        missing_mask = merged["operator_name"].isna()
        if missing_mask.any() and "operator_index" in merged.columns:
            op_profiles = df[df["event"] == "op_profile"]
            if not op_profiles.empty and "operator_index" in op_profiles.columns and "operator_name" in op_profiles.columns:
                op_map = op_profiles.dropna(subset=["operator_index", "operator_name"])
                op_map = op_map.drop_duplicates(subset=["operator_index"], keep="last")
                op_map = op_map.set_index("operator_index")["operator_name"]
                can_fill = missing_mask & merged["operator_index"].notna()
                if can_fill.any():
                    merged.loc[can_fill, "operator_name"] = (
                        merged.loc[can_fill, "operator_index"].map(op_map)
                    )

        # Fallback: fill remaining NaN operator_name/operator_index from selected faults CSV
        still_missing = merged["operator_name"].isna()
        if still_missing.any() and "fault_id" in merged.columns:
            run_dir = Path(results_csv).parent
            for sel_csv in sorted(run_dir.glob("selected_*faults.csv")):
                sel_df = pd.read_csv(sel_csv)
                name_col = "op_name" if "op_name" in sel_df.columns else "operator_name" if "operator_name" in sel_df.columns else None
                idx_col = "op_index" if "op_index" in sel_df.columns else "operator_index" if "operator_index" in sel_df.columns else None
                if name_col and "fault_id" in sel_df.columns:
                    sel_map = sel_df.dropna(subset=["fault_id"]).drop_duplicates(
                        subset=["fault_id"], keep="last"
                    ).set_index("fault_id")
                    can_fill2 = still_missing & merged["fault_id"].notna()
                    if can_fill2.any():
                        merged.loc[can_fill2, "operator_name"] = (
                            merged.loc[can_fill2, "fault_id"].map(sel_map[name_col])
                        )
                    if idx_col and "operator_index" in merged.columns:
                        idx_missing = merged["operator_index"].isna() & merged["fault_id"].notna()
                        if idx_missing.any():
                            merged.loc[idx_missing, "operator_index"] = (
                                merged.loc[idx_missing, "fault_id"].map(sel_map[idx_col])
                            )
                    break

    merged["fault_class"] = merged.apply(classify_row, axis=1)
    merged.to_csv(output_csv, index=False)

    # Print summary
    vc = merged["fault_class"].value_counts()
    total = len(merged)
    print(f"\n{'='*50}")
    print(f"Classification Summary  ({total} fault results)")
    print(f"{'='*50}")
    for cls in ["Masked", "Acceptable", "Observed", "Warning", "Critical", "Crash", "Hang", "Good", "NoInjection"]:
        n = vc.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"  {cls:<15s}: {n:>6d}  ({pct:5.1f}%)")
    print(f"{'='*50}\n")

    return merged
