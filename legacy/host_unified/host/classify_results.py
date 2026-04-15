#!/usr/bin/env python3
"""Classify fault-injected results against the golden baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NON_INJECTION_STATUSES = {
    "timer_not_fired",
    "timing_window_too_short",
    "invalid_frame",
    "invalid_register",
    "idle",
    "armed",
    "panic_non_json",
}


def classify_row(row: pd.Series) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    results = results[results["event"] == "result"].copy()
    results = results[results["mode"] == "fault"].copy()
    golden = pd.read_csv(args.golden)

    golden_ref = golden[["image_id", "golden_pred_class", "golden_top1_prob"]].rename(
        columns={
            "golden_pred_class": "golden_pred_class_ref",
            "golden_top1_prob": "golden_top1_prob_ref",
        }
    )

    merged = results.merge(golden_ref, on="image_id", how="left")
    merged["fault_class"] = merged.apply(classify_row, axis=1)
    merged.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
