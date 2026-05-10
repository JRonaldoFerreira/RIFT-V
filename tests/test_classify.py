"""Unit tests for :mod:`riftv.classify` row classifier."""
from __future__ import annotations

import pandas as pd

from riftv.classify import classify_results, classify_row


def _row(**kwargs) -> pd.Series:
    """Build a result row with sensible defaults overridden by *kwargs*."""
    base = {
        "status": "ok",
        "injection_status": "injected",
        "fault_model": "register",
        "pred_class": 1,
        "label": 1,
        "top1_prob": 0.95,
        "golden_pred_class_ref": 1,
        "golden_top1_prob_ref": 0.95,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_status_ok_masked() -> None:
    # Identical to golden — Masked.
    assert classify_row(_row()) == "Masked"


def test_crash_status_classifies_as_crash() -> None:
    # status != ok and not fatal_hang -> Crash.
    row = _row(status="crash", injection_status="crash")
    assert classify_row(row) == "Crash"


def test_fatal_hang_status_classifies_as_hang() -> None:
    # status == fatal_hang -> Hang (separate from Crash).
    row = _row(status="fatal_hang", injection_status="fatal_hang")
    assert classify_row(row) == "Hang"


def test_fatal_hang_in_injection_status_only() -> None:
    # Older rows may carry fatal_hang only in injection_status; still Hang.
    row = _row(status="crash", injection_status="fatal_hang")
    assert classify_row(row) == "Hang"


def test_register_no_injection_marker_masked_path() -> None:
    # NoInjection bypasses the prediction comparison.
    row = _row(injection_status="timer_not_fired")
    assert classify_row(row) == "NoInjection"


def test_critical_when_pred_diverges_strongly() -> None:
    row = _row(pred_class=7, label=1, top1_prob=0.5)
    assert classify_row(row) == "Critical"


def test_classify_results_recovers_missing_golden_from_fault_rows(tmp_path) -> None:
    """If a golden-phase row for one image_id was dropped (e.g., UART loss),
    classify_results must fall back to the golden_pred_class / golden_top1_prob
    that the firmware echoes on each fault row, instead of producing a residual
    ``Observed`` class for that image_id.
    """
    rows = []
    # Golden phase: image_id=0 only (image_id=1 is "missing" / dropped).
    rows.append({
        "event": "result", "phase": "golden", "fault_model": "register",
        "image_id": 0, "label": 0, "pred_class": 0, "top1_prob": 0.9,
        "status": "ok", "injection_status": "",
        "golden_pred_class": 0, "golden_top1_prob": 0.9,
    })
    # Fault rows for image_id=0 (golden ref present in lookup).
    rows.append({
        "event": "result", "phase": "fault", "fault_model": "register",
        "image_id": 0, "label": 0, "pred_class": 0, "top1_prob": 0.9,
        "status": "ok", "injection_status": "injected",
        "golden_pred_class": 0, "golden_top1_prob": 0.9,
    })
    # Fault rows for image_id=1 (no golden phase row, but golden fields echoed).
    rows.append({
        "event": "result", "phase": "fault", "fault_model": "register",
        "image_id": 1, "label": 1, "pred_class": 1, "top1_prob": 0.95,
        "status": "ok", "injection_status": "injected",
        "golden_pred_class": 1, "golden_top1_prob": 0.95,
    })

    df = pd.DataFrame(rows)
    results_csv = tmp_path / "results.csv"
    out_csv = tmp_path / "classified.csv"
    df.to_csv(results_csv, index=False)

    out = classify_results(results_csv, out_csv)
    # No residual Observed class — both image_ids classify as Masked.
    assert "Observed" not in out["fault_class"].value_counts().index
    cls_for_img1 = out.loc[out["image_id"] == 1, "fault_class"].iloc[0]
    assert cls_for_img1 == "Masked"
