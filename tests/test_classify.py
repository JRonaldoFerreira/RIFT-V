"""Unit tests for :mod:`riftv.classify` row classifier."""
from __future__ import annotations

import pandas as pd

from riftv.classify import classify_row


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
