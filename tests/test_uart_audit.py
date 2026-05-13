"""Round-trip tests for the host-side audit columns in :mod:`riftv.uart`."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from riftv.uart import (
    CSV_COLUMNS,
    RECOVERY_METHOD_DTR_RTS,
    RECOVERY_METHOD_NATURAL_BOOT,
    RECOVERY_METHOD_NONE,
    RECOVERY_METHOD_USB_JTAG,
    RECOVERY_METHOD_WATCHDOG,
    _columns_for,
    build_csv,
)


_AUDIT_COLS = (
    "elapsed_ms",
    "timeout_ms",
    "boot_seen",
    "result_seen",
    "job_ack_seen",
    "recovery_method",
    "inferred",
)


def test_audit_columns_present_in_legacy_schema() -> None:
    # All audit columns must exist in the legacy fallback CSV_COLUMNS.
    for col in _AUDIT_COLS:
        assert col in CSV_COLUMNS, f"missing audit column {col!r} in CSV_COLUMNS"


def test_audit_columns_present_in_per_campaign_schema() -> None:
    for ctype in ("memory", "register"):
        cols = _columns_for(campaign_type=ctype, is_multi=False)
        for col in _AUDIT_COLS:
            assert col in cols, (
                f"missing audit column {col!r} in {ctype} campaign columns"
            )


def test_recovery_method_constants_are_distinct() -> None:
    methods = {
        RECOVERY_METHOD_DTR_RTS,
        RECOVERY_METHOD_USB_JTAG,
        RECOVERY_METHOD_WATCHDOG,
        RECOVERY_METHOD_NATURAL_BOOT,
        RECOVERY_METHOD_NONE,
    }
    assert len(methods) == 5


def test_build_csv_propagates_audit_fields(tmp_path: Path) -> None:
    """A JSONL row with audit fields must surface them in the rendered CSV."""
    jsonl = tmp_path / "results.jsonl"
    csv = tmp_path / "results.csv"

    rows = [
        {
            "event": "result",
            "phase": "fault",
            "mode": "fault",
            "fault_model": "register",
            "fault_id": 42,
            "image_id": 3,
            "label": 7,
            "pred_class": 7,
            "top1_prob": 0.91,
            "golden_pred_class": 7,
            "golden_top1_prob": 0.95,
            "status": "ok",
            "injection_status": "injected",
            "elapsed_ms": 18.7,
            "timeout_ms": 1500,
            "boot_seen": False,
            "result_seen": True,
            "job_ack_seen": True,
            "recovery_method": RECOVERY_METHOD_NONE,
            "inferred": False,
        },
        {
            "event": "result",
            "phase": "fault",
            "mode": "fault",
            "fault_model": "register",
            "fault_id": 43,
            "image_id": 0,
            "label": 2,
            "pred_class": -1,
            "top1_prob": 0.0,
            "status": "fatal_hang",
            "injection_status": "fatal_hang",
            "elapsed_ms": 1500.0,
            "timeout_ms": 1500,
            "boot_seen": False,
            "result_seen": False,
            "job_ack_seen": False,
            "recovery_method": RECOVERY_METHOD_NONE,
            "inferred": False,
        },
        # Inherited row: same fault_id as the parent (44/0), but
        # synthesized for image 1 because the parent left the chip in
        # an unrecoverable state (fatal_hang) and image 1 could not be
        # physically dispatched.  recovery_method/boot_seen are
        # inherited from the parent row.
        {
            "event": "result",
            "phase": "fault",
            "mode": "fault",
            "fault_model": "register",
            "fault_id": 44,
            "image_id": 0,
            "label": 1,
            "pred_class": -1,
            "top1_prob": 0.0,
            "status": "crash",
            "injection_status": "crash",
            "elapsed_ms": 2200.0,
            "timeout_ms": 1500,
            "boot_seen": True,
            "result_seen": False,
            "job_ack_seen": False,
            "recovery_method": "natural_boot",
            "inferred": False,
        },
        {
            "event": "result",
            "phase": "fault",
            "mode": "fault",
            "fault_model": "register",
            "fault_id": 44,
            "image_id": 1,
            "label": 1,
            "pred_class": -1,
            "top1_prob": 0.0,
            "status": "crash",
            "injection_status": "crash",
            "elapsed_ms": None,
            "timeout_ms": 1500,
            "boot_seen": True,
            "result_seen": False,
            "job_ack_seen": False,
            "recovery_method": "natural_boot",
            "inferred": True,
        },
    ]
    with jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    build_csv(jsonl, csv, campaign_type="register", is_multi=False)
    df = pd.read_csv(csv)

    # Exactly the four result rows we wrote.
    assert len(df) == 4

    # Audit columns must be present and carry the original values.
    for col in _AUDIT_COLS:
        assert col in df.columns

    ok_row = df[df["fault_id"] == 42].iloc[0]
    assert ok_row["recovery_method"] == RECOVERY_METHOD_NONE
    assert bool(ok_row["job_ack_seen"]) is True
    assert bool(ok_row["result_seen"]) is True
    assert float(ok_row["elapsed_ms"]) == 18.7
    assert int(ok_row["timeout_ms"]) == 1500
    assert bool(ok_row["inferred"]) is False

    hang_row = df[df["fault_id"] == 43].iloc[0]
    assert hang_row["status"] == "fatal_hang"
    assert hang_row["injection_status"] == "fatal_hang"
    assert bool(hang_row["job_ack_seen"]) is False
    assert bool(hang_row["result_seen"]) is False
    assert hang_row["recovery_method"] == RECOVERY_METHOD_NONE
    assert bool(hang_row["inferred"]) is False

    # Parent crash row (image 0): real measurement, not inferred.
    parent = df[(df["fault_id"] == 44) & (df["image_id"] == 0)].iloc[0]
    assert parent["recovery_method"] == "natural_boot"
    assert bool(parent["boot_seen"]) is True
    assert bool(parent["inferred"]) is False
    assert float(parent["elapsed_ms"]) == 2200.0

    # Inherited row (image 1): inferred=True, elapsed_ms NaN, but
    # recovery_method and boot_seen inherited from parent.
    child = df[(df["fault_id"] == 44) & (df["image_id"] == 1)].iloc[0]
    assert child["recovery_method"] == "natural_boot"
    assert bool(child["boot_seen"]) is True
    assert bool(child["inferred"]) is True
    assert pd.isna(child["elapsed_ms"])
    assert bool(child["result_seen"]) is False
    assert bool(child["job_ack_seen"]) is False
