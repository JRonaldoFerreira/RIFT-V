"""UART receiver — capture JSON-line stream from ESP32 firmware."""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import serial

from tflmfi.config import SerialConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Legacy fallback: used only when campaign_type is not provided
CSV_COLUMNS = [
    "event", "phase", "mode", "fault_model",
    "fault_id", "image_id", "label", "pred_class", "top1_prob",
    "golden_pred_class", "golden_top1_prob",
    "operator_index", "operator_name",
    "tensor_index", "byte_in_tensor",
    "register_id", "register_name", "time_slot_index", "time_slot_count", "delay_us",
    "bit_index", "status", "injection_status",
    "mean_duration_us", "samples",
    "uart_line",
]

# ---------------------------------------------------------------------------
# Per-campaign column schemas
# ---------------------------------------------------------------------------

_COMMON_COLS = [
    "event", "phase", "mode", "fault_model",
]

_RESULT_COLS = [
    "image_id", "label", "pred_class", "top1_prob",
    "golden_pred_class", "golden_top1_prob",
    "status", "injection_status",
]

_MEMORY_COLS = ["operator_index", "tensor_index", "byte_in_tensor", "bit_index"]

_REGISTER_COLS = [
    "operator_index", "operator_name",
    "register_id", "register_name",
    "time_slot_index", "time_slot_count", "delay_us",
    "bit_index",
]

_PROFILE_COLS = ["mean_duration_us", "samples"]


def _columns_for(
    campaign_type: str,
    is_multi: bool,
    groups_csv: Optional[Path] = None,
) -> List[str]:
    """Return the ordered column list appropriate for *campaign_type*."""
    cols = list(_COMMON_COLS)

    if is_multi:
        cols += ["group_id", "fault_count"]
        if groups_csv and groups_csv.exists():
            gdf = pd.read_csv(groups_csv, nrows=0)
            cols += [c for c in gdf.columns if c.startswith("pool_index_")]
    else:
        cols.append("fault_id")

    cols += list(_RESULT_COLS)

    if campaign_type == "memory":
        if not is_multi:
            cols += _MEMORY_COLS
    elif campaign_type == "register":
        if not is_multi:
            cols += _REGISTER_COLS
        cols += _PROFILE_COLS

    return cols

_MAX_LINE_BYTES = 1_048_576

_PANIC_MARKERS = ("guru meditation error", "panic'ed", "stack protection fault")

_SYSTEM_WARNING_MARKERS = (
    "task_wdt", "task watchdog", "register dump",
    "backtrace:", "abort()", "stack smashing",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _uart_reader_thread(
    ser: serial.Serial,
    line_queue: "queue.Queue[str]",
    stop: threading.Event,
) -> None:
    buf = bytearray()
    while not stop.is_set():
        try:
            waiting = ser.in_waiting
            chunk = ser.read(waiting) if waiting else ser.read(1)
        except (serial.SerialException, TypeError, OSError):
            break
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            raw = bytes(buf[:i])
            del buf[:i + 1]
            if len(raw) > _MAX_LINE_BYTES:
                print(f"[WARN] Dropping oversized line ({len(raw)} B).", file=sys.stderr)
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                line_queue.put(line)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_csv(
    jsonl_path: Path,
    csv_path: Path,
    campaign_type: Optional[str] = None,
    is_multi: bool = False,
    groups_csv: Optional[Path] = None,
) -> None:
    """Convert a JSONL capture file into a CSV with campaign-specific columns.

    Parameters
    ----------
    jsonl_path : Path
        Source JSONL file produced by UART capture.
    csv_path : Path
        Destination CSV.
    campaign_type : str, optional
        ``"memory"`` or ``"register"``.  When *None* the legacy
        catch-all ``CSV_COLUMNS`` list is used for backwards compatibility.
    is_multi : bool
        Whether this is a multi-fault campaign.
    groups_csv : Path, optional
        Path to the ``*_fault_groups.csv`` file.  When supplied **and**
        *is_multi* is True, pool_index columns are merged into the
        result rows.
    """
    if not jsonl_path.exists():
        print("[WARN] No JSONL found; skipping CSV generation.", file=sys.stderr)
        return

    rows: List[Dict[str, Any]] = []
    markers = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(entry.get("event", "")).startswith("_"):
                markers += 1
                continue
            rows.append(entry)

    if not rows:
        print("[WARN] JSONL has no data rows; skipping CSV.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)

    # Enrich multi-fault rows with group member indices
    if is_multi and groups_csv and groups_csv.exists():
        gdf = pd.read_csv(groups_csv)
        if "group_id" in df.columns and "group_id" in gdf.columns:
            df = df.merge(gdf, on="group_id", how="left")

    # Select columns
    if campaign_type is not None:
        columns = _columns_for(campaign_type, is_multi, groups_csv)
    else:
        columns = CSV_COLUMNS

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df[columns].to_csv(csv_path, index=False)

    results = df[df["event"] == "result"]
    ok = len(results[results["status"] == "ok"]) if len(results) else 0
    crashes = len(results[results["status"] == "crash"]) if len(results) else 0
    print(f"[CSV] {len(rows)} rows ({markers} markers skipped), {ok} ok, {crashes} crashes", file=sys.stderr)
    print(f"[CSV] {jsonl_path} -> {csv_path}", file=sys.stderr)


def capture(
    serial_cfg: SerialConfig,
    jsonl_path: Path,
    csv_path: Path,
    stop_after: Optional[int] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    campaign_type: Optional[str] = None,
    is_multi: bool = False,
    groups_csv: Optional[Path] = None,
) -> None:
    """Open the serial port, capture the campaign, write JSONL, then build CSV.

    Parameters
    ----------
    serial_cfg : SerialConfig
        Port, baud, and boot timeout settings.
    jsonl_path : Path
        Where to stream JSON lines.
    csv_path : Path
        Where to write the final CSV.
    stop_after : int, optional
        Stop after *N* valid data messages.
    on_result : callable, optional
        Called with each result payload dict for live monitoring.
    campaign_type : str, optional
        ``"memory"`` or ``"register"``.  Passed through to :func:`build_csv`.
    is_multi : bool
        Whether this is a multi-fault campaign.
    groups_csv : Path, optional
        Path to the ``*_fault_groups.csv`` for multi-fault enrichment.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Truncate stale capture files from previous runs
    if jsonl_path.exists():
        jsonl_path.unlink()
    if csv_path.exists():
        csv_path.unlink()

    valid = 0
    campaign_done = False
    lq: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    port = serial_cfg.port
    baud = serial_cfg.baud

    # ---- Phase 1: trigger a hardware reset via DTR/RTS ----
    print(f"[uart] Listening on {port} @ {baud} baud", file=sys.stderr)
    try:
        with serial.Serial(port, baud, timeout=0.05) as rst:
            rst.dtr = False
            rst.rts = True
            time.sleep(0.1)
            rst.rts = False
            rst.dtr = False
            time.sleep(0.05)
    except (serial.SerialException, OSError) as exc:
        print(f"[uart] Reset toggle warning: {exc}", file=sys.stderr)

    # ---- Phase 2: wait for USB re-enumeration ----
    # On boards that use the built-in USB Serial/JTAG (ESP32-C6, C3, H2),
    # the chip reset causes the USB device to disconnect and reconnect.
    # First check whether the port actually disappeared (reset worked),
    # then poll until it reappears.
    print("[uart] Waiting for port after reset...", file=sys.stderr)

    port_disappeared = False
    vanish_deadline = time.monotonic() + 1.5
    while time.monotonic() < vanish_deadline:
        try:
            probe = serial.Serial(port, baud, timeout=0.05)
            probe.close()
            time.sleep(0.1)
        except (serial.SerialException, OSError):
            port_disappeared = True
            break

    if not port_disappeared:
        print(
            "[uart] Port stayed open — DTR/RTS reset may not have worked. "
            "Will look for an existing boot event.",
            file=sys.stderr,
        )

    ser: Optional[serial.Serial] = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
            break
        except (serial.SerialException, OSError):
            time.sleep(0.25)

    if ser is None:
        print(f"[ERROR] Could not reopen {port} after reset.", file=sys.stderr)
        return

    try:
        reader = threading.Thread(target=_uart_reader_thread, args=(ser, lq, stop), daemon=True)
        reader.start()

        # Wait for boot (or detect firmware already running)
        print("[uart] Waiting for boot event...", file=sys.stderr)
        while True:
            try:
                line = lq.get(timeout=float(serial_cfg.boot_timeout_s))
            except queue.Empty:
                print("[ERROR] Boot timeout.", file=sys.stderr)
                stop.set()
                reader.join(timeout=2.0)
                return
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = payload.get("event")
            if evt == "boot":
                print("[uart] Boot received — sending start signal.", file=sys.stderr)
                try:
                    ser.write(b'{"cmd":"start"}\n')
                    ser.flush()
                except (serial.SerialException, OSError) as exc:
                    print(f"[uart] Warning: could not send start: {exc}", file=sys.stderr)
                print("[uart] Campaign running.", file=sys.stderr)
                break
            if evt in ("result", "progress", "done"):
                print(
                    f"[uart] Firmware already running ('{evt}' before boot) "
                    "— joining campaign.",
                    file=sys.stderr,
                )
                lq.put(line)
                break

        # Main receive loop
        while True:
            line = lq.get()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                low = line.lower()
                is_panic = any(m in low for m in _PANIC_MARKERS)
                is_warn = any(m in low for m in _SYSTEM_WARNING_MARKERS)
                if is_panic or is_warn:
                    _append_jsonl(jsonl_path, {
                        "event": "_system_warning",
                        "warning_type": "panic" if is_panic else "system_warning",
                        "uart_line": line[:500],
                    })
                else:
                    print(f"[WARN] Non-JSON: {line[:200]}", file=sys.stderr)
                continue

            event = payload.get("event")

            if event == "progress":
                phase = payload.get("phase", "?")
                cur, tot = payload.get("current", 0), payload.get("total", 0)
                pct = (cur / tot * 100) if tot else 0
                print(f"[PROGRESS] {phase}: {cur}/{tot} ({pct:.1f}%)", file=sys.stderr)
                continue

            if event == "done":
                campaign_done = True
                print("[uart] Campaign finished.", file=sys.stderr)
                break

            _append_jsonl(jsonl_path, payload)
            valid += 1

            if event == "result":
                phase = payload.get("phase", payload.get("mode", "?"))
                fid = payload.get("fault_id", "?")
                img = payload.get("image_id", "?")
                st = payload.get("injection_status", "?")
                print(f"[RX {valid}] phase={phase} fault_id={fid} image={img} inj={st}", file=sys.stderr)
                if on_result:
                    on_result(payload)

            if stop_after is not None and valid >= stop_after:
                break

        stop.set()
        reader.join(timeout=2.0)
    finally:
        stop.set()
        ser.close()

    if not campaign_done:
        print("[WARN] Capture ended WITHOUT 'done' event — results may be incomplete.", file=sys.stderr)

    build_csv(jsonl_path, csv_path, campaign_type=campaign_type,
              is_multi=is_multi, groups_csv=groups_csv)


