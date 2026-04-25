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

from riftv.config import SerialConfig

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
    # Host-side audit fields populated by the job scheduler:
    #   - elapsed_ms / timeout_ms : wall-clock measurements per job
    #   - boot_seen / result_seen / job_ack_seen : protocol observation flags
    #   - recovery_method : which reset strategy brought the chip back
    #   - inferred : True for rows synthesized from a parent fault crash
    #                (image was not actually dispatched)
    "elapsed_ms", "timeout_ms",
    "boot_seen", "result_seen", "job_ack_seen", "recovery_method",
    "inferred",
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
    # Host-side audit fields populated by the job scheduler:
    "elapsed_ms", "timeout_ms",
    "boot_seen", "result_seen", "job_ack_seen", "recovery_method",
    "inferred",
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


def _load_valid_fault_ids(run_dir: Path, is_multi: bool) -> Optional[set]:
    """Load valid fault/group IDs from the selected-faults CSV in *run_dir*.

    Returns ``None`` when no suitable CSV is found (dedup-only mode).
    """
    for csv_path in sorted(run_dir.glob("selected_*faults.csv")):
        try:
            sdf = pd.read_csv(csv_path, usecols=["fault_id"])
            return set(sdf["fault_id"].dropna().astype(int))
        except (ValueError, KeyError):
            continue
    if is_multi:
        for csv_path in sorted(run_dir.glob("*_fault_groups.csv")):
            try:
                sdf = pd.read_csv(csv_path, usecols=["group_id"])
                return set(sdf["group_id"].dropna().astype(int))
            except (ValueError, KeyError):
                continue
    return None


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

    # -- Deduplicate result rows (firmware crash-recovery may re-send) --
    result_mask = df["event"] == "result"
    if result_mask.any():
        non_results = df[~result_mask]
        results_df = df[result_mask].copy()
        before_dedup = len(results_df)

        if is_multi and "group_id" in results_df.columns:
            golden = results_df[results_df["phase"] == "golden"]
            fault = results_df[results_df["phase"] != "golden"]
            golden = golden.drop_duplicates(subset=["image_id"], keep="last")
            fault = fault.drop_duplicates(subset=["group_id", "image_id"], keep="last")
            results_df = pd.concat([golden, fault], ignore_index=True)
        elif "fault_id" in results_df.columns and "image_id" in results_df.columns:
            results_df = results_df.drop_duplicates(
                subset=["fault_id", "image_id"], keep="last",
            )

        dropped = before_dedup - len(results_df)
        if dropped:
            print(
                f"[CSV] Deduplicated: dropped {dropped} duplicate result rows",
                file=sys.stderr,
            )

        # Filter out corrupt fault IDs caused by crash-induced memory corruption
        valid_ids = _load_valid_fault_ids(jsonl_path.parent, is_multi)
        if valid_ids is not None:
            valid_ids_with_golden = valid_ids | {-1}
            if is_multi and "group_id" in results_df.columns:
                corrupt_mask = (
                    (results_df["phase"] != "golden")
                    & ~results_df["group_id"].isin(valid_ids)
                )
            elif "fault_id" in results_df.columns:
                corrupt_mask = ~results_df["fault_id"].isin(valid_ids_with_golden)
            else:
                corrupt_mask = pd.Series(False, index=results_df.index)

            n_corrupt = int(corrupt_mask.sum())
            if n_corrupt:
                print(
                    f"[CSV] Filtered {n_corrupt} rows with corrupt fault IDs",
                    file=sys.stderr,
                )
                results_df = results_df[~corrupt_mask]

        # Dedup non-result rows (e.g. op_profile after firmware restarts)
        op_mask = non_results["event"] == "op_profile"
        if op_mask.any() and "operator_index" in non_results.columns:
            ops = non_results[op_mask].drop_duplicates(
                subset=["operator_index"], keep="last",
            )
            rest = non_results[~op_mask]
            non_results = pd.concat([rest, ops], ignore_index=True)

        df = pd.concat([non_results, results_df], ignore_index=True)

    # Enrich result rows with operator_name from op_profile events
    if "operator_index" in df.columns and "operator_name" in df.columns:
        op_profiles = df[df["event"] == "op_profile"].dropna(subset=["operator_index", "operator_name"])
        if not op_profiles.empty:
            op_map = op_profiles.drop_duplicates(subset=["operator_index"], keep="last")
            op_map = op_map.set_index("operator_index")["operator_name"]
            missing = df["operator_name"].isna() & df["operator_index"].notna()
            if missing.any():
                df.loc[missing, "operator_name"] = (
                    df.loc[missing, "operator_index"].map(op_map)
                )

    # Enrich register-campaign result rows with fault metadata from the
    # selected-faults CSV.  This fills in register_id, register_name,
    # bit_index, operator_index, operator_name, time_slot_* for crash
    # rows (host-synthesized) or any firmware row that is missing fields.
    if campaign_type == "register" and not is_multi:
        faults_csv = None
        for p in sorted(jsonl_path.parent.glob("selected_*faults.csv")):
            faults_csv = p
            break
        if faults_csv and faults_csv.exists():
            try:
                fdf = pd.read_csv(faults_csv)
                _CSV_TO_JSON = {
                    "op_index": "operator_index",
                    "op_name": "operator_name",
                }
                fdf = fdf.rename(columns=_CSV_TO_JSON)
                merge_cols = [
                    c for c in [
                        "fault_id", "operator_index", "operator_name",
                        "register_id", "register_name",
                        "time_slot_index", "time_slot_count", "bit_index",
                    ] if c in fdf.columns
                ]
                if "fault_id" in merge_cols and "fault_id" in df.columns:
                    fdf_merge = fdf[merge_cols].drop_duplicates(subset=["fault_id"])
                    # Use suffixes to detect which columns to backfill
                    df = df.merge(fdf_merge, on="fault_id", how="left",
                                  suffixes=("", "_meta"))
                    for col in merge_cols:
                        if col == "fault_id":
                            continue
                        meta_col = f"{col}_meta"
                        if meta_col in df.columns:
                            if col in df.columns:
                                df[col] = df[col].fillna(df[meta_col])
                            else:
                                df[col] = df[meta_col]
                            df.drop(columns=[meta_col], inplace=True)
            except Exception as exc:
                print(f"[CSV] Warning: could not enrich from faults CSV: {exc}",
                      file=sys.stderr)

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


def _resolve_reset_flag(serial_cfg: SerialConfig) -> bool:
    """Decide whether to attempt a DTR/RTS hardware reset before capture."""
    if serial_cfg.reset_before_capture is not None:
        return serial_cfg.reset_before_capture
    # Auto: reset makes sense only when a USB-UART bridge drives DTR/RTS.
    return serial_cfg.transport == "uart0"


# ---------------------------------------------------------------------------
# Hardware-reset strategies (used by the hang-recovery path)
#
# We try several reset mechanisms in sequence so that recovery does not
# depend on a manual RESET button press, regardless of whether the board
# uses a USB-UART bridge or a native USB Serial/JTAG peripheral.
# ---------------------------------------------------------------------------

def _port_vanished(port: str, baud: int, deadline_s: float = 1.5) -> bool:
    """Return True if *port* disappears within *deadline_s* (real chip reset)."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            probe = serial.Serial(port, baud, timeout=0.05)
            probe.close()
            time.sleep(0.1)
        except (serial.SerialException, OSError):
            return True
    return False


def _try_reset_dtr_rts_classic(port: str, baud: int) -> None:
    """Classic DTR/RTS toggle used by CP210x/CH340/FTDI bridge boards."""
    try:
        with serial.Serial(port, baud, timeout=0.05) as rst:
            rst.dtr = False
            rst.rts = True
            time.sleep(0.1)
            rst.rts = False
            rst.dtr = False
            time.sleep(0.05)
    except (serial.SerialException, OSError) as exc:
        print(f"[uart] DTR/RTS classic toggle warning: {exc}", file=sys.stderr)


def _try_reset_usb_jtag_serial(port: str, baud: int) -> None:
    """Esptool-compatible USB Serial/JTAG reset sequence.

    Mirrors esptool's ``USBJTAGSerialReset``: a specific DTR/RTS dance that
    the on-chip USB Serial/JTAG peripheral interprets as a chip reset.  It
    is the recommended host-side stimulus for native USB boards, but its
    success still depends on driver, OS reenumeration timing, and the
    state of the USB peripheral itself.
    """
    try:
        with serial.Serial(port, baud, timeout=0.05) as rst:
            rst.setRTS(False)
            rst.setDTR(False)  # idle
            time.sleep(0.1)
            rst.setDTR(True)
            rst.setRTS(False)
            rst.setDTR(False)  # avoid (0,0)
            rst.setRTS(True)
            rst.setDTR(False)
            rst.setRTS(True)   # chip in reset
            time.sleep(0.1)
            rst.setRTS(False)
            rst.setDTR(False)
    except (serial.SerialException, OSError) as exc:
        print(f"[uart] USB-JTAG-Serial reset warning: {exc}", file=sys.stderr)


def _try_reset_break(port: str, baud: int) -> None:
    """Send a serial BREAK signal — best-effort stimulus, not a guaranteed
    hardware reset.

    BREAK can dislodge a stuck firmware in some scenarios (e.g. UART
    driver waiting on a frame), but it does not unconditionally reset the
    SoC and must not be treated as equivalent to DTR/RTS or watchdog
    reset.  It is included in the chain because it is cheap and harmless.
    """
    try:
        with serial.Serial(port, baud, timeout=0.05) as rst:
            rst.send_break(duration=0.5)
            time.sleep(0.2)
    except (serial.SerialException, OSError, NotImplementedError) as exc:
        print(f"[uart] Serial BREAK warning: {exc}", file=sys.stderr)


# Stable, machine-readable identifiers used in the ``recovery_method`` field
# written to JSONL/CSV.  These are kept deliberately short so they are easy
# to filter on in pandas/Excel.
RECOVERY_METHOD_DTR_RTS = "dtr_rts_classic"
RECOVERY_METHOD_USB_JTAG = "usb_jtag_serial"
RECOVERY_METHOD_BREAK = "serial_break"
RECOVERY_METHOD_WATCHDOG = "watchdog_wait"
RECOVERY_METHOD_NATURAL_BOOT = "natural_boot"
RECOVERY_METHOD_NONE = "none"


def _attempt_chip_reset(
    port: str,
    baud: int,
    transport: str,
    verify_boot: Callable[[float], bool],
    short_timeout_s: float = 4.0,
    watchdog_timeout_s: float = 8.0,
) -> Optional[str]:
    """Run the reset-strategy chain, gated on a real ``boot`` event.

    Order of attempts:

    1. DTR/RTS classic toggle   — fastest, works on bridge boards.
    2. USB-JTAG-Serial sequence — esptool-compatible native-USB stimulus.
    3. Serial BREAK             — best-effort host stimulus.
    4. Watchdog wait            — gives the firmware's TWDT time to fire.

    Success criterion
    -----------------
    The **primary** signal of recovery is *verify_boot()* returning True
    (firmware emitted a fresh ``{"event":"boot"}`` line).

    Port vanish/reappear is **not** used as a success signal: on bridge
    boards (CP210x/CH340/FTDI) the host-visible USB device is the bridge
    chip, not the ESP32, so DTR/RTS can reset the SoC while the COM port
    stays open.  We therefore log vanish events as auxiliary evidence
    only and keep going through the chain until verify_boot() succeeds.

    Returns the stable identifier of the recovery method that produced a
    boot event (one of the ``RECOVERY_METHOD_*`` constants), or ``None``
    if every stage including the watchdog wait was inconclusive.
    """
    strategies: List[Tuple[str, str, Callable[[str, int], None]]] = [
        (RECOVERY_METHOD_DTR_RTS,  "DTR/RTS classic toggle",   _try_reset_dtr_rts_classic),
        (RECOVERY_METHOD_USB_JTAG, "USB-JTAG-Serial sequence", _try_reset_usb_jtag_serial),
        (RECOVERY_METHOD_BREAK,    "Serial BREAK",             _try_reset_break),
    ]
    if transport == "usb_serial_jtag":
        # Native USB: the JTAG-Serial sequence is the one that can actually
        # drive a reset on this transport, so try it first.
        strategies[0], strategies[1] = strategies[1], strategies[0]

    for method_id, label, strat in strategies:
        print(f"[uart] Reset attempt: {label}...", file=sys.stderr)
        strat(port, baud)

        # Auxiliary evidence (log only — not used to decide success).
        if _port_vanished(port, baud, deadline_s=1.0):
            print(
                f"[uart] {label}: port vanished (auxiliary evidence of reset).",
                file=sys.stderr,
            )

        if verify_boot(short_timeout_s):
            print(
                f"[uart] {label}: firmware boot event received — recovery OK.",
                file=sys.stderr,
            )
            return method_id
        print(
            f"[uart] {label}: no boot event within {short_timeout_s:.0f}s; "
            "trying next strategy...",
            file=sys.stderr,
        )

    # Final fallback: rely on the firmware's own task watchdog (TWDT) to
    # reboot the chip on its own.  This requires the firmware to keep the
    # watchdog armed during inference; if it does, recovery proceeds
    # without any external action.
    print(
        f"[uart] All host-driven stimuli inconclusive; waiting up to "
        f"{watchdog_timeout_s:.0f}s for the firmware task watchdog to "
        "reboot the chip...",
        file=sys.stderr,
    )
    if verify_boot(watchdog_timeout_s):
        return RECOVERY_METHOD_WATCHDOG
    return None


def _open_port_after_reset(
    port: str,
    baud: int,
    do_reset: bool,
    transport: str,
) -> Optional[serial.Serial]:
    """Phase 1+2: optionally reset the board, then (re-)open the capture port.

    For ``uart0`` (bridge) boards the existing DTR/RTS toggle + re-enumeration
    logic is used.  For ``usb_serial_jtag`` (native USB) boards the reset is
    skipped and we simply wait for the port to appear (it may vanish briefly
    after a flash-induced chip reset).
    """

    if do_reset:
        # ---- DTR/RTS reset (classic UART bridge) ----
        print(f"[uart] Triggering DTR/RTS reset on {port}...", file=sys.stderr)
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

        # Check whether the port vanished (indicates real reset)
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
                "[uart] Port stayed open \u2014 DTR/RTS reset may not have worked. "
                "Will look for an existing boot event.",
                file=sys.stderr,
            )
    else:
        # ---- USB native: no DTR/RTS, wait for port after flash-induced reset ----
        print(
            f"[uart] Transport={transport} \u2014 skipping DTR/RTS reset. "
            "Waiting for port to appear...",
            file=sys.stderr,
        )
        # Give the chip time to reboot after flash
        time.sleep(0.5)

    # Poll until the capture port is available
    ser: Optional[serial.Serial] = None
    reopen_timeout = 8.0 if transport == "usb_serial_jtag" else 5.0
    deadline = time.monotonic() + reopen_timeout
    while time.monotonic() < deadline:
        try:
            ser = serial.Serial(port, baud, timeout=0.05)
            break
        except (serial.SerialException, OSError):
            time.sleep(0.25)

    return ser


def capture(
    serial_cfg: SerialConfig,
    jsonl_path: Path,
    csv_path: Path,
    fault_count: int = 0,
    image_count: int = 0,
    stop_after: Optional[int] = None,
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    campaign_type: Optional[str] = None,
    is_multi: bool = False,
    groups_csv: Optional[Path] = None,
) -> None:
    """Drive the campaign via job commands, capture results, write JSONL + CSV.

    The host acts as the campaign scheduler.  It sends commands to firmware:

    - ``run_golden`` — run the golden (fault-free) inference phase
    - ``run_job``    — run a single fault x image inference
    - ``done``       — signal campaign end

    If the firmware crashes during a job (detected by a ``boot`` event),
    the host re-sends ``run_golden`` to reinitialise, then retries or
    marks the job as a crash outcome.

    Parameters
    ----------
    serial_cfg : SerialConfig
        Port, baud, transport and boot timeout settings.
    jsonl_path / csv_path : Path
        Output destinations.
    fault_count : int
        Number of faults (or groups) to iterate over.
    image_count : int
        Number of images per fault.
    stop_after : int, optional
        Stop after *N* valid data messages.
    on_result : callable, optional
        Called with each result payload dict for live monitoring.
    campaign_type : str, optional
        ``"memory"`` or ``"register"``.
    is_multi : bool
        Whether this is a multi-fault campaign.
    groups_csv : Path, optional
        Path to the ``*_fault_groups.csv`` for multi-fault enrichment.
    """
    # If a fault crashes for any image, it will deterministically crash
    # for all remaining images (same register/bit/timing).  Skip retries
    # and immediately mark every remaining image for that fault as crash.
    MAX_CRASH_RETRIES = 0
    GOLDEN_TIMEOUT_S = 120.0

    # Per-job timeout — initially set to the configured maximum; the
    # scheduler may shrink it after the golden phase by measuring the
    # worst inter-result gap and multiplying by *job_timeout_safety_factor*
    # (see :class:`SerialConfig`).
    job_timeout_s: float = float(serial_cfg.job_timeout_max_s)
    # Timestamps of every ``result`` message received during the *first*
    # golden phase, used to calibrate the per-job timeout.
    golden_result_times: List[float] = []

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if jsonl_path.exists():
        jsonl_path.unlink()
    if csv_path.exists():
        csv_path.unlink()

    # ---- Auto-detect fault_count / image_count from run directory ----
    run_dir = jsonl_path.parent
    if fault_count == 0:
        if is_multi:
            for p in sorted(run_dir.glob("*_fault_groups.csv")):
                try:
                    fault_count = len(pd.read_csv(p))
                    break
                except Exception:
                    pass
        if fault_count == 0:
            for p in sorted(run_dir.glob("selected_*faults.csv")):
                try:
                    fault_count = len(pd.read_csv(p))
                    break
                except Exception:
                    pass
    if image_count == 0:
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                image_count = int(meta.get("image_count", meta.get("subset_count", 0)))
            except Exception:
                pass

    # ---- Load fault-ID lookup for crash result generation ----
    fault_id_map: Dict[int, int] = {}
    fault_meta_map: Dict[int, Dict[str, Any]] = {}  # seq_idx -> full row dict
    group_id_map: Dict[int, int] = {}
    if is_multi:
        for p in sorted(run_dir.glob("*_fault_groups.csv")):
            try:
                gdf = pd.read_csv(p)
                if "group_id" in gdf.columns:
                    group_id_map = dict(enumerate(gdf["group_id"].astype(int)))
                break
            except Exception:
                pass
    else:
        for p in sorted(run_dir.glob("selected_*faults.csv")):
            try:
                sdf = pd.read_csv(p)
                if "fault_id" in sdf.columns:
                    fault_id_map = dict(enumerate(sdf["fault_id"].astype(int)))
                    for seq_idx, row in sdf.iterrows():
                        fault_meta_map[int(seq_idx)] = row.to_dict()
                break
            except Exception:
                pass

    valid = 0
    lq: queue.Queue[str] = queue.Queue()
    stop_ev = threading.Event()

    transport = serial_cfg.transport
    port = serial_cfg.run_port or serial_cfg.port
    baud = serial_cfg.baud
    do_reset = _resolve_reset_flag(serial_cfg)

    print(f"[uart] Listening on {port} @ {baud} baud  (transport={transport})", file=sys.stderr)
    print(f"[uart] Campaign: {fault_count} faults x {image_count} images = {fault_count * image_count} jobs", file=sys.stderr)

    # ---- Phase 1+2: reset (if applicable) and open the capture port ----
    ser = _open_port_after_reset(port, baud, do_reset, transport)
    if ser is None:
        print(f"[ERROR] Could not open {port} after reset.", file=sys.stderr)
        return

    # ---- Helper closures ----
    def _send_cmd(cmd_dict: Dict[str, Any]) -> None:
        try:
            ser.write((json.dumps(cmd_dict) + "\n").encode())
            ser.flush()
        except (serial.SerialException, OSError) as exc:
            print(f"[uart] Warning: send failed: {exc}", file=sys.stderr)

    def _read_event(timeout: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Return (payload, None) or (None, error_kind).

        When *timeout* is None, falls back to the (possibly calibrated)
        per-job timeout ``job_timeout_s``.
        """
        if timeout is None:
            timeout = job_timeout_s
        try:
            line = lq.get(timeout=timeout)
        except queue.Empty:
            return None, "timeout"
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
            return None, "non_json"
        return payload, None

    def _wait_for_boot(timeout_s: Optional[float] = None) -> bool:
        timeout = timeout_s or float(serial_cfg.boot_timeout_s)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            payload, err = _read_event(timeout=remaining)
            if err == "timeout":
                return False
            if payload and payload.get("event") == "boot":
                return True
        return False

    # Cache of golden results: image_id -> (pred_class, top1_prob)
    golden_cache: Dict[int, Tuple[int, float]] = {}

    def _run_golden(is_first: bool = True) -> bool:
        """Send run_golden and block until golden_done. Returns success."""
        nonlocal valid
        _send_cmd({"cmd": "run_golden"})
        print("[uart] Running golden phase...", file=sys.stderr)

        while True:
            payload, err = _read_event(timeout=GOLDEN_TIMEOUT_S)
            if err == "timeout":
                print("[uart] Golden phase timeout!", file=sys.stderr)
                return False
            if err:
                continue
            if payload is None:
                continue

            event = payload.get("event")

            if event == "boot":
                print("[uart] Crash during golden! Retrying...", file=sys.stderr)
                _send_cmd({"cmd": "run_golden"})
                continue

            if event in ("result", "op_profile") and is_first:
                _append_jsonl(jsonl_path, payload)
                valid += 1

            # Cache golden predictions for skip_golden patching
            if event == "result" and payload.get("phase") == "golden":
                img_id = payload.get("image_id")
                if img_id is not None:
                    golden_cache[int(img_id)] = (
                        int(payload.get("pred_class", -1)),
                        float(payload.get("top1_prob", 0.0)),
                    )
                # Record the wall-clock arrival time of every golden
                # result so the host can later calibrate the per-job
                # timeout from the worst observed inter-result gap.
                if is_first:
                    golden_result_times.append(time.monotonic())

            if event == "memory_map":
                _append_jsonl(jsonl_path, payload)

            if event == "progress":
                phase = payload.get("phase", "?")
                cur, tot = payload.get("current", 0), payload.get("total", 0)
                pct = (cur / tot * 100) if tot else 0
                print(f"[PROGRESS] {phase}: {cur}/{tot} ({pct:.1f}%)", file=sys.stderr)

            if event == "golden_done":
                print("[uart] Golden phase complete.", file=sys.stderr)
                return True

    def _skip_golden() -> bool:
        """Send skip_golden to bypass inference; host patches golden fields."""
        _send_cmd({"cmd": "skip_golden"})
        print("[uart] Sending skip_golden (host-cached)...", file=sys.stderr)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            payload, err = _read_event(timeout=remaining)
            if err == "timeout":
                print("[uart] skip_golden timeout!", file=sys.stderr)
                return False
            if err:
                continue
            if payload and payload.get("event") == "golden_done":
                print("[uart] Golden skipped (host-cached).", file=sys.stderr)
                return True
        return False

    def _verify_boot_during_recovery(timeout_s: float) -> bool:
        """Open *port*, parse JSON lines, return True iff a ``boot`` event
        is observed within *timeout_s*.

        Used as the success criterion for :func:`_attempt_chip_reset`: the
        firmware emitting a fresh ``{"event":"boot"}`` line is the only
        proof that the chip actually rebooted and is responsive again.

        The probe Serial is always closed before returning so the next
        strategy (or the final reader-thread reopen) can take ownership
        of the port.
        """
        # Reopening can fail briefly while the USB device re-enumerates
        # (especially on usb_serial_jtag).  Poll until we get a handle or
        # exhaust the budget.
        probe: Optional[serial.Serial] = None
        open_deadline = time.monotonic() + min(timeout_s, 3.0)
        while time.monotonic() < open_deadline:
            try:
                probe = serial.Serial(port, baud, timeout=0.05)
                break
            except (serial.SerialException, OSError):
                time.sleep(0.2)
        if probe is None:
            return False

        try:
            deadline = time.monotonic() + timeout_s
            buf = bytearray()
            while time.monotonic() < deadline:
                try:
                    waiting = probe.in_waiting
                    chunk = probe.read(waiting) if waiting else probe.read(1)
                except (serial.SerialException, OSError):
                    return False
                if not chunk:
                    continue
                buf.extend(chunk)
                while True:
                    i = buf.find(b"\n")
                    if i < 0:
                        break
                    line = bytes(buf[:i]).decode("utf-8", errors="replace").strip()
                    del buf[:i + 1]
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("event") == "boot":
                        return True
            return False
        finally:
            try:
                probe.close()
            except Exception:
                pass

    def _force_hw_reset_and_reconnect() -> Optional[str]:
        """Close the port, force a chip reset, reopen, and confirm boot.

        Returns the stable identifier of the recovery method that produced
        the boot event (one of the ``RECOVERY_METHOD_*`` constants), or
        ``None`` if recovery ultimately failed.  Callers are expected to
        record this identifier in the resulting JSONL/CSV row so post-hoc
        analysis can attribute hangs to the strategy that fixed them.
        """
        nonlocal ser, reader

        # 1. Stop the reader thread
        stop_ev.set()
        reader.join(timeout=3.0)

        # 2. Close the current port
        try:
            ser.close()
        except Exception:
            pass

        # 3. Drain leftover data from the queue
        while not lq.empty():
            try:
                lq.get_nowait()
            except queue.Empty:
                break

        # 4. Reset-strategy chain, gated on a real boot event.
        watchdog_wait = max(8.0, float(serial_cfg.boot_timeout_s) / 2.0)
        method = _attempt_chip_reset(
            port, baud, transport,
            verify_boot=_verify_boot_during_recovery,
            short_timeout_s=4.0,
            watchdog_timeout_s=watchdog_wait,
        )
        if method is None:
            print(
                "[uart] HW reset: no boot event after exhausting all "
                "strategies and watchdog wait — recovery failed.",
                file=sys.stderr,
            )
            return None

        # 5. Reopen the capture port for the long-lived reader thread.
        #    The verifier already confirmed the chip is alive, so we just
        #    need a clean handle; the boot line itself was consumed by the
        #    verifier and is intentionally not re-injected.
        new_ser: Optional[serial.Serial] = None
        reopen_timeout = 5.0
        deadline = time.monotonic() + reopen_timeout
        while time.monotonic() < deadline:
            try:
                new_ser = serial.Serial(port, baud, timeout=0.05)
                break
            except (serial.SerialException, OSError):
                time.sleep(0.2)

        if new_ser is None:
            print(
                "[uart] HW reset: chip booted but capture port could not "
                "be reopened.",
                file=sys.stderr,
            )
            return None
        ser = new_ser

        # 6. Restart reader thread
        stop_ev.clear()
        reader = threading.Thread(
            target=_uart_reader_thread, args=(ser, lq, stop_ev), daemon=True,
        )
        reader.start()

        return method

    # ---- Main logic ----
    try:
        reader = threading.Thread(
            target=_uart_reader_thread, args=(ser, lq, stop_ev), daemon=True,
        )
        reader.start()

        if transport == "usb_serial_jtag":
            time.sleep(0.3)

        print("[uart] Waiting for boot event...", file=sys.stderr)
        if not _wait_for_boot():
            print("[ERROR] Boot timeout.", file=sys.stderr)
            stop_ev.set()
            reader.join(timeout=2.0)
            return

        print("[uart] Boot received.", file=sys.stderr)

        # ---- Golden phase ----
        if not _run_golden(is_first=True):
            print("[ERROR] Golden phase failed.", file=sys.stderr)
            stop_ev.set()
            reader.join(timeout=2.0)
            return

        # ---- Adaptive job-timeout calibration ----
        # Use the worst inter-result gap observed during the golden phase
        # as a proxy for the longest legitimate inference time, multiply
        # by the configured safety factor and clamp into [min_s, max_s].
        # Falls back to the configured maximum when the safety factor is
        # disabled (None) or when too few golden samples were collected.
        safety = serial_cfg.job_timeout_safety_factor
        if safety is not None and len(golden_result_times) >= 2:
            gaps = [
                t1 - t0
                for t0, t1 in zip(golden_result_times[:-1], golden_result_times[1:])
            ]
            max_gap_s = max(gaps)
            calibrated = max(
                float(serial_cfg.job_timeout_min_s),
                min(float(serial_cfg.job_timeout_max_s), max_gap_s * float(safety)),
            )
            print(
                f"[uart] Calibrated job timeout from golden: max gap "
                f"{max_gap_s * 1000:.1f} ms × {safety:g} = {calibrated:.2f} s "
                f"(min {serial_cfg.job_timeout_min_s:.1f} s, "
                f"max {serial_cfg.job_timeout_max_s:.1f} s).",
                file=sys.stderr,
            )
            job_timeout_s = calibrated
        else:
            print(
                f"[uart] Job timeout calibration skipped; using configured "
                f"max {job_timeout_s:.1f} s.",
                file=sys.stderr,
            )

        # ---- Fault-job dispatch loop ----
        from collections import deque

        total_jobs = fault_count * image_count
        jobs: "deque[Tuple[int, int]]" = deque(
            (fi, img) for fi in range(fault_count) for img in range(image_count)
        )
        crash_retries: Dict[Tuple[int, int], int] = {}
        # Per-fault recovery metadata captured the first time a fault
        # crashes the chip.  Membership replaces the old ``crashed_faults``
        # set; values are propagated to subsequent images of the same
        # fault index so every inferred row carries the same
        # ``recovery_method``/``boot_seen`` as the parent crash.
        crashed_fault_meta: Dict[int, Dict[str, Any]] = {}
        completed = 0
        crash_outcomes = 0

        def _make_crash_result(fi_idx: int, image_id: int) -> Dict[str, Any]:
            cr: Dict[str, Any] = {
                "event": "result",
                "phase": "fault",
                "mode": "multi_fault" if is_multi else "fault",
                "fault_model": campaign_type or "unknown",
                "status": "crash",
                "injection_status": "crash",
                "image_id": image_id,
                "pred_class": -1,
                "top1_prob": 0.0,
            }
            if is_multi:
                cr["group_id"] = group_id_map.get(fi_idx, fi_idx)
                cr["fault_count"] = 0
            else:
                cr["fault_id"] = fault_id_map.get(fi_idx, fi_idx)
            # Enrich with full fault metadata so crash rows carry the
            # same register fields as firmware-reported results.
            meta = fault_meta_map.get(fi_idx)
            if meta:
                _META_FIELD_MAP = {
                    "op_index": "operator_index",
                    "op_name": "operator_name",
                    "register_id": "register_id",
                    "register_name": "register_name",
                    "time_slot_index": "time_slot_index",
                    "time_slot_count": "time_slot_count",
                    "bit_index": "bit_index",
                }
                for csv_col, json_col in _META_FIELD_MAP.items():
                    if csv_col in meta and json_col not in cr:
                        cr[json_col] = meta[csv_col]
            return cr

        def _attach_audit(
            payload: Dict[str, Any],
            *,
            elapsed_ms: Optional[float],
            boot_seen: bool,
            result_seen: bool,
            job_ack_seen: bool,
            recovery_method: str,
            inferred: bool = False,
        ) -> Dict[str, Any]:
            """Mutate *payload* in place with the host-side audit fields.

            These fields are persisted alongside firmware-reported results
            so post-hoc analysis can distinguish *Completed*, *Crash*,
            *Hang* and recovered outcomes, and so the recovery strategy
            that brought the chip back is recorded next to the failing
            job.

            Parameters
            ----------
            inferred : bool
                ``True`` for rows that were *not* actually dispatched —
                they describe images skipped because the same fault index
                deterministically crashed on a previous image.  Their
                ``recovery_method`` and ``boot_seen`` fields are inherited
                from that parent crash; ``elapsed_ms``, ``result_seen``
                and ``job_ack_seen`` are intentionally null/false because
                no ``run_job`` was sent for these images.
            """
            payload["elapsed_ms"] = (
                None if elapsed_ms is None else round(float(elapsed_ms), 3)
            )
            payload["timeout_ms"] = int(round(job_timeout_s * 1000.0))
            payload["boot_seen"] = bool(boot_seen)
            payload["result_seen"] = bool(result_seen)
            payload["job_ack_seen"] = bool(job_ack_seen)
            payload["recovery_method"] = recovery_method
            payload["inferred"] = bool(inferred)
            return payload

        print(f"[uart] Dispatching {total_jobs} fault jobs...", file=sys.stderr)

        while jobs:
            fi, img = jobs.popleft()

            # ---- Skip images whose fault index already crashed ----
            # These are post-hoc inferred crashes (the fault deterministically
            # crashed on the first image, so we don't dispatch the rest).
            # ``recovery_method`` and ``boot_seen`` are inherited from the
            # parent crash so every row of a deterministically-crashing
            # fault carries the same recovery characterization.  The
            # ``inferred`` flag is set to True so analyses can immediately
            # tell which rows came from real measurements.
            if fi in crashed_fault_meta:
                parent = crashed_fault_meta[fi]
                cr = _make_crash_result(fi, img)
                _attach_audit(
                    cr,
                    elapsed_ms=None,
                    boot_seen=parent.get("boot_seen", False),
                    result_seen=False,
                    job_ack_seen=False,
                    recovery_method=parent.get(
                        "recovery_method", RECOVERY_METHOD_NONE
                    ),
                    inferred=True,
                )
                _append_jsonl(jsonl_path, cr)
                valid += 1
                crash_outcomes += 1
                continue

            job_start_t = time.monotonic()
            _send_cmd({"cmd": "run_job", "fi": fi, "img": img})

            got_ack = False
            boot_received = False
            pending_result = None

            while not got_ack:
                payload, err = _read_event(timeout=job_timeout_s)

                if err == "timeout":
                    print(f"[uart] Job timeout fi={fi} img={img}", file=sys.stderr)
                    break
                if err:
                    continue
                if payload is None:
                    continue

                event = payload.get("event")

                if event == "boot":
                    boot_received = True
                    break

                if event == "result":
                    # Patch golden fields from host cache when golden
                    # was skipped after a crash recovery.
                    if golden_cache and payload.get("phase") == "fault":
                        img_id = payload.get("image_id")
                        if img_id is not None and int(img_id) in golden_cache:
                            gc_pred, gc_prob = golden_cache[int(img_id)]
                            payload["golden_pred_class"] = gc_pred
                            payload["golden_top1_prob"] = gc_prob
                    # Buffer the result — only persist after job_ack to
                    # avoid writing "ok" for jobs that crash before ack.
                    pending_result = payload

                if event == "job_ack":
                    got_ack = True

                if event == "error":
                    msg = payload.get("msg", "unknown")
                    print(f"[uart] Firmware error: {msg}", file=sys.stderr)
                    got_ack = True  # skip to next job

            elapsed_ms = (time.monotonic() - job_start_t) * 1000.0
            result_seen = pending_result is not None

            # ---- Handle crash / hang ----
            if not got_ack:
                key = (fi, img)
                crash_retries[key] = crash_retries.get(key, 0) + 1

                if crash_retries[key] <= MAX_CRASH_RETRIES:
                    print(
                        f"[uart] Crash fi={fi} img={img} "
                        f"(attempt {crash_retries[key]}/{MAX_CRASH_RETRIES})",
                        file=sys.stderr,
                    )
                    jobs.appendleft((fi, img))
                    crash_row_written = False
                else:
                    # Mark this fault index as crashed — skip all remaining
                    # images for it instead of retrying each one individually.
                    # The recovery_method/boot_seen entry is filled in
                    # below once recovery completes; subsequent images
                    # of the same fault inherit those values.
                    crashed_fault_meta[fi] = {
                        "recovery_method": RECOVERY_METHOD_NONE,
                        "boot_seen": False,
                    }
                    print(
                        f"[uart] Persistent crash fi={fi} img={img} "
                        "\u2014 recording as crash for all remaining images.",
                        file=sys.stderr,
                    )
                    crash_row = _make_crash_result(fi, img)
                    crash_row_written = True
                    # We will populate audit fields below after we know
                    # which recovery method (if any) brought the chip back.

                # Track how the firmware (if at all) was brought back.
                # Default: no recovery needed/none worked — overwritten
                # below as evidence accumulates.
                recovery_method = (
                    RECOVERY_METHOD_NATURAL_BOOT if boot_received else RECOVERY_METHOD_NONE
                )

                # Wait for boot if we haven't seen it already
                if not boot_received:
                    print("[uart] Waiting for firmware reboot...", file=sys.stderr)
                    if _wait_for_boot(timeout_s=30.0):
                        boot_received = True
                        recovery_method = RECOVERY_METHOD_NATURAL_BOOT
                    else:
                        # ---- Firmware hung: attempt forced HW reset ----
                        max_hw = serial_cfg.max_hang_recoveries
                        delay = serial_cfg.hang_recovery_delay_s
                        recovered_method: Optional[str] = None
                        for attempt in range(1, max_hw + 1):
                            print(
                                f"[uart] Firmware did not reboot. "
                                f"Forcing HW reset ({attempt}/{max_hw})...",
                                file=sys.stderr,
                            )
                            if delay > 0:
                                time.sleep(delay)
                            method = _force_hw_reset_and_reconnect()
                            if method is not None:
                                print(
                                    f"[uart] HW reset succeeded via "
                                    f"{method} \u2014 device rebooted.",
                                    file=sys.stderr,
                                )
                                recovered_method = method
                                boot_received = True
                                break

                        if recovered_method is not None:
                            recovery_method = recovered_method
                        else:
                            # All reset attempts exhausted for this job.
                            # Mark the fault as fatal_hang (if not already
                            # marked as a persistent crash above) and
                            # record that no recovery method worked.
                            recovery_method = RECOVERY_METHOD_NONE
                            if fi not in crashed_fault_meta:
                                crashed_fault_meta[fi] = {
                                    "recovery_method": RECOVERY_METHOD_NONE,
                                    "boot_seen": False,
                                }
                                fhr = _make_crash_result(fi, img)
                                fhr["status"] = "fatal_hang"
                                fhr["injection_status"] = "fatal_hang"
                                _attach_audit(
                                    fhr,
                                    elapsed_ms=elapsed_ms,
                                    boot_seen=False,
                                    result_seen=result_seen,
                                    job_ack_seen=False,
                                    recovery_method=RECOVERY_METHOD_NONE,
                                )
                                _append_jsonl(jsonl_path, fhr)
                                valid += 1
                                crash_outcomes += 1
                            elif crash_row_written:
                                # We already prepared a crash row above
                                # (persistent-crash branch); upgrade it
                                # to fatal_hang since no recovery worked.
                                crash_row["status"] = "fatal_hang"
                                crash_row["injection_status"] = "fatal_hang"

                            # Remove any re-queued retries for this fault
                            jobs = deque(j for j in jobs if j[0] != fi)

                            if crash_row_written:
                                _attach_audit(
                                    crash_row,
                                    elapsed_ms=elapsed_ms,
                                    boot_seen=boot_received,
                                    result_seen=result_seen,
                                    job_ack_seen=False,
                                    recovery_method=recovery_method,
                                )
                                _append_jsonl(jsonl_path, crash_row)
                                valid += 1
                                crash_outcomes += 1
                                crash_row_written = False  # already flushed
                                # Update fault meta with the final unrecoverable state.
                                crashed_fault_meta[fi] = {
                                    "recovery_method": recovery_method,
                                    "boot_seen": boot_received,
                                }

                            if not jobs:
                                print(
                                    f"[uart] All {max_hw} HW resets failed and "
                                    "no jobs remain. Campaign ending.",
                                    file=sys.stderr,
                                )
                                break

                            # Last-ditch: one more reset attempt so we can
                            # continue with the remaining fault indices.
                            print(
                                "[uart] Attempting last-ditch HW reset "
                                "to continue campaign...",
                                file=sys.stderr,
                            )
                            if delay > 0:
                                time.sleep(delay * 2)
                            last_method = _force_hw_reset_and_reconnect()
                            if last_method is None:
                                print(
                                    "[ERROR] Cannot recover device. "
                                    "Aborting campaign.",
                                    file=sys.stderr,
                                )
                                break
                            print(
                                f"[uart] Device recovered via "
                                f"{last_method}. Continuing with "
                                "remaining faults.",
                                file=sys.stderr,
                            )

                # Flush the persistent-crash row (if not already flushed
                # by the fatal_hang upgrade path above).
                if crash_row_written:
                    _attach_audit(
                        crash_row,
                        elapsed_ms=elapsed_ms,
                        boot_seen=boot_received,
                        result_seen=result_seen,
                        job_ack_seen=False,
                        recovery_method=recovery_method,
                    )
                    _append_jsonl(jsonl_path, crash_row)
                    valid += 1
                    crash_outcomes += 1

                # Record the final recovery info for this fault so all
                # subsequent images of the same fault index inherit it
                # via the early-skip branch at the top of the loop.
                if fi in crashed_fault_meta:
                    crashed_fault_meta[fi] = {
                        "recovery_method": recovery_method,
                        "boot_seen": boot_received,
                    }

                # Skip golden after crash — host has cached golden data.
                # Firmware just needs golden_done=true to accept jobs.
                # NOTE: Register campaigns MUST re-run golden so the
                # firmware collects operator timing profiles (used by
                # compute_delay_us).  skip_golden zeroes op_profiles,
                # causing all delays to fall back to min_timer_delay_us
                # and injecting faults too early (100% masking).
                golden_ok = False
                if golden_cache and campaign_type != "register":
                    golden_ok = _skip_golden()
                    if not golden_ok:
                        print(
                            "[uart] skip_golden failed — falling back to full golden...",
                            file=sys.stderr,
                        )

                if not golden_ok:
                    print("[uart] Re-running golden after crash...", file=sys.stderr)
                    golden_ok = _run_golden(is_first=False)

                if not golden_ok:
                    print("[ERROR] Golden recovery failed after crash.", file=sys.stderr)
                    break

                continue  # next job from the deque

            # ---- Flush buffered result on successful ack ----
            if pending_result is not None:
                _attach_audit(
                    pending_result,
                    elapsed_ms=elapsed_ms,
                    boot_seen=False,
                    result_seen=True,
                    job_ack_seen=True,
                    recovery_method=RECOVERY_METHOD_NONE,
                )
                _append_jsonl(jsonl_path, pending_result)
                valid += 1
                fid = pending_result.get("fault_id", pending_result.get("group_id", "?"))
                st = pending_result.get("injection_status", "?")
                print(
                    f"[RX {valid}] fi={fi} img={img} fault_id={fid} inj={st}",
                    file=sys.stderr,
                )
                if on_result:
                    on_result(pending_result)

            # ---- Job completed successfully ----
            completed += 1
            if completed % 100 == 0 or completed == total_jobs:
                print(
                    f"[uart] Progress: {completed}/{total_jobs} jobs, "
                    f"{crash_outcomes} crash outcomes",
                    file=sys.stderr,
                )

            if stop_after is not None and valid >= stop_after:
                break

        # ---- Signal campaign end ----
        _send_cmd({"cmd": "done"})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            payload, err = _read_event(timeout=max(0.1, deadline - time.monotonic()))
            if err:
                break
            if payload and payload.get("event") == "done":
                break

        print(
            f"[uart] Campaign finished. {completed} completed, "
            f"{crash_outcomes} crash outcomes.",
            file=sys.stderr,
        )

        stop_ev.set()
        reader.join(timeout=2.0)
    finally:
        stop_ev.set()
        ser.close()

    build_csv(jsonl_path, csv_path, campaign_type=campaign_type,
              is_multi=is_multi, groups_csv=groups_csv)


