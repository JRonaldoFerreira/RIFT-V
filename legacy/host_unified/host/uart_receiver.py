#!/usr/bin/env python3
"""Receive JSON Lines from ESP32-C3 via UART and persist to JSONL + CSV."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import serial


CSV_COLUMNS = [
    "event",
    "phase",
    "mode",
    "fault_model",
    "fault_id",
    "image_id",
    "label",
    "pred_class",
    "top1_prob",
    "golden_pred_class",
    "golden_top1_prob",
    "operator_index",
    "operator_name",
    "tensor_index",
    "byte_in_tensor",
    "register_id",
    "register_name",
    "time_slot_index",
    "time_slot_count",
    "delay_us",
    "bit_index",
    "status",
    "injection_status",
    "mean_duration_us",
    "samples",
    "uart_line",
]

_MAX_LINE_BYTES = 1_048_576

_PANIC_MARKERS = (
    "guru meditation error",
    "panic'ed",
    "stack protection fault",
)

_SYSTEM_WARNING_MARKERS = (
    "task_wdt",
    "task watchdog",
    "register dump",
    "backtrace:",
    "abort()",
    "stack smashing",
)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _uart_line_reader(
    ser: serial.Serial,
    line_queue: "queue.Queue[str]",
    stop: threading.Event,
) -> None:
    buf = bytearray()
    while not stop.is_set():
        try:
            waiting = ser.in_waiting
            if waiting:
                chunk = ser.read(waiting)
            else:
                chunk = ser.read(1)
        except serial.SerialException:
            break

        if not chunk:
            continue

        buf.extend(chunk)
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break

            raw_line = bytes(buf[:i])
            del buf[: i + 1]

            if len(raw_line) > _MAX_LINE_BYTES:
                print(f"[WARN] Dropping oversized line ({len(raw_line)} bytes).", file=sys.stderr)
                continue

            line = raw_line.decode("utf-8", errors="replace").strip()
            if line:
                line_queue.put(line)


def _build_csv(jsonl_path: Path, csv_path: Path) -> None:
    if not jsonl_path.exists():
        print("[WARN] No JSONL found; skipping CSV generation.", file=sys.stderr)
        return

    data_rows: List[Dict[str, Any]] = []
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

            ev = str(entry.get("event", ""))
            if ev.startswith("_"):
                markers += 1
                continue

            data_rows.append(entry)

    if not data_rows:
        print("[WARN] JSONL has no data rows; skipping CSV generation.", file=sys.stderr)
        return

    df = pd.DataFrame(data_rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df[CSV_COLUMNS].to_csv(csv_path, index=False)

    results = df[df["event"] == "result"]
    ok_rows = len(results[results["status"] == "ok"]) if len(results) else 0
    crashes = len(results[results["status"] == "crash"]) if len(results) else 0

    print(f"[SUMMARY] Total data rows: {len(df)} ({markers} markers skipped)", file=sys.stderr)
    print(f"[SUMMARY] Results: {len(results)} total, {ok_rows} ok, {crashes} crashes", file=sys.stderr)
    print(f"[SUMMARY] JSONL -> {jsonl_path}", file=sys.stderr)
    print(f"[SUMMARY] CSV   -> {csv_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Path for JSONL stream. Defaults to <csv-out-stem>.jsonl.",
    )
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after N valid data messages.")
    args = parser.parse_args()

    if args.jsonl_out is None:
        args.jsonl_out = args.csv_out.with_suffix(".jsonl")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)

    valid_messages = 0
    line_queue: queue.Queue[str] = queue.Queue()
    stop_reader = threading.Event()

    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        print(f"Listening on {args.port} @ {args.baud} baud...", file=sys.stderr)

        ser.dtr = False
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.1)
        ser.reset_input_buffer()

        reader = threading.Thread(
            target=_uart_line_reader,
            args=(ser, line_queue, stop_reader),
            daemon=True,
        )
        reader.start()

        print("ESP32 reset - waiting for boot event...", file=sys.stderr)
        while True:
            try:
                line = line_queue.get(timeout=30.0)
            except queue.Empty:
                print("[ERROR] Timeout waiting for boot line.", file=sys.stderr)
                stop_reader.set()
                return

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Non-JSON line (boot): {line[:200]}...", file=sys.stderr)
                continue

            if payload.get("event") == "boot":
                print("Boot event received - campaign starting.", file=sys.stderr)
                break

        while True:
            line = line_queue.get()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                low = line.lower()
                is_panic = any(m in low for m in _PANIC_MARKERS)
                is_warning = any(m in low for m in _SYSTEM_WARNING_MARKERS)
                if is_panic or is_warning:
                    _append_jsonl(
                        args.jsonl_out,
                        {
                            "event": "_system_warning",
                            "warning_type": "panic" if is_panic else "system_warning",
                            "uart_line": line[:500],
                        },
                    )
                else:
                    print(f"[WARN] Non-JSON line: {line[:200]}", file=sys.stderr)
                continue

            event = payload.get("event")

            if event == "progress":
                phase = payload.get("phase", "?")
                cur = payload.get("current", 0)
                tot = payload.get("total", 0)
                pct = (cur / tot * 100) if tot else 0
                print(f"[PROGRESS] {phase}: {cur}/{tot} ({pct:.1f}%)", file=sys.stderr)
                continue

            if event == "done":
                print("Campaign finished.", file=sys.stderr)
                break

            _append_jsonl(args.jsonl_out, payload)
            valid_messages += 1

            if event == "result":
                phase = payload.get("phase", payload.get("mode", "?"))
                fault_model = payload.get("fault_model", "?")
                fault_id = payload.get("fault_id", "?")
                image_id = payload.get("image_id", "?")
                status = payload.get("status", "?")
                injection_status = payload.get("injection_status", "?")
                print(
                    f"[RX {valid_messages}] result phase={phase} model={fault_model} "
                    f"fault_id={fault_id} image_id={image_id} status={status} injection_status={injection_status}",
                    file=sys.stderr,
                )
            else:
                print(f"[RX {valid_messages}] {payload}", file=sys.stderr)

            if args.stop_after is not None and valid_messages >= args.stop_after:
                break

        stop_reader.set()
        reader.join(timeout=2.0)

    _build_csv(args.jsonl_out, args.csv_out)


if __name__ == "__main__":
    main()
