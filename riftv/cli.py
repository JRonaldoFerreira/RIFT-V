#!/usr/bin/env python3
"""CLI entry point for RIFT-V experiments.

Usage examples:

    # Full pipeline (prepare → build → flash → run → analyze):
    riftv configs/registerfi_example.yaml

    # Prepare only (generate population + headers):
    riftv configs/memoryfi_example.yaml --step prepare

    # Build + flash only (headers must already be installed):
    riftv configs/registerfi_example.yaml --step build --step flash

    # Capture only (firmware must already be flashed):
    riftv configs/registerfi_example.yaml --step run

    # Rebuild CSV from JSONL (e.g. after column schema change):
    riftv configs/memoryfi_example.yaml --step rebuild-csv --run-dir runs/my_run

    # Analyze existing results:
    riftv configs/registerfi_example.yaml --step analyze --results runs/my_run/results.csv

    # Generate a microFI-comparable overhead report from a finished run:
    riftv configs/registerfi_example.yaml --step overhead --run-dir runs/my_run

    # Trigger a microFI-style differential calibration on the flashed
    # board (does not require a full campaign):
    riftv configs/registerfi_example.yaml --step measure-overhead \\
          --overhead-fault 1 --overhead-image 0 --overhead-reps 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from riftv.config import load_config
from riftv.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="riftv",
        description="RIFT-V — Runtime-aware statistical fault injection for TinyML on RISC-V",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    parser.add_argument(
        "--step",
        action="append",
        choices=[
            "prepare", "build", "flash", "run", "rebuild-csv",
            "analyze", "overhead", "measure-overhead", "all",
        ],
        help="Steps to execute (default: all). Can be repeated.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose PlatformIO output")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop UART capture after N messages")
    parser.add_argument("--results", type=Path, default=None, help="Existing results CSV for analyze step")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory (for rebuild-csv / analyze)")
    parser.add_argument("--upload-port", type=str, default=None, help="Override serial upload port")
    parser.add_argument(
        "--overhead-fault", type=int, default=0,
        help="Fault index to use for the differential overhead calibration (default 0).",
    )
    parser.add_argument(
        "--overhead-image", type=int, default=0,
        help="Image index to use for the differential overhead calibration (default 0).",
    )
    parser.add_argument(
        "--overhead-reps", type=int, default=100,
        help="Number of repetitions per phase in the differential overhead calibration (default 100).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = Experiment(cfg)

    steps = set(args.step) if args.step else {"all"}

    if "all" in steps:
        exp.execute(verbose=args.verbose, stop_after=args.stop_after)
        return

    # Hybrid campaigns require the orchestrated pipeline in
    # Experiment._execute_hybrid (build → flash → run for register, then
    # for memory).  Running flash/run/analyze individually only exercises
    # the register sub-campaign and silently drops the memory side, so we
    # block those steps here and steer the user to the full pipeline.
    if cfg.campaign.type == "hybrid":
        unsafe = steps & {
            "build", "flash", "run", "analyze", "rebuild-csv",
            "overhead", "measure-overhead",
        }
        if unsafe:
            print(
                "[riftv] Hybrid campaigns must be driven through the full "
                "pipeline (sequential register + memory builds, flashes, "
                "captures and classifications). Individual --step "
                f"{sorted(unsafe)} would only execute the register half.\n"
                "        Run without --step:\n"
                f"            riftv {args.config}\n"
                "        Or use --step prepare to (re)generate headers only.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    if "prepare" in steps:
        exp.prepare()

    if "build" in steps:
        exp.build(verbose=args.verbose)

    if "flash" in steps:
        exp.flash(upload_port=args.upload_port, verbose=args.verbose)

    if "run" in steps:
        exp.run(stop_after=args.stop_after)

    if "rebuild-csv" in steps:
        exp.rebuild_csv(run_dir=args.run_dir)

    if "analyze" in steps:
        run_dir = args.run_dir
        results = args.results
        if run_dir and not results:
            results = run_dir / "results.csv"
        exp.analyze(results_csv=results)

    if "overhead" in steps:
        from riftv.overhead import load_overhead_report

        run_dir = args.run_dir or exp.out_dir
        jsonl = run_dir / "results.jsonl"
        report = load_overhead_report(jsonl)
        print(report.format())
        csv_out = run_dir / "overhead.csv"
        report.to_csv(csv_out)
        print(f"\n[overhead] CSV written → {csv_out}", file=sys.stderr)

    if "measure-overhead" in steps:
        _run_measure_overhead(
            cfg,
            fault_idx=args.overhead_fault,
            image_id=args.overhead_image,
            reps=args.overhead_reps,
            run_dir=args.run_dir or exp.out_dir,
        )


def _run_measure_overhead(cfg, fault_idx: int, image_id: int, reps: int, run_dir: Path) -> None:
    """Send a measure_overhead command to the flashed firmware and
    persist the resulting events.

    Assumes the firmware is already flashed for the active campaign
    type.  Use ``riftv config --step flash`` first if needed.
    """
    import json
    import sys
    import time
    import serial

    from riftv.overhead import load_overhead_report

    port = cfg.serial.port
    baud = cfg.serial.baud
    print(
        f"[measure-overhead] Opening {port} @ {baud} baud, "
        f"fault_idx={fault_idx} image_id={image_id} reps={reps}",
        file=sys.stderr,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "overhead_calibration.jsonl"

    ser = serial.Serial(port, baud, timeout=2.0)
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "boot":
                break

        cmd = {"cmd": "skip_golden"}
        ser.write((json.dumps(cmd) + "\n").encode("utf-8"))

        cmd = {"cmd": "measure_overhead", "fi": fault_idx,
               "img": image_id, "reps": reps}
        ser.write((json.dumps(cmd) + "\n").encode("utf-8"))
        print(f"[measure-overhead] Sent {cmd}", file=sys.stderr)

        deadline = time.monotonic() + 600.0
        with jsonl_path.open("a", encoding="utf-8") as fh:
            seen_summary = False
            seen_calibration = False
            while time.monotonic() < deadline:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                evt = payload.get("event")
                if evt in ("overhead_summary", "overhead_calibration"):
                    fh.write(json.dumps(payload) + "\n")
                    fh.flush()
                    print(
                        f"[measure-overhead] {evt}: "
                        f"{payload.get('cycles_per_injection', '?')} cycles "
                        f"({payload.get('us_per_injection', '?')} µs)",
                        file=sys.stderr,
                    )
                    if evt == "overhead_summary":
                        seen_summary = True
                    if evt == "overhead_calibration":
                        seen_calibration = True
                    if seen_calibration and seen_summary:
                        break
    finally:
        ser.close()

    report = load_overhead_report(jsonl_path)
    print("\n" + report.format())
    csv_out = run_dir / "overhead.csv"
    report.to_csv(csv_out)
    print(f"\n[measure-overhead] CSV written → {csv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
