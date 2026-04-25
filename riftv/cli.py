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
        choices=["prepare", "build", "flash", "run", "rebuild-csv", "analyze", "all"],
        help="Steps to execute (default: all). Can be repeated.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose PlatformIO output")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop UART capture after N messages")
    parser.add_argument("--results", type=Path, default=None, help="Existing results CSV for analyze step")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory (for rebuild-csv / analyze)")
    parser.add_argument("--upload-port", type=str, default=None, help="Override serial upload port")
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
        unsafe = steps & {"build", "flash", "run", "analyze", "rebuild-csv"}
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


if __name__ == "__main__":
    main()
