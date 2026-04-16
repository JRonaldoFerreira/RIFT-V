#!/usr/bin/env python3
"""CLI entry point for tflmfi experiments.

Usage examples:

    # Full pipeline (prepare → build → flash → run → analyze):
    tflmfi configs/registerfi_example.yaml

    # Prepare only (generate population + headers):
    tflmfi configs/memoryfi_example.yaml --step prepare

    # Build + flash only (headers must already be installed):
    tflmfi configs/registerfi_example.yaml --step build --step flash

    # Capture only (firmware must already be flashed):
    tflmfi configs/registerfi_example.yaml --step run

    # Analyze existing results:
    tflmfi configs/registerfi_example.yaml --step analyze --results runs/my_run/results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tflmfi.boards import find_board, select_board_interactive
from tflmfi.config import load_config
from tflmfi.experiment import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tflmfi — Unified FI experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", type=Path, help="Path to experiment YAML config")
    parser.add_argument(
        "--step",
        action="append",
        choices=["prepare", "build", "flash", "run", "analyze", "all"],
        help="Steps to execute (default: all). Can be repeated.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose PlatformIO output")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop UART capture after N messages")
    parser.add_argument("--results", type=Path, default=None, help="Existing results CSV for analyze step")
    parser.add_argument("--upload-port", type=str, default=None, help="Override serial upload port")
    parser.add_argument(
        "--board", type=str, default=None,
        help="Board name, PIO board id, or YAML stem from configs/esp32/ "
             "(e.g. 'esp32_c3_super_mini'). Overrides firmware.board in the config.",
    )
    parser.add_argument(
        "--select-board", action="store_true",
        help="Show interactive board selection menu before running.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Board selection ──────────────────────────────────────────────
    if args.board:
        board = find_board(args.board)
        cfg.apply_board(board)
    elif args.select_board:
        board = select_board_interactive()
        cfg.apply_board(board)

    exp = Experiment(cfg)

    steps = set(args.step) if args.step else {"all"}

    if "all" in steps:
        exp.execute(verbose=args.verbose, stop_after=args.stop_after)
        return

    if "prepare" in steps:
        exp.prepare()

    if "build" in steps:
        exp.build(verbose=args.verbose)

    if "flash" in steps:
        exp.flash(upload_port=args.upload_port, verbose=args.verbose)

    if "run" in steps:
        exp.run(stop_after=args.stop_after)

    if "analyze" in steps:
        exp.analyze(results_csv=args.results)


if __name__ == "__main__":
    main()
