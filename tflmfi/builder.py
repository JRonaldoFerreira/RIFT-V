"""PlatformIO build and flash orchestration."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tflmfi.config import ExperimentConfig


def _pio_env_name(campaign_type: str) -> str:
    return {"register": "registerfi", "memory": "memoryfi"}.get(campaign_type, campaign_type)


def _find_project_dir(cfg: ExperimentConfig) -> Path:
    """Resolve the PlatformIO project root (parent of src/)."""
    src = cfg.firmware_src_path
    if (src.parent / "platformio.ini").exists():
        return src.parent
    if (src / "platformio.ini").exists():
        return src
    raise FileNotFoundError(
        f"Could not find platformio.ini near firmware src: {src}. "
        "Ensure firmware.src_dir points to the src/ folder inside a PlatformIO project."
    )


def _build_env(cfg: ExperimentConfig) -> dict:
    """Return a copy of os.environ with TFLMFI_EXTRA_FLAGS set for fault_duration."""
    env = os.environ.copy()
    if cfg.campaign.fault_duration == "transient":
        env["TFLMFI_EXTRA_FLAGS"] = "-DFI_FAULT_TRANSIENT=1"
    else:
        env["TFLMFI_EXTRA_FLAGS"] = ""
    return env


def build(
    cfg: ExperimentConfig,
    campaign_type: str,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``pio run -e <env>`` for the given campaign type."""
    project_dir = _find_project_dir(cfg)
    pio_env = _pio_env_name(campaign_type)
    cmd = ["pio", "run", "-e", pio_env, "-d", str(project_dir)]

    if verbose:
        cmd.append("-v")
    if extra_args:
        cmd.extend(extra_args)

    run_env = _build_env(cfg)
    print(f"[builder] Running: {' '.join(cmd)}", file=sys.stderr)
    print(f"[builder] fault_duration={cfg.campaign.fault_duration}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False, env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"PlatformIO build failed (exit {result.returncode}).")
    return result


def flash(
    cfg: ExperimentConfig,
    campaign_type: str,
    upload_port: Optional[str] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``pio run -e <env> -t upload`` for the given campaign type."""
    project_dir = _find_project_dir(cfg)
    pio_env = _pio_env_name(campaign_type)
    cmd = ["pio", "run", "-e", pio_env, "-t", "upload", "-d", str(project_dir)]

    if upload_port:
        cmd.extend(["--upload-port", upload_port])
    if verbose:
        cmd.append("-v")

    run_env = _build_env(cfg)
    print(f"[builder] Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False, env=run_env)
    if result.returncode != 0:
        raise RuntimeError(f"PlatformIO flash failed (exit {result.returncode}).")
    return result


def build_and_flash(
    cfg: ExperimentConfig,
    campaign_type: str,
    upload_port: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Build then flash in one step."""
    build(cfg, campaign_type, verbose=verbose)
    flash(cfg, campaign_type, upload_port=upload_port, verbose=verbose)
