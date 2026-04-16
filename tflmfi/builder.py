"""PlatformIO build and flash orchestration."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tflmfi.config import ExperimentConfig


def _pio_env_name(campaign_type: str, cfg: ExperimentConfig) -> str:
    """Return the PIO environment name.

    If a board was explicitly selected (``cfg.firmware.pio_env`` is set),
    that environment takes priority.  Otherwise fall back to the
    campaign-type mapping (register → registerfi, memory → memoryfi).
    """
    if cfg.firmware.pio_env:
        return cfg.firmware.pio_env
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


def _ensure_clean_cmake_cache(project_dir: Path, pio_env: str) -> None:
    """Remove stale CMake cache when managed_components is missing.

    If *managed_components/* was deleted (e.g. to force a fresh component
    download) but *.pio/build/* still holds a cached CMake configuration that
    references those paths, PlatformIO will skip component resolution and the
    build will fail.  Cleaning the CMake artefacts forces a full
    reconfiguration on the next ``pio run``.
    """
    managed = project_dir / "managed_components"
    if managed.exists():
        return
    env_build = project_dir / ".pio" / "build" / pio_env
    if not env_build.exists():
        return
    cmake_cache = env_build / "CMakeCache.txt"
    cmake_files = env_build / "CMakeFiles"
    if cmake_cache.exists() or cmake_files.exists():
        if cmake_cache.exists():
            cmake_cache.unlink()
        if cmake_files.exists():
            shutil.rmtree(cmake_files)
        print(
            "[builder] Cleaned stale CMake cache "
            "(managed_components missing, forcing re-resolution).",
            file=sys.stderr,
        )


def build(
    cfg: ExperimentConfig,
    campaign_type: str,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``pio run -e <env>`` for the given campaign type."""
    project_dir = _find_project_dir(cfg)
    pio_env = _pio_env_name(campaign_type, cfg)
    _ensure_clean_cmake_cache(project_dir, pio_env)
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
    pio_env = _pio_env_name(campaign_type, cfg)
    cmd = ["pio", "run", "-e", pio_env, "-t", "upload", "-d", str(project_dir)]

    if upload_port:
        cmd.extend(["--upload-port", upload_port])
    if verbose:
        cmd.append("-v")

    run_env = _build_env(cfg)
    print(f"[builder] Running: {' '.join(cmd)}", file=sys.stderr)

    flash_warnings: List[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=run_env,
    )
    for line in proc.stdout:
        sys.stderr.write(line)
        low = line.lower()
        if "flash memory size mismatch" in low:
            flash_warnings.append(line.strip())

    proc.wait()

    if flash_warnings:
        print(f"\n[WARNING] {'='*60}", file=sys.stderr)
        print("[WARNING] PlatformIO detected flash memory issues:", file=sys.stderr)
        for w in flash_warnings:
            print(f"[WARNING]   {w}", file=sys.stderr)
        print(
            "[WARNING] This may prevent the device from booting correctly.\n"
            "[WARNING] Adjust flash size in sdkconfig.defaults or via 'pio run -t menuconfig'.",
            file=sys.stderr,
        )
        print(f"[WARNING] {'='*60}\n", file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"PlatformIO flash failed (exit {proc.returncode}).")
    return subprocess.CompletedProcess(cmd, proc.returncode)


def build_and_flash(
    cfg: ExperimentConfig,
    campaign_type: str,
    upload_port: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Build then flash in one step."""
    build(cfg, campaign_type, verbose=verbose)
    flash(cfg, campaign_type, upload_port=upload_port, verbose=verbose)
