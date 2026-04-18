"""PlatformIO build and flash orchestration.

The builder generates a *per-context* ``platformio.ini`` inside
``firmware/.tflmfi/`` and points PlatformIO at it with ``-c``.  This avoids
relying on the static ``firmware/platformio.ini`` and ensures that the board,
platform, flash_size and sdkconfig settings declared in the YAML actually
reach PlatformIO.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tflmfi.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _context_fingerprint(cfg: ExperimentConfig, pio_env: str) -> str:
    """Return a short hash of board+flash_size+env+defaults to detect changes."""
    blob = json.dumps({
        "board": cfg.firmware.board,
        "platform": cfg.firmware.platform,
        "framework": cfg.firmware.framework,
        "flash_size": cfg.firmware.flash_size,
        "idf_target": cfg.firmware.idf_target,
        "monitor_speed": cfg.firmware.monitor_speed,
        "sdkconfig_defaults": cfg.firmware.sdkconfig_defaults,
        "serial_transport": cfg.serial.transport,
        "pio_env": pio_env,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def _clean_stale_cache(project_dir: Path, pio_env: str, new_fp: str) -> None:
    """Wipe PlatformIO build cache when the context (board/flash/env) changed."""
    tflmfi_dir = project_dir / ".tflmfi"
    fp_file = tflmfi_dir / f".fingerprint.{pio_env}"

    old_fp = fp_file.read_text(encoding="utf-8").strip() if fp_file.exists() else ""
    if old_fp == new_fp:
        return  # nothing changed

    # Context changed — purge stale artefacts
    build_dir = project_dir / ".pio" / "build" / pio_env
    if build_dir.exists():
        print(f"[builder] Context changed — removing stale build cache: {build_dir}", file=sys.stderr)
        shutil.rmtree(build_dir, ignore_errors=True)

    # Remove old generated sdkconfig files
    for old in project_dir.glob("sdkconfig.*"):
        if old.name.startswith("sdkconfig.defaults"):
            continue  # keep the base defaults
        if old.name in ("sdkconfig.boards", "sdkconfig.campaigns"):
            continue  # keep directories
        if old.is_file():
            print(f"[builder] Removing stale sdkconfig: {old.name}", file=sys.stderr)
            old.unlink(missing_ok=True)

    # Remove old generated ini files
    for old_ini in tflmfi_dir.glob("platformio.*.ini"):
        old_ini.unlink(missing_ok=True)

    tflmfi_dir.mkdir(parents=True, exist_ok=True)
    fp_file.write_text(new_fp, encoding="utf-8")


# ---------------------------------------------------------------------------
# Generated platformio.ini
# ---------------------------------------------------------------------------

def _generate_ini(
    cfg: ExperimentConfig,
    project_dir: Path,
    pio_env: str,
) -> Path:
    """Write a context-specific ``platformio.ini`` under ``.tflmfi/``."""
    tflmfi_dir = project_dir / ".tflmfi"
    tflmfi_dir.mkdir(parents=True, exist_ok=True)

    ini_path = tflmfi_dir / f"platformio.{pio_env}.{cfg.firmware.board}.ini"

    flags: List[str] = ["-DTF_LITE_STATIC_MEMORY"]
    flags.extend(cfg.firmware.extra_build_flags)
    if cfg.campaign.fault_duration == "transient":
        flags.append("-DFI_FAULT_TRANSIENT=1")

    # Campaign-specific flag
    if pio_env == "registerfi":
        flags.append("-DFI_CAMPAIGN_REGISTER=1")
    elif pio_env == "memoryfi":
        flags.append("-DFI_CAMPAIGN_MEMORY=1")

    # Protocol UART baud rate — used by firmware to configure UART0 directly
    flags.append(f"-DFI_PROTOCOL_BAUD={cfg.firmware.monitor_speed}")

    # Serial transport — tells firmware whether to use UART0 or USB Serial/JTAG
    if cfg.serial.transport == "usb_serial_jtag":
        flags.append("-DFI_TRANSPORT_USB_SERIAL_JTAG=1")

    flags_str = "\n    ".join(flags)

    # Collect sdkconfig_defaults chain
    sdk_defaults: List[str] = []
    base_defaults = project_dir / "sdkconfig.defaults"
    if base_defaults.exists():
        sdk_defaults.append(str(base_defaults))
    board_defaults = project_dir / "sdkconfig.boards" / f"{cfg.firmware.board}.defaults"
    if board_defaults.exists():
        sdk_defaults.append(str(board_defaults))
    campaign_defaults = project_dir / "sdkconfig.campaigns" / f"{pio_env}.defaults"
    if campaign_defaults.exists():
        sdk_defaults.append(str(campaign_defaults))
    for extra in cfg.firmware.sdkconfig_defaults:
        p = Path(extra)
        if not p.is_absolute():
            p = project_dir / p
        if p.exists():
            sdk_defaults.append(str(p))

    # Runtime sdkconfig fragment (baud + flash_size)
    runtime_frag = _generate_sdkconfig_runtime(cfg, project_dir, pio_env)
    sdk_defaults.append(str(runtime_frag))

    # Build the ini content
    lines = [
        f"; Auto-generated by tflmfi builder — do not edit",
        f"; Context: env={pio_env}  board={cfg.firmware.board}",
        f"",
        f"[platformio]",
        f"src_dir = {project_dir / 'src'}",
        f"lib_dir = {project_dir / 'lib'}",
        f"include_dir = {project_dir / 'include'}",
        f"",
        f"[env:{pio_env}]",
        f"platform = {cfg.firmware.platform}",
        f"board = {cfg.firmware.board}",
        f"framework = {cfg.firmware.framework}",
        f"monitor_speed = {cfg.firmware.monitor_speed}",
        f"build_flags =",
        f"    {flags_str}",
    ]

    # flash_size override
    if cfg.firmware.flash_size:
        lines.append(f"board_upload.flash_size = {cfg.firmware.flash_size}")

    # idf_target override
    if cfg.firmware.idf_target:
        lines.append(f"board_build.idf_target = {cfg.firmware.idf_target}")

    # sdkconfig_defaults chain
    if sdk_defaults:
        lines.append(f"board_build.sdkconfig_defaults = {';'.join(sdk_defaults)}")

    ini_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ini_path


def _generate_sdkconfig_runtime(
    cfg: ExperimentConfig,
    project_dir: Path,
    pio_env: str,
) -> Path:
    """Write a runtime sdkconfig fragment with baud and flash_size."""
    tflmfi_dir = project_dir / ".tflmfi"
    tflmfi_dir.mkdir(parents=True, exist_ok=True)
    frag = tflmfi_dir / f"sdkconfig.runtime.{pio_env}"

    content_lines = [
        f"# Auto-generated by tflmfi builder",
    ]

    # Console / transport config
    if cfg.serial.transport == "usb_serial_jtag":
        content_lines.append("CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y")
    else:
        content_lines.append(f"CONFIG_ESP_CONSOLE_UART_BAUDRATE={cfg.firmware.monitor_speed}")
    if cfg.firmware.flash_size:
        content_lines.append(f"CONFIG_ESPTOOLPY_FLASHSIZE=\"{cfg.firmware.flash_size}\"")
        # Also set the matching boolean, e.g. CONFIG_ESPTOOLPY_FLASHSIZE_2MB=y
        size_tag = cfg.firmware.flash_size.upper().replace(" ", "")
        content_lines.append(f"CONFIG_ESPTOOLPY_FLASHSIZE_{size_tag}=y")

    frag.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
    return frag


# ---------------------------------------------------------------------------
# Build / Flash
# ---------------------------------------------------------------------------

def build(
    cfg: ExperimentConfig,
    campaign_type: str,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run ``pio run -e <env> -c <generated_ini>`` for the given campaign type."""
    project_dir = _find_project_dir(cfg)
    pio_env = _pio_env_name(campaign_type)

    # Cache cleanup on context change
    fp = _context_fingerprint(cfg, pio_env)
    _clean_stale_cache(project_dir, pio_env, fp)

    # Generate context-specific ini + sdkconfig
    ini_path = _generate_ini(cfg, project_dir, pio_env)

    cmd = ["pio", "run", "-e", pio_env, "-c", str(ini_path), "-d", str(project_dir)]

    if verbose:
        cmd.append("-v")
    if extra_args:
        cmd.extend(extra_args)

    print(f"[builder] Running: {' '.join(cmd)}", file=sys.stderr)
    print(f"[builder] board={cfg.firmware.board}  platform={cfg.firmware.platform}  "
          f"flash_size={cfg.firmware.flash_size or 'board-default'}  "
          f"fault_duration={cfg.campaign.fault_duration}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"PlatformIO build failed (exit {result.returncode}).")

    return result


def flash(
    cfg: ExperimentConfig,
    campaign_type: str,
    upload_port: Optional[str] = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``pio run -e <env> -t upload -c <generated_ini>``."""
    project_dir = _find_project_dir(cfg)
    pio_env = _pio_env_name(campaign_type)

    # Ensure ini exists (build should have been called first, but be safe)
    ini_path = _generate_ini(cfg, project_dir, pio_env)

    cmd = ["pio", "run", "-e", pio_env, "-t", "upload",
           "-c", str(ini_path), "-d", str(project_dir)]

    if upload_port:
        cmd.extend(["--upload-port", upload_port])
    if verbose:
        cmd.append("-v")

    print(f"[builder] Running: {' '.join(cmd)}", file=sys.stderr)
    print(f"[builder] board={cfg.firmware.board}  platform={cfg.firmware.platform}  "
          f"flash_size={cfg.firmware.flash_size or 'board-default'}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False)
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
