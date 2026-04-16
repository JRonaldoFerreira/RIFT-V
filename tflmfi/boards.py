"""ESP32 board catalog — loader, selector, and interactive menu.

Board definitions live as individual YAML files under ``configs/esp32/``.
This module scans that directory, parses each file, and offers an
interactive terminal menu so users without the PlatformIO IDE can
pick the target board before a build/flash cycle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Flash-size helpers
# ---------------------------------------------------------------------------

_VALID_FLASH_SIZES = ("2MB", "4MB", "8MB", "16MB", "32MB")

_FLASH_SIZE_BYTES = {
    "2MB":  2_000_000,
    "4MB":  4_000_000,
    "8MB":  8_000_000,
    "16MB": 16_000_000,
    "32MB": 32_000_000,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BoardSpec:
    """Hardware specification for a single ESP32 board.

    ``flash_size`` is the single source of truth for flash capacity
    (e.g. ``"2MB"``, ``"4MB"``).  All PlatformIO and sdkconfig values
    are derived automatically — see the ``pio_overrides`` and
    ``sdkconfig_overrides`` properties.
    """

    name: str
    description: str
    arch: str
    core: str
    pio_board: str
    platform: str
    framework: str
    flash_size: str  # "2MB", "4MB", "8MB", "16MB", "32MB"
    sram_kb: int
    monitor_speed: int
    usb_vid_pid: str = ""
    extra_build_flags: List[str] = field(default_factory=list)
    # path to the YAML source (for traceability)
    _source: Path = field(default=Path("."), repr=False)

    # ---- derived properties -----------------------------------------------

    @property
    def flash_size_bytes(self) -> int:
        return _FLASH_SIZE_BYTES[self.flash_size]

    @property
    def pio_overrides(self) -> Dict[str, str]:
        return {
            "board_upload.flash_size": self.flash_size,
            "board_build.flash_size": self.flash_size,
        }

    @property
    def sdkconfig_overrides(self) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        for sz in _VALID_FLASH_SIZES:
            key = f"CONFIG_ESPTOOLPY_FLASHSIZE_{sz}"
            overrides[key] = "y" if sz == self.flash_size else ""
        return overrides


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_BOARDS_DIR = Path(__file__).resolve().parent.parent / "configs" / "esp32"


def _parse_board_yaml(path: Path) -> BoardSpec:
    """Parse a single board YAML and return a ``BoardSpec``."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "board" not in raw:
        raise ValueError(f"Board file {path} must contain a top-level 'board' mapping.")

    b: Dict[str, Any] = raw["board"]

    flash_size = b.get("flash_size", "4MB")
    if flash_size not in _VALID_FLASH_SIZES:
        raise ValueError(
            f"Board {path.name}: flash_size must be one of "
            f"{_VALID_FLASH_SIZES}, got '{flash_size}'"
        )

    return BoardSpec(
        name=b["name"],
        description=b.get("description", ""),
        arch=b.get("arch", "riscv"),
        core=b.get("core", "rv32imc"),
        pio_board=b["pio_board"],
        platform=b.get("platform", "espressif32"),
        framework=b.get("framework", "espidf"),
        flash_size=flash_size,
        sram_kb=int(b.get("sram_kb", 400)),
        monitor_speed=int(b.get("monitor_speed", 460800)),
        usb_vid_pid=b.get("usb_vid_pid", ""),
        extra_build_flags=b.get("extra_build_flags", []),
        _source=path,
    )


def load_boards(boards_dir: Optional[Path] = None) -> List[BoardSpec]:
    """Load all board YAML files from *boards_dir* (defaults to ``configs/esp32/``).

    Returns the list sorted alphabetically by ``name``.
    """
    d = boards_dir or _DEFAULT_BOARDS_DIR
    if not d.is_dir():
        raise FileNotFoundError(
            f"Boards directory not found: {d}\n"
            "Create configs/esp32/ with at least one board YAML file."
        )

    boards: List[BoardSpec] = []
    for p in sorted(d.glob("*.yaml")):
        try:
            boards.append(_parse_board_yaml(p))
        except Exception as exc:
            print(f"[boards] WARNING: skipping {p.name}: {exc}", file=sys.stderr)

    if not boards:
        raise FileNotFoundError(
            f"No valid board YAML files found in {d}.\n"
            "Add at least one .yaml file following the board schema."
        )

    return boards


def find_board(name_or_pio: str, boards_dir: Optional[Path] = None) -> BoardSpec:
    """Find a board by its ``name`` or ``pio_board`` identifier.

    The match is case-insensitive and also accepts the YAML filename stem
    (e.g. ``esp32_c3_super_mini``).
    """
    boards = load_boards(boards_dir)
    key = name_or_pio.lower().strip()
    for b in boards:
        if (
            key == b.name.lower()
            or key == b.pio_board.lower()
            or key == b._source.stem.lower()
        ):
            return b
    available = ", ".join(b.name for b in boards)
    raise ValueError(
        f"Board '{name_or_pio}' not found. Available boards: {available}"
    )


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def select_board_interactive(boards_dir: Optional[Path] = None) -> BoardSpec:
    """Present a numbered menu on stderr/stdin and return the chosen board."""
    boards = load_boards(boards_dir)

    print("\n╔══════════════════════════════════════════════════╗", file=sys.stderr)
    print("║     Seleção de Placa ESP32 (RISC-V)             ║", file=sys.stderr)
    print("╚══════════════════════════════════════════════════╝\n", file=sys.stderr)

    for idx, b in enumerate(boards, start=1):
        print(
            f"  [{idx}] {b.name}\n"
            f"      PIO board : {b.pio_board}\n"
            f"      Flash     : {b.flash_size}  |  SRAM: {b.sram_kb} KB\n"
            f"      {b.description}\n",
            file=sys.stderr,
        )

    while True:
        try:
            raw_input = input("Escolha o número da placa (1-{}): ".format(len(boards)))
        except (EOFError, KeyboardInterrupt):
            print("\nAbortado pelo usuário.", file=sys.stderr)
            sys.exit(1)

        try:
            choice = int(raw_input.strip())
        except ValueError:
            print("  ⚠ Digite um número válido.", file=sys.stderr)
            continue

        if 1 <= choice <= len(boards):
            selected = boards[choice - 1]
            print(f"\n✔ Placa selecionada: {selected.name}\n", file=sys.stderr)
            return selected

        print(f"  ⚠ Opção fora do intervalo (1-{len(boards)}).", file=sys.stderr)
