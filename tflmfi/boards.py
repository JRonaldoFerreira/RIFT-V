"""Board catalog derived from ``firmware/platformio.ini`` environments.

Each ``[env:xxx]`` section that declares a ``board`` key is exposed as a
selectable board.  Friendly names come from the PlatformIO board JSON
files in ``firmware/boards/``.
"""
from __future__ import annotations

import configparser
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BoardEnv:
    """A PlatformIO environment entry representing a selectable board."""

    env_name: str        # PIO environment name (used with ``pio run -e``)
    board: str           # PIO board id (e.g. ``esp32-c3-devkitm-1``)
    flash_size: str      # e.g. "2MB", "4MB"
    friendly_name: str   # human-readable (from board JSON or env_name)
    platform: str        # e.g. "espressif32"
    framework: str       # e.g. "espidf"
    monitor_speed: int   # e.g. 460800


# ---------------------------------------------------------------------------
# INI parser
# ---------------------------------------------------------------------------

_DEFAULT_INI = Path(__file__).resolve().parent.parent / "firmware" / "platformio.ini"


def _read_board_json_name(boards_dir: Path, board_id: str) -> Optional[str]:
    """Try to read the friendly name from a PlatformIO board JSON."""
    json_path = boards_dir / f"{board_id}.json"
    if not json_path.exists():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f).get("name")
    except (json.JSONDecodeError, OSError):
        return None


def load_boards(ini_path: Optional[Path] = None) -> List[BoardEnv]:
    """Parse ``platformio.ini`` and return all selectable board environments.

    Returns the list sorted by ``env_name``.
    """
    ini = ini_path or _DEFAULT_INI
    if not ini.exists():
        raise FileNotFoundError(f"platformio.ini not found: {ini}")

    cp = configparser.RawConfigParser()
    cp.read(str(ini), encoding="utf-8")

    # Global [env] defaults
    env_defaults = dict(cp.items("env")) if cp.has_section("env") else {}

    boards_dir = ini.parent / "boards"
    result: List[BoardEnv] = []

    for section in sorted(cp.sections()):
        if not section.startswith("env:"):
            continue
        env_name = section[4:]

        board = cp.get(section, "board", fallback=env_defaults.get("board", ""))
        if not board:
            continue

        flash_size = cp.get(
            section, "board_build.flash_size",
            fallback=cp.get(section, "board_upload.flash_size", fallback="4MB"),
        )
        platform = cp.get(
            section, "platform",
            fallback=env_defaults.get("platform", "espressif32"),
        )
        framework = cp.get(
            section, "framework",
            fallback=env_defaults.get("framework", "espidf"),
        )
        monitor_speed = int(cp.get(
            section, "monitor_speed",
            fallback=env_defaults.get("monitor_speed", "460800"),
        ))

        friendly_name = _read_board_json_name(boards_dir, board) or env_name

        result.append(BoardEnv(
            env_name=env_name,
            board=board,
            flash_size=flash_size,
            friendly_name=friendly_name,
            platform=platform,
            framework=framework,
            monitor_speed=monitor_speed,
        ))

    if not result:
        raise ValueError(f"No board environments found in {ini}")

    return result


def find_board(name: str, ini_path: Optional[Path] = None) -> BoardEnv:
    """Find a board by env_name, board id, or friendly name (case-insensitive)."""
    boards = load_boards(ini_path)
    key = name.lower().strip()
    for b in boards:
        if key in (b.env_name.lower(), b.board.lower(), b.friendly_name.lower()):
            return b
    available = ", ".join(b.env_name for b in boards)
    raise ValueError(f"Board '{name}' not found. Available: {available}")


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def select_board_interactive(ini_path: Optional[Path] = None) -> BoardEnv:
    """Present a numbered menu on stderr/stdin and return the chosen board."""
    boards = load_boards(ini_path)

    print("\n╔══════════════════════════════════════════════════╗", file=sys.stderr)
    print("║     Seleção de Placa / Ambiente PlatformIO       ║", file=sys.stderr)
    print("╚══════════════════════════════════════════════════╝\n", file=sys.stderr)

    for idx, b in enumerate(boards, start=1):
        print(
            f"  [{idx}] {b.friendly_name}\n"
            f"      Env   : {b.env_name}\n"
            f"      Board : {b.board}\n"
            f"      Flash : {b.flash_size}\n",
            file=sys.stderr,
        )

    while True:
        try:
            raw_input = input(f"Escolha o número (1-{len(boards)}): ")
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
            print(
                f"\n✔ Selecionado: {selected.friendly_name} "
                f"(env:{selected.env_name})\n",
                file=sys.stderr,
            )
            return selected

        print(f"  ⚠ Opção fora do intervalo (1-{len(boards)}).", file=sys.stderr)
