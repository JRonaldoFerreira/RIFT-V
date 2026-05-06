"""Out-of-band reset supervisor — Raspberry Pi Pico controlling EN/RST.

The Pico runs a tiny Arduino sketch that listens for ``RESET``, ``PING``
and ``STATUS`` commands over its own USB-Serial link.  When the host
detects a fatal hang on the ESP32 target, it sends ``RESET`` to the
Pico, which pulls EN/RST low for a configurable duration (open-drain by
software), forcing a hard reset independently of the target's USB stack,
firmware state or watchdog configuration.

This is an *out-of-band* recovery channel: the Pico's USB connection is
entirely separate from the ESP32's, so it remains reachable even when
the target's ``usb_serial_jtag`` peripheral has disappeared.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import serial


class PicoSupervisor:
    """Control a Raspberry Pi Pico wired to the target's EN/RST line and
    (optionally) a MOSFET that switches the target's USB 5 V supply.

    Protocol (newline-terminated, 115200 baud)::

        Host → Pico              Pico → Host
        ────────────────         ──────────────────────
        PING\\n                   OK PONG\\n
        STATUS\\n                 OK STATUS READY\\n
        RESET\\n                  OK RESET_DONE\\n
        RESET 500\\n              OK RESET_DONE 500\\n
        POWER_CYCLE\\n            OK POWER_CYCLE_DONE\\n
        POWER_CYCLE 5000\\n       OK POWER_CYCLE_DONE 5000\\n

    Any response that does not start with ``OK`` is treated as an error.
    """

    DEFAULT_BAUD = 115200
    DEFAULT_TIMEOUT = 2.0

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.port = port
        self._ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        # Give the Pico time to reset after DTR assertion on connect.
        time.sleep(2.0)
        # Drain any leftover boot messages (e.g. ``SUPERVISOR_READY``).
        self._ser.reset_input_buffer()
        # Verify connectivity.
        self.ping()
        print(
            f"[supervisor] Pico supervisor ready on {port}",
            file=sys.stderr,
        )

    # ---- Low-level I/O --------------------------------------------------

    def _cmd(self, cmd: str, timeout: Optional[float] = None) -> str:
        """Send *cmd*, read one response line, assert it starts with ``OK``."""
        old_timeout = self._ser.timeout
        if timeout is not None:
            self._ser.timeout = timeout
        try:
            self._ser.reset_input_buffer()
            self._ser.write((cmd.strip() + "\n").encode())
            self._ser.flush()
            raw = self._ser.readline()
            response = raw.decode(errors="replace").strip()
            if not response.startswith("OK"):
                raise RuntimeError(
                    f"Supervisor command failed: cmd={cmd!r}, "
                    f"response={response!r}"
                )
            return response
        finally:
            if timeout is not None:
                self._ser.timeout = old_timeout

    # ---- Public commands -------------------------------------------------

    def ping(self) -> str:
        """Verify the supervisor is alive.  Returns ``'OK PONG'``."""
        return self._cmd("PING")

    def status(self) -> str:
        """Query supervisor status.  Returns ``'OK STATUS READY'``."""
        return self._cmd("STATUS")

    def reset(self, ms: int = 300) -> str:
        """Pulse EN/RST low for *ms* milliseconds.

        The Pico clamps the value to [50, 3000].  After releasing the
        reset line, the Pico waits an additional ~800 ms for the target
        to begin booting before replying, so the total round-trip time
        is roughly ``ms + 800`` ms.

        Returns the Pico's acknowledgement string, e.g.
        ``'OK RESET_DONE 300'``.
        """
        # Budget: ms (assert) + 800 ms (Pico-side boot wait) + margin.
        cmd_timeout = (min(max(ms, 50), 3000) + 800) / 1000.0 + 2.0
        return self._cmd(f"RESET {ms}", timeout=cmd_timeout)

    def power_cycle(self, off_ms: int = 3000) -> str:
        """Cut USB 5 V power for *off_ms* milliseconds, then restore.

        This simulates a physical USB cable unplug/replug.  The Pico
        also holds EN/RST low while power is off and releases it after
        power is restored, ensuring a clean boot.

        The Pico clamps the off-time to [1000, 10000].  After restoring
        power the Pico waits ~3100 ms (100 ms settle + 3000 ms boot) before
        replying, so total round-trip is roughly ``off_ms + 3100`` ms.

        Returns the Pico's acknowledgement string, e.g.
        ``'OK POWER_CYCLE_DONE 3000'``.
        """
        clamped = min(max(off_ms, 1000), 10000)
        cmd_timeout = (clamped + 3100) / 1000.0 + 2.0
        return self._cmd(f"POWER_CYCLE {off_ms}", timeout=cmd_timeout)

    # ---- Lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Release the serial port."""
        try:
            self._ser.close()
        except Exception:
            pass

    def __enter__(self) -> "PicoSupervisor":
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[override]
        self.close()

    def __repr__(self) -> str:
        return f"PicoSupervisor(port={self.port!r})"
