"""Hardware smoke-test for the Raspberry Pi Pico external reset supervisor.

This is *not* a unit test — it talks to a real Pico over USB.  Skip it
when the supervisor is not connected.

Usage (from the project root)::

    # Standalone smoke test (no ESP32 needed):
    python -m tests.test_pico_supervisor COM15

    # Or with pytest, pointing to the right port via env var:
    set RIFTV_SUPERVISOR_PORT=COM15
    pytest tests/test_pico_supervisor.py -v
"""
from __future__ import annotations

import os
import sys
import time

from riftv.supervisor import PicoSupervisor


SUPERVISOR_PORT = os.environ.get("RIFTV_SUPERVISOR_PORT")


# pytest is optional: needed only when running the file through `pytest`.
# The standalone smoke-test mode (``python tests/test_pico_supervisor.py
# COM22``) works without it.
try:
    import pytest

    pytestmark = pytest.mark.skipif(
        SUPERVISOR_PORT is None,
        reason="Set RIFTV_SUPERVISOR_PORT=COMxx to run hardware tests",
    )

    @pytest.fixture
    def supervisor():
        """Open a fresh PicoSupervisor connection per test."""
        sup = PicoSupervisor(SUPERVISOR_PORT)
        yield sup
        sup.close()

    def test_ping(supervisor):
        """Pico must respond to PING with OK PONG."""
        assert supervisor.ping() == "OK PONG"

    def test_status(supervisor):
        """Pico must report READY status."""
        assert supervisor.status() == "OK STATUS READY"

    def test_reset_default(supervisor):
        """Default reset (300 ms pulse) must complete without error."""
        response = supervisor.reset()
        assert response.startswith("OK RESET_DONE")

    def test_reset_custom_duration(supervisor):
        """Custom-duration reset must echo back the (clamped) duration."""
        response = supervisor.reset(500)
        assert response == "OK RESET_DONE 500"

    def test_reset_clamping(supervisor):
        """Reset durations are clamped to [50, 3000] ms by the Pico firmware."""
        too_low = supervisor.reset(10)         # below 50 ms → clamped up
        too_high = supervisor.reset(9999)      # above 3000 ms → clamped down

        assert too_low == "OK RESET_DONE 50"
        assert too_high == "OK RESET_DONE 3000"

except ImportError:
    # pytest unavailable — only the standalone smoke-test mode below works.
    pass


# ----------------------------------------------------------------------
# Standalone CLI mode: ``python -m tests.test_pico_supervisor COM15``
# ----------------------------------------------------------------------

def _smoke_main(port: str) -> int:
    print(f"[smoke] Opening Pico supervisor on {port}...")
    sup = PicoSupervisor(port)
    try:
        print(f"[smoke] PING   → {sup.ping()}")
        print(f"[smoke] STATUS → {sup.status()}")

        print("[smoke] RESET  → pulsing EN/RST for 300 ms...")
        t0 = time.monotonic()
        ack = sup.reset(300)
        dt = time.monotonic() - t0
        print(f"[smoke] RESET  → {ack}  ({dt*1000:.0f} ms round-trip)")

        print("[smoke] All commands OK.  Pico supervisor is functional.")
        return 0
    except Exception as exc:
        print(f"[smoke] FAILED: {exc}")
        return 1
    finally:
        sup.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m tests.test_pico_supervisor <COM_PORT>")
        sys.exit(2)
    sys.exit(_smoke_main(sys.argv[1]))
