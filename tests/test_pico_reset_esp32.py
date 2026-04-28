"""Quick test: verify the Pico supervisor physically resets the ESP32.

Strategy
--------
The ESP32 may already be in a watchdog boot loop.  To tell whether the
Pico *actually* forces a reset we:

  1. Measure the natural boot-loop interval (if any).
  2. Wait until just after a boot message (start of a fresh interval).
  3. Immediately fire the Pico RESET.
  4. If the next boot arrives *much sooner* than the natural interval,
     the Pico caused it.  If no boot loop exists and a boot appears
     after the reset, that is also a clear success.

Usage:
    python -m tests.test_pico_reset_esp32 COM19 COM22
                                          ^ESP  ^Pico
"""
from __future__ import annotations

import sys
import time

import serial


def _read_boot(esp: serial.Serial, timeout: float):
    """Wait up to *timeout* s for a boot line. Return (line, elapsed) or (None, elapsed)."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    while time.monotonic() < deadline:
        raw = esp.readline()
        if raw:
            line = raw.decode(errors="replace").strip()
            if "boot" in line.lower():
                return line, time.monotonic() - t0
    return None, time.monotonic() - t0


def main(esp_port: str, pico_port: str) -> int:
    ESP_BAUD = 460800
    PICO_BAUD = 115200

    # ---- Open ESP32 ----
    print(f"[test] Opening ESP32 on {esp_port} @ {ESP_BAUD}...")
    esp = serial.Serial(esp_port, ESP_BAUD, timeout=2)
    time.sleep(0.5)
    esp.reset_input_buffer()

    # ---- Phase 1: measure natural boot-loop cadence ----
    print("[test] Phase 1 — measuring natural boot interval (up to 20 s)...")
    intervals = []
    last_boot = None
    phase1_deadline = time.monotonic() + 20
    while time.monotonic() < phase1_deadline and len(intervals) < 5:
        line, _ = _read_boot(esp, timeout=6)
        if line is None:
            break
        now = time.monotonic()
        if last_boot is not None:
            intervals.append(now - last_boot)
        last_boot = now

    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        print(f"[test] Boot loop detected — avg interval: {avg_interval:.2f} s  "
              f"({len(intervals)} samples)")
    else:
        avg_interval = None
        if last_boot is not None:
            print("[test] Only one boot seen — no loop measured.")
        else:
            print("[test] No boot messages seen (ESP32 may be idle — good).")

    # ---- Phase 2: sync to a fresh boot and immediately fire Pico RESET ----
    # Open the Pico early so the DTR settle delay doesn't eat our window.
    print(f"[test] Opening Pico on {pico_port}...")
    pico = serial.Serial(pico_port, PICO_BAUD, timeout=2)
    time.sleep(2)
    pico.reset_input_buffer()

    # If there is a boot loop, wait for the next boot so we are right at
    # the start of an interval.  Otherwise just proceed.
    if avg_interval is not None:
        print("[test] Syncing to next natural boot...")
        _read_boot(esp, timeout=avg_interval + 4)

    # ---- Phase 2: long-hold silence test ----
    # Strategy: send a RESET with a hold time LONGER than the natural boot
    # interval.  While EN/RST is held low the ESP32 cannot run, so it
    # produces NO boot messages.  If we see silence for longer than the
    # natural interval, the wire is working.
    #
    # We pick hold_ms = max(2500, avg_interval*3*1000) to guarantee the
    # hold spans several natural cycles.
    if avg_interval is not None:
        hold_ms = max(2500, int(avg_interval * 3 * 1000))
    else:
        hold_ms = 2500

    # The Pico RESET command blocks for hold_ms + 800 ms settle before
    # replying.  During the hold, we watch the ESP32 serial for silence.
    esp.reset_input_buffer()

    print(f"[test] Phase 2 — sending RESET {hold_ms} (hold {hold_ms} ms)...")
    t_send = time.monotonic()
    pico.write(f"RESET {hold_ms}\n".encode())
    pico.flush()

    # While the Pico holds EN/RST low, monitor the ESP32 for silence.
    # We check for the duration of the hold (Pico won't reply until done).
    silence_window = hold_ms / 1000.0
    print(f"[test] Monitoring ESP32 for {silence_window:.1f} s while Pico holds reset...")
    boots_during_hold = 0
    hold_deadline = time.monotonic() + silence_window
    esp.timeout = 0.3  # short reads
    while time.monotonic() < hold_deadline:
        raw = esp.readline()
        if raw:
            line = raw.decode(errors="replace").strip()
            if "boot" in line.lower():
                boots_during_hold += 1

    # Now read the Pico acknowledgement (it should arrive ~0.8 s after release).
    pico.timeout = 3.0
    pico_ack = pico.readline().decode(errors="replace").strip()
    print(f"[test] Pico response: {pico_ack}")
    pico.close()

    if not pico_ack.startswith("OK"):
        print("[test] FAIL — Pico did not acknowledge the reset.")
        esp.close()
        return 1

    # ---- Phase 3: check for boot after release ----
    esp.timeout = 5.0
    esp.reset_input_buffer()
    boot_after, elapsed = _read_boot(esp, timeout=8)
    esp.close()

    # Expected boots during hold if wire is NOT connected:
    if avg_interval is not None:
        expected_during_hold = silence_window / avg_interval
    else:
        expected_during_hold = 0

    print(f"[test] Boots during {silence_window:.1f} s hold: {boots_during_hold} "
          f"(expected ~{expected_during_hold:.0f} without wire)")
    if boot_after:
        print(f"[test] Boot after release: {boot_after} ({elapsed:.2f} s)")
    else:
        print(f"[test] Boot after release: none")

    # ---- Verdict ----
    if avg_interval is not None:
        # Boot loop case: if wire works, ESP32 was silent during the hold.
        if boots_during_hold == 0:
            print(f"\n[test] SUCCESS — ESP32 was SILENT during the {silence_window:.1f} s hold.")
            print("       The Pico held EN/RST low and suppressed the boot loop.")
            if boot_after:
                print("       Boot resumed after release — circuit confirmed working.")
            return 0
        else:
            print(f"\n[test] FAIL — ESP32 kept booting during hold ({boots_during_hold} boots).")
            print("       The Pico is NOT controlling EN/RST.")
            print("       Check wiring: Pico GND ↔ ESP32 GND, Pico GP2 ↔ ESP32 EN/RST")
            return 1
    else:
        # No boot loop: just check if a boot appeared after release.
        if boot_after:
            print(f"\n[test] SUCCESS — ESP32 booted {elapsed:.2f} s after Pico released reset.")
            return 0
        else:
            print(f"\n[test] FAIL — no boot from ESP32 after Pico reset.")
            print("       Check wiring: Pico GND ↔ ESP32 GND, Pico GP2 ↔ ESP32 EN/RST")
            return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m tests.test_pico_reset_esp32 <ESP_PORT> <PICO_PORT>")
        print("Example: python -m tests.test_pico_reset_esp32 COM19 COM22")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
