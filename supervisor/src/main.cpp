/*
 * RIFT-V External Reset Supervisor
 * =================================
 * Board:  Raspberry Pi Pico (RP2040)
 * Core:   arduino-pico by Earle Philhower (via PlatformIO)
 *
 * Purpose
 * -------
 * Acts as an out-of-band reset controller for the ESP32-C3/C6 target
 * under test.  The host PC sends commands over the Pico's own USB-Serial
 * link; the Pico drives the target's EN/RST line via open-drain logic
 * and can perform a full USB power cycle via a relay on the 5 V line.
 *
 * This allows the RIFT-V framework to physically reset the target even
 * when the target's own USB connection has disappeared due to a fatal
 * hang (e.g. sp/gp corruption on usb_serial_jtag).
 *
 * Wiring
 * ------
 *   Pico GND  ──────────────── GND    of ESP32 target
 *   Pico GP2  ──[1 kΩ opt.]── EN/RST of ESP32 target
 *   Pico GP3  ──────────────── IN of relay module (controls ESP32 USB 5 V)
 *   Pico GP4  ──[330Ω]─────── D+ line (forces USB disconnect signal)
 *
 *   GP3 drives a relay module (active-HIGH) that switches the ESP32's
 *   USB VBUS supply.  LOW = relay at rest = power on,
 *   HIGH = relay activated = power off.
 *   This simulates a physical USB cable unplug/replug.
 *
 *   Pico → Relay module:
 *
 *     Pico VBUS (pin 40) ──── VCC of relay
 *     Pico GND  (pin 38) ──── GND of relay
 *     Pico GP3  (pin 5)  ──── IN  of relay
 *
 *   USB-C breakout → Relay → ESP32 (power-cycle path):
 *
 *     Breakout VBUS/5V ──── COM of relay
 *     Relay NC         ──── 5V/VIN of ESP32
 *     Breakout GND     ──── GND    of ESP32
 *     Breakout D+      ──── GPIO13 / USB D+ of ESP32
 *     Breakout D-      ──── GPIO12 / USB D- of ESP32
 *
 *   Use COM and NC only — do NOT use the NO terminal.
 *   Do NOT connect 3V3 between the two boards.
 *   Keep D+/D- wires short and parallel (USB is signal-sensitive).
 *
 * Open-drain by software (GP2)
 * ----------------------------
 * GP2 is NEVER driven HIGH.  Two states only:
 *
 *   Normal:  pinMode(GP2, INPUT)       → Hi-Z, target runs freely
 *   Reset:   pinMode(GP2, OUTPUT LOW)  → pulls EN/RST to GND
 *
 * This is safe for direct 3.3 V ↔ 3.3 V connection.  Do NOT use with
 * 5 V Arduino boards (Uno/Nano) — voltage mismatch can damage the ESP32.
 *
 * Protocol (115200 baud, newline-terminated)
 * ------------------------------------------
 *   Host → Pico            Pico → Host
 *   ────────────────       ──────────────────────
 *   PING\n                 OK PONG\n
 *   STATUS\n               OK STATUS READY\n
 *   RESET\n                OK RESET_DONE\n           (300 ms default pulse)
 *   RESET 500\n            OK RESET_DONE 500\n       (custom pulse in ms)
 *   POWER_CYCLE\n          OK POWER_CYCLE_DONE\n     (3 s off, default)
 *   POWER_CYCLE 5000\n     OK POWER_CYCLE_DONE 5000\n(custom off-time ms)
 *   <anything else>        ERR UNKNOWN_CMD\n
 *
 * Integration
 * -----------
 * In the RIFT-V YAML config, set:
 *
 *   serial:
 *     supervisor_port: COM21   # ← Pico's COM port
 *
 * The framework's PicoSupervisor class (riftv/supervisor.py) handles
 * communication automatically during fault-injection campaigns.
 */

#include <Arduino.h>

// ─── Configuration ──────────────────────────────────────────────────
const int RESET_PIN           = 2;    // GP2 — open-drain to EN/RST
const int POWER_PIN           = 3;    // GP3 — relay IN for USB 5 V
const int USB_DP_PIN          = 4;    // GP4 — force D+ low for disconnect
const unsigned long BAUD_RATE = 115200;

// Reset pulse clamp range (milliseconds)
const int PULSE_MIN_MS = 50;
const int PULSE_MAX_MS = 3000;

// Power-cycle off-time clamp range (milliseconds)
const int PCYCLE_MIN_MS = 1000;
const int PCYCLE_MAX_MS = 10000;
const int PCYCLE_DEFAULT_MS = 3000;

// Post-release boot delay: wait for the target to begin booting before
// replying to the host so the host doesn't race ahead of the chip.
const unsigned long BOOT_SETTLE_MS = 3000;

// ─── Helpers ────────────────────────────────────────────────────────

String readCommand() {
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();
  return cmd;
}

void release_reset() {
  // Hi-Z: the Pico stops interfering with EN/RST.
  // The ESP32's own pull-up keeps EN/RST high.
  pinMode(RESET_PIN, INPUT);
}

void assert_reset() {
  // Pull EN/RST to GND.  Never use digitalWrite(RESET_PIN, HIGH).
  pinMode(RESET_PIN, OUTPUT);
  digitalWrite(RESET_PIN, LOW);
}

void pulse_reset(unsigned long ms) {
  assert_reset();
  delay(ms);

  release_reset();

  // Let the ESP32 boot before replying.
  delay(BOOT_SETTLE_MS);
}

void usb_disconnect() {
  // Pull D+ low so the PC detects a real USB disconnect.
  pinMode(USB_DP_PIN, OUTPUT);
  digitalWrite(USB_DP_PIN, LOW);
}

void usb_release() {
  // Release D+ — Hi-Z so it doesn't interfere with normal USB traffic.
  pinMode(USB_DP_PIN, INPUT);
}

void power_off() {
  // Force USB disconnect signal BEFORE cutting power.
  usb_disconnect();
  delay(50);  // Let the PC register the disconnect.
  // Activate relay (active-HIGH) → COM disconnects from NC → ESP32 loses 5 V.
  digitalWrite(POWER_PIN, HIGH);
}

void power_on() {
  // Deactivate relay → COM reconnects to NC → ESP32 gets 5 V.
  digitalWrite(POWER_PIN, LOW);
  delay(100);  // Let power stabilize before releasing D+.
  // Release D+ so the ESP32 USB PHY can drive it normally.
  usb_release();
}

void power_cycle(unsigned long off_ms) {
  // Also assert EN/RST while power is off, so the chip doesn't
  // start in a weird state when power returns.
  assert_reset();
  power_off();
  delay(off_ms);

  // Power on first, then release reset after a short settle.
  power_on();
  delay(100);
  release_reset();

  // Let the ESP32 boot and USB enumerate before replying.
  delay(BOOT_SETTLE_MS);
}

// ─── Arduino entry points ───────────────────────────────────────────

void setup() {
  Serial.begin(BAUD_RATE);

  // Safe initial state: power on, D+ released, not in reset.
  pinMode(POWER_PIN, OUTPUT);
  pinMode(USB_DP_PIN, INPUT);  // Hi-Z — don't interfere with USB
  power_on();
  release_reset();

  delay(1000);
  Serial.println("SUPERVISOR_READY");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String cmd = readCommand();

  if (cmd == "PING") {
    Serial.println("OK PONG");
  }

  else if (cmd == "RESET") {
    pulse_reset(300);
    Serial.println("OK RESET_DONE");
  }

  else if (cmd.startsWith("RESET ")) {
    int ms = cmd.substring(6).toInt();

    if (ms < PULSE_MIN_MS) {
      ms = PULSE_MIN_MS;
    }
    if (ms > PULSE_MAX_MS) {
      ms = PULSE_MAX_MS;
    }

    pulse_reset(ms);

    Serial.print("OK RESET_DONE ");
    Serial.println(ms);
  }

  else if (cmd == "POWER_CYCLE") {
    power_cycle(PCYCLE_DEFAULT_MS);
    Serial.println("OK POWER_CYCLE_DONE");
  }

  else if (cmd.startsWith("POWER_CYCLE ")) {
    int ms = cmd.substring(12).toInt();

    if (ms < PCYCLE_MIN_MS) {
      ms = PCYCLE_MIN_MS;
    }
    if (ms > PCYCLE_MAX_MS) {
      ms = PCYCLE_MAX_MS;
    }

    power_cycle(ms);

    Serial.print("OK POWER_CYCLE_DONE ");
    Serial.println(ms);
  }

  else if (cmd == "STATUS") {
    Serial.println("OK STATUS READY");
  }

  else {
    Serial.println("ERR UNKNOWN_CMD");
  }
}
