# RIFT-V

<div align="center">

**RIFT-V** stands for **R**untime-aware **I**njection of **F**aults in **T**inyML on RISC-**V**.

</div>

<br><br>



**RIFT-V** is a runtime-aware statistical fault injection framework for evaluating the reliability of TinyML inference on RISC-V microcontrollers.


---

RIFT-V provides a declarative workflow for bit-flip fault-injection campaigns on TensorFlow Lite Micro models running on RISC-V microcontrollers such as ESP32-C3 and ESP32-C6. It supports **register**, **memory**, Single Event Upset(SEU), MBU, and **hybrid** campaigns, with stratified statistical sampling, automatic C/C++ header generation, PlatformIO build/flash orchestration, UART capture, result classification, and physical memory analysis.

## Requirements

- Python >= 3.10
- A supported ESP32 RISC-V board with working USB/serial drivers
- A Python virtual environment, recommended for dependency isolation
- PlatformIO Core for firmware build and flash steps

The typical user workflow is intentionally simple: **edit a YAML configuration file and run `riftv ...`**. RIFT-V automatically selects the correct PlatformIO environment, generates the required headers, builds the firmware, flashes the board, captures results, and analyzes the campaign output.

## Recommended Installation

### 1. Create and activate a virtual environment

**PowerShell (Windows)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**bash (Linux/macOS)**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install the framework

**Recommended for most users**

```bash
pip install -e ".[all]"
```

This single command installs:

- The Python package `riftv` and the `riftv` command-line interface
- RIFT-V runtime dependencies
- Development and test dependencies
- TensorFlow, for quantization and test-subset generation
- PlatformIO Core, for firmware build and flash operations

**Alternative profiles**

Choose one of these instead of `[all]` if you want a smaller install:

- `pip install -e .` — framework, CLI, and runtime dependencies only
- `pip install -e ".[dev]"` — adds test/lint dependencies
- `pip install -e ".[tf]"` — adds TensorFlow
- `pip install -e ".[firmware]"` — adds PlatformIO Core

> USB/serial drivers and toolchain/SDK packages downloaded by PlatformIO remain operating-system dependent.

### 3. Quick verification

```bash
riftv --help
```

If this command works, the CLI installation is correct.

### 4. Optional: keep the PlatformIO cache inside the project

This is useful when you want to avoid relying on the user's global PlatformIO directory.

**PowerShell**

```powershell
$env:PLATFORMIO_CORE_DIR = "$PWD/firmware/.platformio-core"
```

**bash**

```bash
export PLATFORMIO_CORE_DIR="$PWD/firmware/.platformio-core"
```

## Project Structure

```text
riftv/                   # Main Python package
  cli.py                 # CLI entry point
  config.py              # YAML schema and configuration loader
  population.py          # Model introspection, population generation, and sampling
  experiment.py          # Orchestrator: prepare → build → flash → run → analyze
  headers.py             # C/C++ header generation for the firmware
  builder.py             # PlatformIO build and flash orchestration
  uart.py                # UART capture from firmware JSON-line streams
  classify.py            # Result classification: Masked, Critical, Crash, etc.
  analysis.py            # Physical memory analysis by tensor, operator, and offset

configs/                 # Example YAML configurations
  registerfi_example.yaml
  memoryfi_example.yaml
  hybrid_campaign.yaml
  multi_injection_memory_example.yaml
  multi_injection_register_example.yaml

firmware/                # ESP32 firmware project
  src/
    main_unified.cpp     # Unified firmware entry point selected by compile-time macros
    register_fi.cc/h     # Register fault-injection module
    memory_fi.cc/h       # Memory fault-injection module for model weights

tests/                   # Pytest test suite
models/                  # TFLite models and NPZ test subsets
runs/                    # Automatically generated campaign outputs
```

## Usage

### What users usually need to configure

In most cases, users only need to review these YAML fields:

- `model.tflite` — TFLite model to use
- `model.subset_npz` — test subset to embed into the firmware
- `campaign.type` — `register`, `memory`, or `hybrid`
- `serial.port` — serial port used by the board
- `serial.transport` — `uart0` or `usb_serial_jtag`
- `firmware.board` — PlatformIO board identifier, if different from the example

The rest of the pipeline is handled by RIFT-V.

### Recommended daily workflow

1. Choose or copy a YAML file from `configs/`
2. Adjust `serial.port` and `serial.transport`
3. Run the full pipeline

```bash
riftv configs/registerfi_example.yaml
```

This command automatically executes:

- `prepare()` — generates the population, statistical sample, and firmware headers
- `build()` — builds the firmware
- `flash()` — flashes the firmware to the board
- `run()` — captures UART output
- `analyze()` — classifies the collected results

It also automatically selects the correct PlatformIO environment based on `campaign.type`.

### Running individual steps

Use individual steps when debugging or repeating a specific phase:

```bash
riftv configs/registerfi_example.yaml --step prepare
riftv configs/registerfi_example.yaml --step build
riftv configs/registerfi_example.yaml --step flash
riftv configs/registerfi_example.yaml --step run
riftv configs/registerfi_example.yaml --step analyze

# Rebuild results.csv from results.jsonl
riftv configs/memoryfi_example.yaml --step rebuild-csv --run-dir runs/my_run

# Analyze an existing run
riftv configs/memoryfi_example.yaml --step analyze --run-dir runs/my_run
```

### CLI vs. Python API

- Use the **CLI** (`riftv`) for the standard campaign workflow
- Use the **Python API** (`import riftv`) when integrating RIFT-V into notebooks, custom scripts, or larger automation pipelines

### Programmatic usage

The main lifecycle is `prepare()` → `build()` → `flash()` → `run()` → `analyze()`.

If you already have a compatible int8 `.tflite` model and `.npz` test subset, you can start directly from `prepare()`. If you still need to convert a Keras/SavedModel model or generate a test subset, use the optional `quantize_model()` and `prepare_test_subset()` phases first.

```python
from riftv import load_config, Experiment

cfg = load_config("configs/memoryfi_example.yaml")
exp = Experiment(cfg)

# Optional — only needed when generating these files from a model or dataset:
# exp.quantize_model("models/my_model.h5", dataset_name="fashion_mnist")
# exp.prepare_test_subset(subset_size=100, dataset_name="fashion_mnist")

exp.prepare()
exp.build()
exp.flash()
exp.run()
exp.analyze()
```

To execute the full pipeline in one call:

```python
from riftv import load_config, Experiment

cfg = load_config("configs/registerfi_example.yaml")
exp = Experiment(cfg)
exp.execute()
```

### External scripts

If a user activates the Python environment where RIFT-V is installed and runs a script with that Python interpreter, RIFT-V automatically resolves:

- The PlatformIO executable from the active environment
- The local `PLATFORMIO_CORE_DIR`, which defaults to `firmware/.platformio-core`

This means users can create Python scripts **outside the repository**, as long as:

- The environment with RIFT-V (`riftv`) is active
- The script imports `riftv`
- The YAML file correctly points to `firmware/src`

Manual `PATH` or `PLATFORMIO_CORE_DIR` configuration is not required for the normal programmatic workflow.

### Example: running a script outside the repository

```python
from pathlib import Path
from riftv import load_config, Experiment

config_path = Path(r"C:\path\to\RIFT-V\configs\registerfi_example.yaml")

cfg = load_config(config_path)
exp = Experiment(cfg)
exp.execute()
```

Example execution:

```powershell
python C:\my_scripts\run_registerfi.py
```

## YAML Configuration

Each experiment is defined by a YAML file with the following sections:

| Section | Description |
|---------|-------------|
| `experiment` | Experiment name and description |
| `model` | Paths to the `.tflite` model and `.npz` subset |
| `campaign` | Campaign type, fault duration, target operators, and model-specific options |
| `sampling` | Stratified sampling method, confidence, margin, prior probability, seed, and grouping columns |
| `firmware` | Firmware source directory, PlatformIO platform, board, framework, and target |
| `serial` | Serial port, baud rate, boot timeout, and transport mode |
| `output` | Output directory and generated formats |

See the examples in `configs/`, especially:

- `registerfi_example.yaml`
- `memoryfi_example.yaml`
- `hybrid_campaign.yaml`
- `memoryfi_esp32_c6_example.yaml`
- `memoryfi_esp32_c3_supermini_example.yaml`

### Serial transport (`serial.transport`)

The communication channel between firmware and host depends on **how the board's USB connector is wired**.

| Value | When to use | DTR/RTS reset |
|-------|-------------|---------------|
| `uart0` | Boards with a USB-UART bridge such as CP2102, CH340, or FTDI | Automatically enabled |
| `usb_serial_jtag` | Boards whose USB connector is wired directly to the native USB Serial/JTAG peripheral, such as ESP32-C3 Super Mini or ESP32-C6/H2/P4 native USB boards | Disabled, because DTR/RTS reset does not work without a bridge |

When `transport: usb_serial_jtag`:

- The builder injects `-DFI_TRANSPORT_USB_SERIAL_JTAG=1` and `CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG=y` into `sdkconfig`
- The firmware uses the `usb_serial_jtag` driver instead of `UART0`
- The host skips DTR/RTS toggling and waits for the firmware boot event

Related optional fields:

```yaml
serial:
  port: COM17
  baud: 460800
  transport: usb_serial_jtag
  run_port: COM17              # Capture port, if different from the flash port
  reset_before_capture: null   # null=auto, true=force DTR/RTS, false=never
  boot_timeout_s: 30

  # ---- Hang recovery cascade --------------------------------------
  # When a faulted job fails to ack and the firmware does not reboot
  # on its own, the host runs a reset cascade (DTR/RTS, USB-JTAG,
  # serial BREAK, watchdog wait).  The strategy that produced a fresh
  # boot event is recorded per-row in ``recovery_method``.
  max_hang_recoveries: 3       # forced HW-reset attempts per hang
  hang_recovery_delay_s: 2.0   # delay between attempts (seconds)

  # ---- Adaptive per-job timeout -----------------------------------
  # The host measures the worst inter-result gap during the golden
  # phase and uses ``max_gap × safety_factor`` as the per-job
  # timeout, clamped into [min_s, max_s].  This avoids waiting an
  # arbitrary 60 s on tiny models while leaving headroom on larger
  # ones.  Set ``job_timeout_safety_factor`` to ``null`` to disable
  # calibration and pin the timeout to ``job_timeout_max_s``.
  job_timeout_safety_factor: 20.0
  job_timeout_min_s: 1.0
  job_timeout_max_s: 60.0
```

> To identify the correct transport, inspect the board's USB interface. If it uses CP2102, CH340, CH343, FT232, or a similar bridge, choose `uart0`. If the connector is wired directly to the SoC's native USB peripheral, choose `usb_serial_jtag`.

#### Identifying the transport in Windows Device Manager

The COM-port name assigned by Windows is often a strong indicator:

| Device Manager name | Type | `transport` |
|---------------------|------|-------------|
| *"USB Serial Device"* using the generic `usbser.sys` driver | Native USB CDC / USB Serial-JTAG | `usb_serial_jtag` |
| *"USB-Enhanced-SERIAL CH343"* or *"USB-SERIAL CH340"* | WCH USB-UART bridge | `uart0` |
| *"Silicon Labs CP210x USB to UART Bridge"* | Silicon Labs USB-UART bridge | `uart0` |
| *"USB Serial Port"* | FTDI USB-UART bridge | `uart0` |

Tested examples:

- `ESP32-C3 Super Mini` → *"USB Serial Device (COM17)"* → `transport: usb_serial_jtag`
- `ESP32-C6 DevKitC WiFi ZigBee N4` → *"USB-Enhanced-SERIAL CH343 (COM18)"* → `transport: uart0`

## Result Classification

Each fault-injection result is classified into one of the following categories:

| Class | Meaning |
|-------|---------|
| **Masked** | Output is identical to the golden reference |
| **Acceptable** | Same predicted class, with less than 5% confidence variation |
| **Good** | Faulted result improves a wrong golden prediction |
| **Observed** | Predicted class differs from the golden reference but remains correct with high confidence |
| **Warning** | Same predicted class with a significant confidence drop |
| **Critical** | Wrong predicted class or very low confidence |
| **Crash** | Firmware did not complete inference but rebooted on its own |
| **Hang** | Firmware stopped responding and required a forced HW reset to recover (`status == "fatal_hang"`) |
| **NoInjection** | Register campaign result where the requested injection did not actually occur |

## Result Auditing & Recovery

Every row written to `results.jsonl` / `results.csv` carries host-side audit
fields that distinguish a *Completed* job from a *Crash*, a *Hang*, or an
inferred sibling row, and that record which recovery strategy brought the
chip back when needed:

| Field | Meaning |
|-------|---------|
| `elapsed_ms` | Wall-clock between `run_job` send and `job_ack` (or timeout). `null` for inferred rows that were never dispatched. |
| `timeout_ms` | Per-job timeout in effect when the row was produced. |
| `boot_seen` | `true` when a `{"event":"boot"}` was observed for this fault (during the job, during the natural-reboot wait, or as part of the HW-reset cascade). |
| `result_seen` | `true` when the firmware emitted a `result` for this specific job. |
| `job_ack_seen` | `true` when the firmware emitted `job_ack` for this specific job. |
| `recovery_method` | Stable id of the strategy that recovered the chip: `none`, `natural_boot`, `dtr_rts_classic`, `usb_jtag_serial`, `serial_break`, or `watchdog_wait`. Inherited by inferred rows so the whole fault carries the same characterization. |
| `inferred` | `true` for rows synthesized after a deterministic fault crash (image was *not* actually dispatched — its `recovery_method` and `boot_seen` are inherited from the parent crash, while `elapsed_ms`/`result_seen`/`job_ack_seen` stay null/false). |

The recovery cascade itself is configurable in
`serial.{max_hang_recoveries, hang_recovery_delay_s}`; it tries DTR/RTS,
USB-JTAG, serial BREAK and the firmware watchdog in that order, and the
*first* strategy that produces a fresh boot event is the one recorded.

The per-job timeout is calibrated automatically from the golden phase
(see `job_timeout_safety_factor` in the YAML reference), so tiny models
do not pay an arbitrary 60 s wall-clock penalty per crashed job.

## Physical Analysis

The `analysis` module supports physical-space analysis of fault criticality:

- Criticality by tensor and operator
- Binning by physical offset in the serialized model blob
- Tensor × offset heatmap data
- Contiguity analysis across adjacent tensors
- Semantic vs. physical criticality comparison
- Bit-position vulnerability analysis

```python
from riftv import generate_physical_report

report = generate_physical_report("runs/my_campaign/")
print(report.summary())
report.save("runs/my_campaign/")
```

## PlatformIO Environments

The `firmware/platformio.ini` file defines two campaign build environments:

- **`registerfi`** — defines `FI_CAMPAIGN_REGISTER=1`
- **`memoryfi`** — defines `FI_CAMPAIGN_MEMORY=1`

RIFT-V automatically selects the correct environment based on `campaign.type`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
