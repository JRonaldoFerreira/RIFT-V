"""Declarative YAML configuration loader and schema for FI experiments."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    tflite: str
    subset_npz: str


@dataclass
class MemoryCampaignConfig:
    target_ops: List[str] = field(default_factory=lambda: ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"])


# Valid RISC-V GPR names (x1–x31) — must stay in sync with population.RISCV_REGISTERS.
VALID_REGISTER_NAMES = {
    "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
}


@dataclass
class RegisterCampaignConfig:
    target_ops: List[str] = field(default_factory=lambda: ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"])
    slot_count: int = 3
    registers: Union[str, List[str]] = "all"
    bits: Union[str, List[int]] = "all"


@dataclass
class MultiInjectionConfig:
    enabled: bool = False
    count: int = 2
    num_groups: int = 500
    seed: int = 42


@dataclass
class CampaignConfig:
    type: str  # "memory", "register"
    fault_duration: str = "permanent"  # "transient" (per-operator, MicroFI) or "permanent" (whole inference)
    memory: Optional[MemoryCampaignConfig] = None
    register: Optional[RegisterCampaignConfig] = None
    multi_injection: Optional[MultiInjectionConfig] = None


@dataclass
class SamplingConfig:
    method: str = "stratified"
    confidence: float = 0.95
    margin: float = 0.05
    p_prior: float = 0.5
    seed: int = 1337
    group_by: List[str] = field(default_factory=lambda: ["op_index", "op_name"])


@dataclass
class FirmwareConfig:
    src_dir: str = "firmware/src"
    platform: str = "espressif32"
    board: str = "esp32-c3-devkitm-1"
    framework: str = "espidf"
    monitor_speed: int = 460800
    extra_build_flags: List[str] = field(default_factory=list)
    flash_size: Optional[str] = None          # e.g. "2MB", "4MB" — overrides board default
    idf_target: Optional[str] = None          # e.g. "esp32c6" — override IDF_TARGET
    sdkconfig_defaults: List[str] = field(default_factory=list)  # extra sdkconfig fragments
    tensor_arena_size_kb: Optional[int] = None  # None = auto-detect from heap at runtime


# Valid serial transports — extend this tuple when adding new channels.
VALID_TRANSPORTS = ("uart0", "usb_serial_jtag")


@dataclass
class SerialConfig:
    port: str = "COM3"
    baud: int = 460800
    boot_timeout_s: int = 30
    transport: str = "uart0"              # "uart0" or "usb_serial_jtag"
    run_port: Optional[str] = None        # capture port (defaults to *port* when None)
    reset_before_capture: Optional[bool] = None  # None = auto (True for uart0, False for usb_serial_jtag)
    supervisor_port: Optional[str] = None  # Pico supervisor COM port for out-of-band EN/RST reset
    max_hang_recoveries: int = 3          # HW reset attempts when firmware hangs without rebooting
    hang_recovery_delay_s: float = 2.0    # seconds to wait between HW reset attempts
    # ---- Crash-inference threshold (statistical / cost trade-off) -------
    # Number of consecutive crashes needed on the *same* fault index
    # before its remaining images are synthesized as ``inferred=True``
    # crash rows (skipping the actual dispatch + crash-recovery cost).
    # Any ok/SDC observation on the same fault resets the counter, so
    # the crashes must be strictly back-to-back.
    #
    # 0 (default)  → never infer; every (fault, image) pair is dispatched
    #                and measured.  Most statistically faithful — the only
    #                images skipped are those after a ``fatal_hang`` (chip
    #                physically unable to accept jobs).  Recommended.
    # 1            → infer immediately after the *first* crash.  Equivalent
    #                to the legacy "deterministic crash" assumption: cheap
    #                but biased for intermittent faults (a fault with true
    #                p_crash = 0.1 reported as 100 % crash if it tripped
    #                on the first image).
    # N ≥ 2        → require N consecutive crashes (1 original + N-1
    #                verification crashes) before inferring.  Middle ground
    #                between the two extremes.  N=6 misses intermittent
    #                faults with p_crash ≥ 0.95 about 77 % of the time but
    #                catches anything with p_crash ≤ 0.5 with ≥ 96.9 %
    #                probability.
    crash_inference_threshold: int = 0
    # ---- Adaptive job-timeout calibration --------------------------------
    # The host scheduler measures the worst inter-result gap during the
    # golden phase and uses ``max_gap × safety_factor`` as the per-job
    # timeout, clamped to [min_s, max_s].  This avoids waiting an
    # arbitrary ~60 s on tiny models while still leaving headroom on
    # large ones.  Set *safety_factor* to None to disable calibration
    # and fall back to *max_s* as a fixed timeout.
    job_timeout_safety_factor: Optional[float] = 20.0
    job_timeout_min_s: float = 1.0
    job_timeout_max_s: float = 60.0


@dataclass
class OutputConfig:
    dir: str = "runs/{name}_{timestamp}"
    formats: List[str] = field(default_factory=lambda: ["csv", "jsonl"])


@dataclass
class ExperimentConfig:
    name: str
    description: str
    model: ModelConfig
    campaign: CampaignConfig
    sampling: SamplingConfig
    firmware: FirmwareConfig
    serial: SerialConfig
    output: OutputConfig

    # Resolved at load time
    _root: Path = field(default=Path("."), repr=False)

    # ---- helpers -----------------------------------------------------------

    def resolve_path(self, p: str) -> Path:
        """Resolve a path relative to the config file's parent directory."""
        path = Path(p)
        if path.is_absolute():
            return path
        return (self._root / path).resolve()

    def resolve_output_dir(self) -> Path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rendered = self.output.dir.replace("{name}", self.name).replace("{timestamp}", ts)
        return self.resolve_path(rendered)

    @property
    def tflite_path(self) -> Path:
        return self.resolve_path(self.model.tflite)

    @property
    def subset_npz_path(self) -> Path:
        return self.resolve_path(self.model.subset_npz)

    @property
    def firmware_src_path(self) -> Path:
        return self.resolve_path(self.firmware.src_dir)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _parse_model(raw: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        tflite=raw["tflite"],
        subset_npz=raw["subset_npz"],
    )


def _parse_campaign(raw: Dict[str, Any]) -> CampaignConfig:
    ctype = raw["type"]
    fault_duration = raw.get("fault_duration", "permanent")
    if fault_duration not in ("transient", "permanent"):
        raise ValueError(f"campaign.fault_duration must be 'transient' or 'permanent', got '{fault_duration}'")

    memory = None
    register = None

    if ctype in ("memory", "hybrid"):
        mem_raw = raw.get("memory", {})
        memory = MemoryCampaignConfig(
            target_ops=mem_raw.get("target_ops", MemoryCampaignConfig().target_ops),
        )

    if ctype in ("register", "hybrid"):
        reg_raw = raw.get("register", {})
        raw_regs = reg_raw.get("registers", "all")
        if isinstance(raw_regs, list):
            unknown = set(raw_regs) - VALID_REGISTER_NAMES
            if unknown:
                raise ValueError(
                    f"campaign.register.registers contains unknown register name(s): {sorted(unknown)}. "
                    f"Valid names: {sorted(VALID_REGISTER_NAMES)}"
                )
            registers: Union[str, List[str]] = raw_regs
        else:
            if raw_regs != "all":
                raise ValueError(
                    f"campaign.register.registers must be 'all' or a list of register names, got '{raw_regs}'"
                )
            registers = "all"
        raw_bits = reg_raw.get("bits", "all")
        if isinstance(raw_bits, list):
            for b in raw_bits:
                if not isinstance(b, int) or b < 0 or b > 31:
                    raise ValueError(
                        f"campaign.register.bits entries must be integers in 0..31, got {b!r}"
                    )
            bits: Union[str, List[int]] = sorted(set(raw_bits))
        else:
            if raw_bits != "all":
                raise ValueError(
                    f"campaign.register.bits must be 'all' or a list of bit indices (0-31), got '{raw_bits}'"
                )
            bits = "all"
        register = RegisterCampaignConfig(
            target_ops=reg_raw.get("target_ops", RegisterCampaignConfig().target_ops),
            slot_count=reg_raw.get("slot_count", 3),
            registers=registers,
            bits=bits,
        )

    mi_raw = raw.get("multi_injection", {})
    multi_injection = None
    if mi_raw and mi_raw.get("enabled", False):
        multi_injection = MultiInjectionConfig(
            enabled=True,
            count=mi_raw.get("count", 2),
            num_groups=mi_raw.get("num_groups", 500),
            seed=mi_raw.get("seed", 42),
        )

    return CampaignConfig(
        type=ctype, fault_duration=fault_duration,
        memory=memory, register=register, multi_injection=multi_injection,
    )


def _parse_sampling(raw: Optional[Dict[str, Any]]) -> SamplingConfig:
    if raw is None:
        return SamplingConfig()
    return SamplingConfig(
        method=raw.get("method", "stratified"),
        confidence=raw.get("confidence", 0.95),
        margin=raw.get("margin", 0.05),
        p_prior=raw.get("p_prior", 0.5),
        seed=raw.get("seed", 1337),
        group_by=raw.get("group_by", ["op_index", "op_name"]),
    )


def _parse_firmware(raw: Optional[Dict[str, Any]]) -> FirmwareConfig:
    if raw is None:
        return FirmwareConfig()
    sdk_defaults = raw.get("sdkconfig_defaults", [])
    if isinstance(sdk_defaults, str):
        sdk_defaults = [sdk_defaults]
    return FirmwareConfig(
        src_dir=raw.get("src_dir", FirmwareConfig.src_dir),
        platform=raw.get("platform", FirmwareConfig.platform),
        board=raw.get("board", FirmwareConfig.board),
        framework=raw.get("framework", FirmwareConfig.framework),
        monitor_speed=raw.get("monitor_speed", FirmwareConfig.monitor_speed),
        extra_build_flags=raw.get("extra_build_flags", []),
        flash_size=raw.get("flash_size", None),
        idf_target=raw.get("idf_target", None),
        sdkconfig_defaults=sdk_defaults,
        tensor_arena_size_kb=raw.get("tensor_arena_size_kb", None),
    )


def _parse_serial(raw: Optional[Dict[str, Any]]) -> SerialConfig:
    if raw is None:
        return SerialConfig()
    transport = raw.get("transport", "uart0")
    if transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"serial.transport must be one of {VALID_TRANSPORTS}, got '{transport}'"
        )
    reset_val = raw.get("reset_before_capture", None)
    safety_raw = raw.get("job_timeout_safety_factor", 20.0)
    safety = None if safety_raw is None else float(safety_raw)
    return SerialConfig(
        port=raw.get("port", "COM3"),
        baud=raw.get("baud", 460800),
        boot_timeout_s=raw.get("boot_timeout_s", 30),
        transport=transport,
        run_port=raw.get("run_port", None),
        reset_before_capture=reset_val,
        supervisor_port=raw.get("supervisor_port", None),
        max_hang_recoveries=raw.get("max_hang_recoveries", 3),
        hang_recovery_delay_s=float(raw.get("hang_recovery_delay_s", 2.0)),
        crash_inference_threshold=int(
            raw.get(
                "crash_inference_threshold",
                # Backward-compat alias from the short-lived previous name.
                raw.get("crash_verification_batch", 0),
            )
        ),
        job_timeout_safety_factor=safety,
        job_timeout_min_s=float(raw.get("job_timeout_min_s", 1.0)),
        job_timeout_max_s=float(raw.get("job_timeout_max_s", 60.0)),
    )


def _parse_output(raw: Optional[Dict[str, Any]]) -> OutputConfig:
    if raw is None:
        return OutputConfig()
    return OutputConfig(
        dir=raw.get("dir", OutputConfig().dir),
        formats=raw.get("formats", ["csv", "jsonl"]),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Config file must be a YAML mapping at the top level.")

    exp_raw = raw.get("experiment", raw)

    name = exp_raw.get("name", path.stem)
    description = exp_raw.get("description", "")

    if "model" not in raw:
        raise ValueError("Config must include a 'model' section with 'tflite' and 'subset_npz'.")
    if "campaign" not in raw:
        raise ValueError("Config must include a 'campaign' section with at least 'type'.")

    cfg = ExperimentConfig(
        name=name,
        description=description,
        model=_parse_model(raw["model"]),
        campaign=_parse_campaign(raw["campaign"]),
        sampling=_parse_sampling(raw.get("sampling")),
        firmware=_parse_firmware(raw.get("firmware")),
        serial=_parse_serial(raw.get("serial")),
        output=_parse_output(raw.get("output")),
        _root=path.parent.resolve(),
    )

    # Validate campaign type
    valid_types = ("memory", "register", "hybrid")
    if cfg.campaign.type not in valid_types:
        raise ValueError(f"campaign.type must be one of {valid_types}, got '{cfg.campaign.type}'")

    return cfg
