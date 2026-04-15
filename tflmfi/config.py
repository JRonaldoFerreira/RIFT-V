"""Declarative YAML configuration loader and schema for FI experiments."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class RegisterCampaignConfig:
    target_ops: List[str] = field(default_factory=lambda: ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"])
    slot_count: int = 3
    registers: str = "all"


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


@dataclass
class SerialConfig:
    port: str = "COM3"
    baud: int = 460800
    boot_timeout_s: int = 30


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
        register = RegisterCampaignConfig(
            target_ops=reg_raw.get("target_ops", RegisterCampaignConfig().target_ops),
            slot_count=reg_raw.get("slot_count", 3),
            registers=reg_raw.get("registers", "all"),
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
    return FirmwareConfig(
        src_dir=raw.get("src_dir", FirmwareConfig.src_dir),
        platform=raw.get("platform", FirmwareConfig.platform),
        board=raw.get("board", FirmwareConfig.board),
        framework=raw.get("framework", FirmwareConfig.framework),
        monitor_speed=raw.get("monitor_speed", FirmwareConfig.monitor_speed),
        extra_build_flags=raw.get("extra_build_flags", []),
    )


def _parse_serial(raw: Optional[Dict[str, Any]]) -> SerialConfig:
    if raw is None:
        return SerialConfig()
    return SerialConfig(
        port=raw.get("port", "COM3"),
        baud=raw.get("baud", 460800),
        boot_timeout_s=raw.get("boot_timeout_s", 30),
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
