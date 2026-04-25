"""Model introspection, fault-population generation, and statistical sampling."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from riftv.config import CampaignConfig, MultiInjectionConfig, SamplingConfig

# ---------------------------------------------------------------------------
# FlatBuffers schema import — supports tensorflow or standalone tflite package
# ---------------------------------------------------------------------------

try:
    # Prefer lightweight standalone package (no heavy TF DLL loading)
    from tflite.Model import Model as _TFLModel
    from tflite.BuiltinOperator import BuiltinOperator as _TFLBuiltin

    class _SchemaAdapter:
        BuiltinOperator = _TFLBuiltin

        @staticmethod
        def get_model(buf: bytes):
            return _TFLModel.GetRootAs(buf, 0)

except ImportError:
    try:
        from tensorflow.lite.python import schema_py_generated as _schema_fb

        class _SchemaAdapter:  # type: ignore[no-redef]
            BuiltinOperator = _schema_fb.BuiltinOperator

            @staticmethod
            def get_model(buf: bytes):
                return _schema_fb.Model.GetRootAsModel(buf, 0)

    except Exception as exc:
        raise ImportError(
            "Neither 'tflite' nor 'tensorflow' package found. "
            "Install one of: pip install tflite  OR  pip install tensorflow"
        ) from exc

_schema = _SchemaAdapter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TRAINABLE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}

RISCV_GPR_COUNT = 31  # x1–x31 (x0/zero is hardwired, not injectable)

# Indices match kRegisterDescriptors[] in firmware register_fi.cc (excframe path).
# Each entry: (register_id, name, register_class).
# The ISR accesses the named field in the RvExcFrame interrupt exception frame.
RISCV_REGISTERS: List[Tuple[int, str, str]] = [
    ( 0, "ra",  "caller_saved"),   # return address
    ( 1, "sp",  "special"),        # stack pointer  (note: not restored from frame on mret)
    ( 2, "gp",  "special"),        # global pointer
    ( 3, "tp",  "special"),        # thread pointer
    ( 4, "t0",  "caller_saved"),
    ( 5, "t1",  "caller_saved"),
    ( 6, "t2",  "caller_saved"),
    ( 7, "s0",  "callee_saved"),   # frame pointer / s0
    ( 8, "s1",  "callee_saved"),
    ( 9, "a0",  "caller_saved"),   # argument / return value
    (10, "a1",  "caller_saved"),
    (11, "a2",  "caller_saved"),
    (12, "a3",  "caller_saved"),
    (13, "a4",  "caller_saved"),
    (14, "a5",  "caller_saved"),
    (15, "a6",  "caller_saved"),
    (16, "a7",  "caller_saved"),
    (17, "s2",  "callee_saved"),
    (18, "s3",  "callee_saved"),
    (19, "s4",  "callee_saved"),
    (20, "s5",  "callee_saved"),
    (21, "s6",  "callee_saved"),
    (22, "s7",  "callee_saved"),
    (23, "s8",  "callee_saved"),
    (24, "s9",  "callee_saved"),
    (25, "s10", "callee_saved"),
    (26, "s11", "callee_saved"),
    (27, "t3",  "caller_saved"),
    (28, "t4",  "caller_saved"),
    (29, "t5",  "caller_saved"),
    (30, "t6",  "caller_saved"),
]


# ---------------------------------------------------------------------------
# Data classes for model introspection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorInstance:
    op_index: int
    op_name: str
    num_inputs: int
    num_outputs: int


@dataclass(frozen=True)
class TensorRecord:
    tensor_index: int
    name: str
    buffer_index: int
    global_offset: int
    length_bytes: int
    shape: str
    dtype: int
    is_constant: bool


@dataclass(frozen=True)
class TensorUse:
    op_index: int
    op_name: str
    tensor_index: int
    tensor_name: str
    io_role: str


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def z_value(confidence: float) -> float:
    alpha = 1.0 - confidence
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def finite_population_sample_size(
    population_size: int, confidence: float, margin: float, p: float,
) -> int:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    z = z_value(confidence)
    n0 = (z ** 2) * p * (1.0 - p) / (margin ** 2)
    n = (population_size * n0) / (population_size + n0 - 1.0)
    return int(math.ceil(n))


def sample_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n == 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=seed).sort_values("fault_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# FlatBuffers model parsing
# ---------------------------------------------------------------------------

def _builtin_name_map() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    bo = _schema.BuiltinOperator
    for attr in dir(bo):
        if attr.startswith("_"):
            continue
        value = getattr(bo, attr)
        if isinstance(value, int):
            mapping[value] = attr
    return mapping


def _decode_name(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    return str(raw)


def _buffer_data_info(buffer_obj, model_bytes: bytes) -> Tuple[int, int]:
    tab = buffer_obj._tab
    field_off = tab.Offset(4)
    if field_off != 0:
        vector_len = tab.VectorLen(field_off)
        vector_start = tab.Vector(field_off)
        if vector_len > 0:
            return int(vector_start), int(vector_len)

    data_np = buffer_obj.DataAsNumpy()
    if data_np is None or not hasattr(data_np, "__len__") or len(data_np) == 0:
        return -1, 0

    raw = bytes(data_np.tolist())
    idx = model_bytes.find(raw)
    if idx < 0:
        raise RuntimeError("Could not locate buffer bytes in serialized model")
    return idx, len(raw)


def _tensor_shape_as_text(tensor_obj) -> str:
    dims = [int(tensor_obj.Shape(i)) for i in range(tensor_obj.ShapeLength())]
    return "x".join(str(d) for d in dims)


class ModelIntrospector:
    """Parse a .tflite model and extract operators, tensors, and their relationships."""

    def __init__(self, tflite_path: Path):
        self.tflite_path = tflite_path
        self.model_bytes = tflite_path.read_bytes()
        self._fb = _schema.get_model(self.model_bytes)
        self._opcode_names = _builtin_name_map()

    def collect_operators(self) -> List[OperatorInstance]:
        subgraph = self._fb.Subgraphs(0)
        ops: List[OperatorInstance] = []
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode = self._fb.OperatorCodes(op.OpcodeIndex())
            name = self._opcode_names.get(opcode.BuiltinCode(), f"BUILTIN_{opcode.BuiltinCode()}")
            ops.append(OperatorInstance(i, name, op.InputsLength(), op.OutputsLength()))
        return ops

    def collect_tensor_records(self) -> Dict[int, TensorRecord]:
        subgraph = self._fb.Subgraphs(0)
        records: Dict[int, TensorRecord] = {}
        for idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(idx)
            buf_idx = int(tensor.Buffer())
            name = _decode_name(tensor.Name())
            shape = _tensor_shape_as_text(tensor)
            dtype = int(tensor.Type())

            if buf_idx < 0 or buf_idx >= self._fb.BuffersLength():
                g_off, length, is_const = -1, 0, False
            else:
                buf_obj = self._fb.Buffers(buf_idx)
                g_off, length = _buffer_data_info(buf_obj, self.model_bytes)
                is_const = length > 0 and buf_idx != 0

            records[idx] = TensorRecord(idx, name, buf_idx, g_off, length, shape, dtype, is_const)
        return records

    def collect_tensor_uses(self, tensor_records: Dict[int, TensorRecord]) -> List[TensorUse]:
        subgraph = self._fb.Subgraphs(0)
        uses: List[TensorUse] = []
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode = self._fb.OperatorCodes(int(op.OpcodeIndex()))
            op_name = self._opcode_names.get(int(opcode.BuiltinCode()), f"OP_{opcode.BuiltinCode()}")

            for j in range(op.InputsLength()):
                ti = int(op.Inputs(j))
                if ti < 0:
                    continue
                rec = tensor_records[ti]
                uses.append(TensorUse(i, op_name, ti, rec.name, "input"))

            for j in range(op.OutputsLength()):
                ti = int(op.Outputs(j))
                if ti < 0:
                    continue
                rec = tensor_records[ti]
                uses.append(TensorUse(i, op_name, ti, rec.name, "output"))
        return uses


# ---------------------------------------------------------------------------
# Population generators
# ---------------------------------------------------------------------------

def generate_register_population(
    introspector: ModelIntrospector,
    target_ops: Optional[List[str]] = None,
    slot_count: int = 3,
    registers: Union[str, List[str]] = "all",
    bits: Union[str, List[int]] = "all",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate full register-FI population.

    Parameters
    ----------
    registers : str or list of str
        ``"all"`` to use every GPR (x1–x31), or a list of register names
        (e.g. ``["a0", "a1", "t0"]``) to restrict the population.
    bits : str or list of int
        ``"all"`` to flip any of the 32 bits (0–31), or a list of specific
        bit indices (e.g. ``[0, 7, 15, 31]``) to restrict the population.

    Returns (population_df, trainable_ops_df, all_ops_df).
    """
    if target_ops is None:
        target_ops = list(DEFAULT_TRAINABLE_OPS)
    target_set = set(target_ops)

    # Resolve register subset
    if isinstance(registers, list):
        reg_name_set = set(registers)
        active_regs = [(rid, rname, rcls) for rid, rname, rcls in RISCV_REGISTERS if rname in reg_name_set]
        if not active_regs:
            raise ValueError(f"None of the requested registers {registers} matched RISCV_REGISTERS.")
    else:
        active_regs = RISCV_REGISTERS

    operators = introspector.collect_operators()
    all_ops_df = pd.DataFrame([asdict(op) for op in operators]).sort_values("op_index").reset_index(drop=True)

    trainable_ops_df = all_ops_df[all_ops_df["op_name"].isin(target_set)].copy().reset_index(drop=True)
    if trainable_ops_df.empty:
        raise RuntimeError(f"No operators matching {target_ops} found in model.")

    rows: List[Dict[str, object]] = []
    fault_id = 0
    for _, op in trainable_ops_df.iterrows():
        for ts in range(slot_count):
            for reg_id, reg_name, reg_class in active_regs:
                bit_range = range(32) if bits == "all" else bits
                for bit in bit_range:
                    rows.append({
                        "fault_id": fault_id,
                        "op_index": int(op["op_index"]),
                        "op_name": str(op["op_name"]),
                        "time_slot_index": ts,
                        "time_slot_count": slot_count,
                        "register_id": reg_id,
                        "register_name": reg_name,
                        "register_class": reg_class,
                        "bit_index": bit,
                    })
                    fault_id += 1

    population_df = pd.DataFrame(rows)
    return population_df, trainable_ops_df, all_ops_df


def generate_memory_population(
    introspector: ModelIntrospector,
    target_ops: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[int, TensorRecord]]:
    """Generate full memory-FI population.

    Returns (population_df, tensor_records).
    """
    if target_ops is None:
        target_ops = list(DEFAULT_TRAINABLE_OPS)
    target_set = set(target_ops)

    tensor_records = introspector.collect_tensor_records()
    uses = introspector.collect_tensor_uses(tensor_records)

    rows: List[Dict[str, object]] = []
    fault_id = 0
    for use in uses:
        if use.io_role != "input" or use.op_name not in target_set:
            continue
        rec = tensor_records[use.tensor_index]
        if not rec.is_constant or rec.length_bytes <= 0:
            continue

        for byte_in_tensor in range(rec.length_bytes):
            g_off = rec.global_offset + byte_in_tensor
            for bit in range(8):
                rows.append({
                    "fault_id": fault_id,
                    "op_index": use.op_index,
                    "op_name": use.op_name,
                    "tensor_index": use.tensor_index,
                    "tensor_name": use.tensor_name,
                    "tensor_buffer_index": rec.buffer_index,
                    "tensor_shape": rec.shape,
                    "tensor_length_bytes": rec.length_bytes,
                    "tensor_global_offset": rec.global_offset,
                    "byte_in_tensor": byte_in_tensor,
                    "global_byte_offset": g_off,
                    "bit_index": bit,
                })
                fault_id += 1

    population_df = pd.DataFrame(rows)
    return population_df, tensor_records


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    population_df: pd.DataFrame,
    sampling: SamplingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply stratified sampling to a population DataFrame.

    Returns (selected_df, strata_summary_df).
    """
    rng = np.random.default_rng(sampling.seed)
    parts: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []

    grouped = population_df.groupby(sampling.group_by, dropna=False, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        n_pop = len(group_df)
        n_sample = finite_population_sample_size(n_pop, sampling.confidence, sampling.margin, sampling.p_prior)
        gseed = int(rng.integers(0, 2 ** 31 - 1))
        sampled = sample_group(group_df, n_sample, gseed)
        parts.append(sampled)

        row: Dict[str, object] = {"population_size": n_pop, "selected_faults": len(sampled), "sampling_seed": gseed}
        for i, col in enumerate(sampling.group_by):
            row[col] = group_key[i]
        summaries.append(row)

    if parts:
        selected_df = pd.concat(parts, ignore_index=True).sort_values(
            [sampling.group_by[0], "fault_id"]
        ).reset_index(drop=True)
    else:
        selected_df = population_df.iloc[0:0].copy()

    strata_df = pd.DataFrame(summaries)
    return selected_df, strata_df


# ---------------------------------------------------------------------------
# Multi-injection group generation
# ---------------------------------------------------------------------------

def _group_has_collision(
    indices: tuple,
    selected_df: pd.DataFrame,
    no_collide_columns: Optional[List[str]],
    no_timing_conflict: bool,
) -> bool:
    """Return True if a candidate group should be rejected.

    Checks:
    1. ``no_collide_columns`` — no two faults may share the same tuple of
       values in these columns (e.g. same ``global_byte_offset`` for memory,
       or same ``(register_id, bit_index)`` for register).
    2. ``no_timing_conflict`` — (register only) no two faults targeting the
       same ``op_index`` with different ``time_slot_index`` values.
    """
    if no_collide_columns is None and not no_timing_conflict:
        return False

    subset = selected_df.iloc[list(indices)]

    if no_collide_columns is not None:
        collision_keys = subset[no_collide_columns].apply(tuple, axis=1)
        if collision_keys.duplicated().any():
            return True

    if no_timing_conflict and "op_index" in subset.columns and "time_slot_index" in subset.columns:
        for _, grp in subset.groupby("op_index"):
            if grp["time_slot_index"].nunique() > 1:
                return True

    return False


def generate_fault_groups(
    selected_df: pd.DataFrame,
    mi_cfg: MultiInjectionConfig,
    no_collide_columns: Optional[List[str]] = None,
    no_timing_conflict: bool = False,
) -> pd.DataFrame:
    """Generate random groups of simultaneous faults from selected faults.

    Parameters
    ----------
    selected_df : pd.DataFrame
        The pool of individually-selected faults.
    mi_cfg : MultiInjectionConfig
        Multi-injection parameters (count, num_groups, seed).
    no_collide_columns : list of str, optional
        Column names whose combined values must be unique within a group.
        For **memory** campaigns pass ``["global_byte_offset"]`` to prevent
        two faults from targeting the same byte (which causes save/restore
        corruption).  For **register** campaigns pass
        ``["register_id", "bit_index"]`` to prevent XOR cancellation.
    no_timing_conflict : bool
        If True (register campaigns), reject groups where two faults share
        the same ``op_index`` but have different ``time_slot_index`` values,
        which would require simultaneous timers that the firmware cannot
        honour within a single operator invocation.

    Returns a DataFrame with columns:
        group_id, pool_index_0, pool_index_1, ..., pool_index_{count-1}
    where each pool_index_i is a positional index into *selected_df*.
    """
    count = mi_cfg.count
    num_groups = mi_cfg.num_groups
    seed = mi_cfg.seed
    n_faults = len(selected_df)

    if n_faults < count:
        raise ValueError(
            f"Cannot create groups of {count} faults from only {n_faults} selected faults."
        )

    rng = np.random.default_rng(seed)
    seen: set = set()
    rows: List[Dict[str, object]] = []
    group_id = 0
    max_attempts = num_groups * 100
    attempts = 0

    while len(rows) < num_groups and attempts < max_attempts:
        indices = tuple(sorted(rng.choice(n_faults, size=count, replace=False)))
        attempts += 1
        if indices in seen:
            continue
        seen.add(indices)
        if _group_has_collision(indices, selected_df, no_collide_columns, no_timing_conflict):
            continue
        row: Dict[str, object] = {"group_id": group_id}
        for i, idx in enumerate(indices):
            row[f"pool_index_{i}"] = int(idx)
        rows.append(row)
        group_id += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-level pipeline: prepare campaign artifacts
# ---------------------------------------------------------------------------

@dataclass
class CampaignArtifacts:
    """All artifacts produced by the population preparation step."""
    campaign_type: str
    population_csv: Path
    selected_csv: Path
    strata_csv: Path
    meta_json: Path
    headers_dir: Path
    all_csvs: List[Path]
    groups_csv: Optional[Path] = None
    multi_injection: Optional[MultiInjectionConfig] = None


def prepare_register_campaign(
    introspector: ModelIntrospector,
    campaign: CampaignConfig,
    sampling: SamplingConfig,
    out_dir: Path,
) -> CampaignArtifacts:
    """Generate all register-FI campaign artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_cfg = campaign.register

    target_ops = reg_cfg.target_ops if reg_cfg else list(DEFAULT_TRAINABLE_OPS)
    slot_count = reg_cfg.slot_count if reg_cfg else 3
    registers = reg_cfg.registers if reg_cfg else "all"

    bits = reg_cfg.bits if reg_cfg else "all"

    pop_df, trainable_df, all_ops_df = generate_register_population(
        introspector, target_ops, slot_count, registers=registers, bits=bits,
    )

    all_ops_csv = out_dir / "all_operator_instances.csv"
    trainable_csv = out_dir / "trainable_operator_instances.csv"
    pop_csv = out_dir / "register_population.csv"

    all_ops_df.to_csv(all_ops_csv, index=False)
    trainable_df.to_csv(trainable_csv, index=False)
    pop_df.to_csv(pop_csv, index=False)

    pop_by_op = (
        pop_df.groupby(["op_index", "op_name"], as_index=False)
        .size().rename(columns={"size": "population_size"}).sort_values("op_index")
    )
    pop_by_op.to_csv(out_dir / "register_population_by_operator.csv", index=False)

    if isinstance(registers, list):
        active_reg_count = len(registers)
        active_reg_names = registers
    else:
        active_reg_count = len(RISCV_REGISTERS)
        active_reg_names = [name for _, name, _ in RISCV_REGISTERS]
    if isinstance(bits, list):
        active_bit_count = len(bits)
        active_bits = bits
    else:
        active_bit_count = 32
        active_bits = list(range(32))
    pop_meta = {
        "population_definition": "N_R={(op_k, time_slot_index, register_id, bit_index)}",
        "slot_count": slot_count,
        "num_operators": int(len(trainable_df)),
        "num_registers": active_reg_count,
        "registers": active_reg_names,
        "bits_per_register": active_bit_count,
        "bits": active_bits,
        "population_size": int(len(pop_df)),
    }
    (out_dir / "register_population.meta.json").write_text(json.dumps(pop_meta, indent=2), encoding="utf-8")

    selected_df, strata_df = stratified_sample(pop_df, sampling)

    selected_csv = out_dir / "selected_register_faults.csv"
    strata_csv = out_dir / "selected_register_faults_by_layer.csv"
    meta_json = selected_csv.with_suffix(".meta.json")

    selected_df.to_csv(selected_csv, index=False)
    strata_df.to_csv(strata_csv, index=False)

    sample_meta: Dict[str, object] = {
        "method": "stratified_register",
        "population_size_total": int(len(pop_df)),
        "selected_faults_total": int(len(selected_df)),
        "confidence": sampling.confidence,
        "margin": sampling.margin,
        "p": sampling.p_prior,
    }

    # Multi-injection group generation
    mi_cfg = campaign.multi_injection
    groups_csv: Optional[Path] = None
    if mi_cfg and mi_cfg.enabled:
        groups_df = generate_fault_groups(
            selected_df, mi_cfg,
            no_collide_columns=["register_id", "bit_index"],
        )
        groups_csv = out_dir / "register_fault_groups.csv"
        groups_df.to_csv(groups_csv, index=False)
        sample_meta["multi_injection"] = {
            "enabled": True,
            "count": mi_cfg.count,
            "num_groups": int(len(groups_df)),
            "seed": mi_cfg.seed,
        }

    meta_json.write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")

    headers_dir = out_dir / "headers"
    all_csvs = [all_ops_csv, trainable_csv, pop_csv, selected_csv, strata_csv]
    if groups_csv is not None:
        all_csvs.append(groups_csv)

    return CampaignArtifacts(
        campaign_type="register",
        population_csv=pop_csv,
        selected_csv=selected_csv,
        strata_csv=strata_csv,
        meta_json=meta_json,
        headers_dir=headers_dir,
        all_csvs=all_csvs,
        groups_csv=groups_csv,
        multi_injection=mi_cfg,
    )


def prepare_memory_campaign(
    introspector: ModelIntrospector,
    campaign: CampaignConfig,
    sampling: SamplingConfig,
    out_dir: Path,
) -> CampaignArtifacts:
    """Generate all memory-FI campaign artifacts.

    Faults target constant tensor data (model weights) in the serialised
    model blob.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mem_cfg = campaign.memory

    target_ops = mem_cfg.target_ops if mem_cfg else list(DEFAULT_TRAINABLE_OPS)

    pop_df, tensor_records = generate_memory_population(introspector, target_ops)

    pop_csv = out_dir / "memory_population.csv"
    pop_df.to_csv(pop_csv, index=False)

    tensor_records_dict = tensor_records if isinstance(tensor_records, dict) else {r.tensor_index: r for r in tensor_records.values()} if hasattr(tensor_records, 'values') else tensor_records
    if isinstance(tensor_records_dict, dict):
        tensor_df = pd.DataFrame([asdict(r) for r in tensor_records_dict.values()])
    else:
        tensor_df = pd.DataFrame()
    if not tensor_df.empty:
        tensor_df = tensor_df[(tensor_df["is_constant"]) | (tensor_df["tensor_index"] == 0)]
        tensor_df.sort_values("tensor_index", inplace=True)
    tensor_df.to_csv(out_dir / "tensor_map.csv", index=False)

    group_cols = ["op_index", "op_name", "tensor_index", "tensor_name"]
    available_group_cols = [c for c in group_cols if c in pop_df.columns]
    if available_group_cols and "fault_id" in pop_df.columns:
        valid_pop = pop_df[pop_df["op_index"] >= 0] if "op_index" in pop_df.columns else pop_df
        if not valid_pop.empty:
            op_summary = (
                valid_pop.groupby(available_group_cols, as_index=False)
                .agg(population_items=("fault_id", "count"))
                .sort_values(available_group_cols[:2] if len(available_group_cols) >= 2 else available_group_cols)
            )
            op_summary.to_csv(out_dir / "population_by_operator.csv", index=False)

    selected_df, strata_df = stratified_sample(pop_df, sampling)

    selected_csv = out_dir / "selected_faults.csv"
    strata_csv = out_dir / "selected_faults_by_layer.csv"
    meta_json = selected_csv.with_suffix(".meta.json")

    selected_df.to_csv(selected_csv, index=False)
    strata_df.to_csv(strata_csv, index=False)

    sample_meta: Dict[str, object] = {
        "method": "stratified_memory",
        "population_size_total": int(len(pop_df)),
        "selected_faults_total": int(len(selected_df)),
        "confidence": sampling.confidence,
        "margin": sampling.margin,
        "p": sampling.p_prior,
    }

    mi_cfg = campaign.multi_injection
    groups_csv: Optional[Path] = None
    if mi_cfg and mi_cfg.enabled:
        groups_df = generate_fault_groups(
            selected_df, mi_cfg,
            no_collide_columns=["global_byte_offset"],
        )
        groups_csv = out_dir / "memory_fault_groups.csv"
        groups_df.to_csv(groups_csv, index=False)
        sample_meta["multi_injection"] = {
            "enabled": True,
            "count": mi_cfg.count,
            "num_groups": int(len(groups_df)),
            "seed": mi_cfg.seed,
        }

    meta_json.write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")

    headers_dir = out_dir / "headers"
    all_csvs = [pop_csv, selected_csv, strata_csv]
    if groups_csv is not None:
        all_csvs.append(groups_csv)

    return CampaignArtifacts(
        campaign_type="memory",
        population_csv=pop_csv,
        selected_csv=selected_csv,
        strata_csv=strata_csv,
        meta_json=meta_json,
        headers_dir=headers_dir,
        all_csvs=all_csvs,
        groups_csv=groups_csv,
        multi_injection=mi_cfg,
    )
