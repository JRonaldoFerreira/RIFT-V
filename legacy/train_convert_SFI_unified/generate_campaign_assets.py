#!/usr/bin/env python3
"""Generate FI populations and firmware headers for register or memory campaigns."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow.lite.python import schema_py_generated as schema_fb

TRAINABLE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}
REGISTERS: List[Tuple[int, str, str]] = [
    (0, "ra", "link"),
    (1, "sp", "stack"),
    (2, "gp", "pointer"),
    (3, "tp", "pointer"),
    (4, "t0", "temporary"),
    (5, "t1", "temporary"),
    (6, "t2", "temporary"),
    (7, "s0", "saved"),
    (8, "s1", "saved"),
    (9, "a0", "argument"),
    (10, "a1", "argument"),
    (11, "a2", "argument"),
    (12, "a3", "argument"),
    (13, "a4", "argument"),
    (14, "a5", "argument"),
    (15, "a6", "argument"),
    (16, "a7", "argument"),
    (17, "s2", "saved"),
    (18, "s3", "saved"),
    (19, "s4", "saved"),
    (20, "s5", "saved"),
    (21, "s6", "saved"),
    (22, "s7", "saved"),
    (23, "s8", "saved"),
    (24, "s9", "saved"),
    (25, "s10", "saved"),
    (26, "s11", "saved"),
    (27, "t3", "temporary"),
    (28, "t4", "temporary"),
    (29, "t5", "temporary"),
    (30, "t6", "temporary"),
]

CAMPAIGN_HEADER_FILES: Dict[str, List[str]] = {
    "register": ["model_data.h", "mnist_subset.h", "register_fault_list.h", "register_operator_map.h"],
    "memory": ["model_data.h", "mnist_subset.h", "fault_list.h", "tensor_map.h"],
}


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


def z_value(confidence: float) -> float:
    alpha = 1.0 - confidence
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def finite_population_sample_size(population_size: int, confidence: float, margin: float, p: float) -> int:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if not (0.0 < margin < 1.0):
        raise ValueError("margin must be in (0, 1)")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    z = z_value(confidence)
    n0 = (z**2) * p * (1.0 - p) / (margin**2)
    n = (population_size * n0) / (population_size + n0 - 1.0)
    return int(math.ceil(n))


def sample_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n == 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=seed).sort_values("fault_id").reset_index(drop=True)


def get_model(buf: bytes):
    return schema_fb.Model.GetRootAsModel(buf, 0)


def decode_name(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    return str(raw)


def builtin_name_map() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for attr in dir(schema_fb.BuiltinOperator):
        if attr.startswith("_"):
            continue
        value = getattr(schema_fb.BuiltinOperator, attr)
        if isinstance(value, int):
            mapping[value] = attr
    return mapping


def collect_operator_instances(model) -> List[OperatorInstance]:
    subgraph = model.Subgraphs(0)
    opcode_name = builtin_name_map()
    operators: List[OperatorInstance] = []

    for op_index in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(op_index)
        opcode_index = op.OpcodeIndex()
        opcode = model.OperatorCodes(opcode_index)
        op_name = opcode_name.get(opcode.BuiltinCode(), f"BUILTIN_{opcode.BuiltinCode()}")
        operators.append(
            OperatorInstance(
                op_index=op_index,
                op_name=op_name,
                num_inputs=op.InputsLength(),
                num_outputs=op.OutputsLength(),
            )
        )

    return operators


def buffer_data_info(buffer_obj, model_bytes: bytes) -> Tuple[int, int]:
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


def tensor_shape_as_text(tensor_obj) -> str:
    dims = [int(tensor_obj.Shape(i)) for i in range(tensor_obj.ShapeLength())]
    return "x".join(str(d) for d in dims)


def collect_tensor_records(model, model_bytes: bytes) -> Dict[int, TensorRecord]:
    subgraph = model.Subgraphs(0)
    records: Dict[int, TensorRecord] = {}

    for tensor_index in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(tensor_index)
        buffer_index = int(tensor.Buffer())
        name = decode_name(tensor.Name())
        shape = tensor_shape_as_text(tensor)
        dtype = int(tensor.Type())

        if buffer_index < 0 or buffer_index >= model.BuffersLength():
            global_offset, length_bytes, is_constant = -1, 0, False
        else:
            buffer_obj = model.Buffers(buffer_index)
            global_offset, length_bytes = buffer_data_info(buffer_obj, model_bytes)
            is_constant = length_bytes > 0 and buffer_index != 0

        records[tensor_index] = TensorRecord(
            tensor_index=tensor_index,
            name=name,
            buffer_index=buffer_index,
            global_offset=global_offset,
            length_bytes=length_bytes,
            shape=shape,
            dtype=dtype,
            is_constant=is_constant,
        )

    return records


def collect_tensor_uses(model, tensor_records: Dict[int, TensorRecord]) -> List[TensorUse]:
    op_names = builtin_name_map()
    subgraph = model.Subgraphs(0)
    uses: List[TensorUse] = []

    for op_index in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(op_index)
        opcode_index = int(op.OpcodeIndex())
        opcode = model.OperatorCodes(opcode_index)
        builtin_code = int(opcode.BuiltinCode())
        op_name = op_names.get(builtin_code, f"OP_{builtin_code}")

        for input_idx in range(op.InputsLength()):
            tensor_index = int(op.Inputs(input_idx))
            if tensor_index < 0:
                continue
            tensor_rec = tensor_records[tensor_index]
            uses.append(
                TensorUse(
                    op_index=op_index,
                    op_name=op_name,
                    tensor_index=tensor_index,
                    tensor_name=tensor_rec.name,
                    io_role="input",
                )
            )

        for output_idx in range(op.OutputsLength()):
            tensor_index = int(op.Outputs(output_idx))
            if tensor_index < 0:
                continue
            tensor_rec = tensor_records[tensor_index]
            uses.append(
                TensorUse(
                    op_index=op_index,
                    op_name=op_name,
                    tensor_index=tensor_index,
                    tensor_name=tensor_rec.name,
                    io_role="output",
                )
            )

    return uses


def format_c_array_u8(name: str, data: bytes, cols: int = 12) -> str:
    lines = [f"const unsigned char {name}[] = {{"]
    for i in range(0, len(data), cols):
        chunk = data[i : i + cols]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {line},")
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(data)}u;")
    return "\n".join(lines)


def model_header(tflite_path: Path) -> str:
    data = tflite_path.read_bytes()
    body = format_c_array_u8("g_model_data", data)
    return "\n".join(["#pragma once", "#include <stdint.h>", "", body, ""])


def subset_header(npz_path: Path) -> str:
    subset = np.load(npz_path)
    images = subset["images"].astype(np.float32)
    labels = subset["labels"].astype(np.int32)
    test_indices = subset["test_indices"].astype(np.int32)

    n, h, w, c = images.shape
    flat = images.reshape(-1)

    img_vals = ",\n  ".join(f"{v:.8f}f" for v in flat)
    label_vals = ", ".join(str(int(v)) for v in labels)
    index_vals = ", ".join(str(int(v)) for v in test_indices)

    return "\n".join(
        [
            "#pragma once",
            "#include <stdint.h>",
            "",
            f"static const uint32_t kNumCampaignImages = {n}u;",
            f"static const uint32_t kImageHeight = {h}u;",
            f"static const uint32_t kImageWidth = {w}u;",
            f"static const uint32_t kImageChannels = {c}u;",
            "",
            f"static const float g_mnist_images[{flat.size}] = {{",
            f"  {img_vals}",
            "};",
            "",
            f"static const int32_t g_mnist_labels[{n}] = {{{label_vals}}};",
            f"static const int32_t g_mnist_test_indices[{n}] = {{{index_vals}}};",
            "",
        ]
    )


def register_fault_list_header(selected_faults_csv: Path) -> str:
    df = pd.read_csv(selected_faults_csv)
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "  int32_t fault_id;",
        "  uint32_t op_index;",
        "  uint8_t time_slot_index;",
        "  uint8_t time_slot_count;",
        "  uint8_t register_id;",
        "  uint8_t bit_index;",
        "} register_fault_t;",
        "",
        "static const register_fault_t g_fault_list[] = {",
    ]

    for _, row in df.iterrows():
        lines.append(
            "  {"
            f"{int(row['fault_id'])}, "
            f"{int(row['op_index'])}u, "
            f"{int(row['time_slot_index'])}u, "
            f"{int(row['time_slot_count'])}u, "
            f"{int(row['register_id'])}u, "
            f"{int(row['bit_index'])}u"
            "},"
        )

    lines.extend(["};", f"static const uint32_t g_fault_list_count = {len(df)}u;", ""])
    return "\n".join(lines)


def register_operator_map_header(operators_csv: Path) -> str:
    df = pd.read_csv(operators_csv)
    df = df[["op_index", "op_name"]].drop_duplicates().sort_values(["op_index", "op_name"])

    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "  uint32_t op_index;",
        "  const char* op_name;",
        "} register_op_name_t;",
        "",
        "static const register_op_name_t g_register_op_names[] = {",
    ]

    for _, row in df.iterrows():
        op_name = str(row["op_name"]).replace('"', '\\"')
        lines.append(f'  {{{int(row["op_index"])}u, "{op_name}"}},')

    lines.extend(["};", f"static const uint32_t g_register_op_names_count = {len(df)}u;", ""])
    return "\n".join(lines)


def tensor_map_header(records: List[TensorRecord]) -> str:
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "  const char* name;",
        "  uint32_t tensor_index;",
        "  uint32_t offset;",
        "  uint32_t length;",
        "} tensor_span_t;",
        "",
        "static const tensor_span_t g_tensor_map[] = {",
    ]

    constant_count = 0
    for rec in records:
        if not rec.is_constant or rec.length_bytes <= 0:
            continue
        safe_name = rec.name.replace('"', '\\"')
        lines.append(f'  {{"{safe_name}", {rec.tensor_index}u, {rec.global_offset}u, {rec.length_bytes}u}},')
        constant_count += 1

    lines.extend(["};", f"static const uint32_t g_tensor_map_count = {constant_count}u;", ""])
    return "\n".join(lines)


def memory_fault_list_header(selected_faults_csv: Path) -> str:
    df = pd.read_csv(selected_faults_csv)
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        "typedef struct {",
        "  int32_t fault_id;",
        "  uint32_t op_index;",
        "  uint32_t tensor_index;",
        "  uint32_t tensor_offset;",
        "  uint32_t tensor_length;",
        "  uint32_t byte_in_tensor;",
        "  uint32_t global_byte_offset;",
        "  uint8_t bit_index;",
        "} memory_fault_t;",
        "",
        "static const memory_fault_t g_fault_list[] = {",
    ]

    for _, row in df.iterrows():
        lines.append(
            "  {"
            f"{int(row['fault_id'])}, "
            f"{int(row['op_index'])}u, "
            f"{int(row['tensor_index'])}u, "
            f"{int(row['tensor_global_offset'])}u, "
            f"{int(row['tensor_length_bytes'])}u, "
            f"{int(row['byte_in_tensor'])}u, "
            f"{int(row['global_byte_offset'])}u, "
            f"{int(row['bit_index'])}u"
            "},"
        )

    lines.extend(["};", f"static const uint32_t g_fault_list_count = {len(df)}u;", ""])
    return "\n".join(lines)


def write_common_headers(headers_dir: Path, tflite_path: Path, subset_npz: Path) -> None:
    headers_dir.mkdir(parents=True, exist_ok=True)
    (headers_dir / "model_data.h").write_text(model_header(tflite_path), encoding="utf-8")
    (headers_dir / "mnist_subset.h").write_text(subset_header(subset_npz), encoding="utf-8")


def install_campaign_headers(campaign: str, headers_dir: Path, firmware_src: Path) -> List[Path]:
    if campaign not in CAMPAIGN_HEADER_FILES:
        raise ValueError(f"Unsupported campaign: {campaign}")
    if not firmware_src.exists() or not firmware_src.is_dir():
        raise FileNotFoundError(
            f"Firmware src directory not found: {firmware_src}. Use --firmware-src to set a valid path."
        )

    installed: List[Path] = []
    for header_name in CAMPAIGN_HEADER_FILES[campaign]:
        source = headers_dir / header_name
        if not source.exists():
            raise FileNotFoundError(f"Generated header not found: {source}")
        target = firmware_src / header_name
        shutil.copy2(source, target)
        installed.append(target)

    return installed


def run_register_campaign(
    tflite_path: Path,
    subset_npz: Path,
    out_dir: Path,
    seed: int,
    confidence: float,
    margin: float,
    p_prior: float,
    slot_count: int,
) -> Path:
    model_bytes = tflite_path.read_bytes()
    fb_model = get_model(model_bytes)

    operators = collect_operator_instances(fb_model)
    ops_df = pd.DataFrame([asdict(op) for op in operators]).sort_values(["op_index"]).reset_index(drop=True)
    ops_df.to_csv(out_dir / "all_operator_instances.csv", index=False)

    trainable_ops_df = (
        ops_df[ops_df["op_name"].isin(TRAINABLE_OPS)].copy().sort_values(["op_index"]).reset_index(drop=True)
    )
    if trainable_ops_df.empty:
        raise RuntimeError("No trainable operators selected for RegisterFI")

    trainable_ops_df.to_csv(out_dir / "trainable_operator_instances.csv", index=False)

    population_rows: List[Dict[str, object]] = []
    fault_id = 0
    for _, op in trainable_ops_df.iterrows():
        for time_slot_index in range(slot_count):
            for register_id, register_name, register_class in REGISTERS:
                for bit_index in range(32):
                    population_rows.append(
                        {
                            "fault_id": fault_id,
                            "op_index": int(op["op_index"]),
                            "op_name": str(op["op_name"]),
                            "time_slot_index": int(time_slot_index),
                            "time_slot_count": int(slot_count),
                            "register_id": int(register_id),
                            "register_name": str(register_name),
                            "register_class": str(register_class),
                            "bit_index": int(bit_index),
                        }
                    )
                    fault_id += 1

    population_df = pd.DataFrame(population_rows)
    population_csv = out_dir / "register_population.csv"
    population_df.to_csv(population_csv, index=False)

    population_by_operator = (
        population_df.groupby(["op_index", "op_name"], as_index=False)
        .size()
        .rename(columns={"size": "population_size"})
        .sort_values(["op_index"])
    )
    population_by_operator.to_csv(out_dir / "register_population_by_operator.csv", index=False)

    population_meta = {
        "population_definition": "N_R={(op_k, time_slot_index, register_id, bit_index)} with normalized timing slots calibrated on-device during golden phase.",
        "slot_count": slot_count,
        "num_operator_instances": int(len(trainable_ops_df)),
        "num_registers": len(REGISTERS),
        "bits_per_register": 32,
        "population_size": int(len(population_df)),
    }
    (out_dir / "register_population.meta.json").write_text(json.dumps(population_meta, indent=2), encoding="utf-8")

    rng = np.random.default_rng(seed)
    selected_parts = []
    strata_summary = []

    grouped = population_df.groupby(["op_index", "op_name"], dropna=False, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_size = len(group_df)
        sample_size = finite_population_sample_size(group_size, confidence, margin, p_prior)
        group_seed = int(rng.integers(0, 2**31 - 1))
        sampled_group = sample_group(group_df, sample_size, group_seed)

        selected_parts.append(sampled_group)
        summary_row = {
            "population_size": group_size,
            "selected_faults": len(sampled_group),
            "estimated_total_inferences_for_100_images": int(len(sampled_group) * 100),
            "sampling_seed": group_seed,
            "op_index": group_key[0],
            "op_name": group_key[1],
        }
        strata_summary.append(summary_row)

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    selected_df = selected_df.sort_values(["op_index", "fault_id"]).reset_index(drop=True)
    strata_df = pd.DataFrame(strata_summary)

    selected_csv = out_dir / "selected_register_faults.csv"
    selected_df.to_csv(selected_csv, index=False)
    strata_df.to_csv(out_dir / "selected_register_faults_by_layer.csv", index=False)
    strata_df[["op_index", "op_name", "selected_faults"]].sort_values(["op_index"]).to_csv(
        out_dir / "selected_register_faults_by_operator.csv",
        index=False,
    )

    sample_meta = {
        "method": "one_step_layer_wise_registerfi",
        "population_size_total": int(len(population_df)),
        "selected_faults_total": int(len(selected_df)),
        "confidence": confidence,
        "margin": margin,
        "p": p_prior,
        "group_cols": ["op_index", "op_name"],
        "num_layers": int(len(strata_df)),
        "estimated_total_inferences_for_100_images": int(len(selected_df) * 100),
    }
    (selected_csv.with_suffix(".meta.json")).write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")

    headers_dir = out_dir / "headers_registerfi"
    write_common_headers(headers_dir, tflite_path, subset_npz)
    (headers_dir / "register_fault_list.h").write_text(register_fault_list_header(selected_csv), encoding="utf-8")
    (headers_dir / "register_operator_map.h").write_text(
        register_operator_map_header(out_dir / "trainable_operator_instances.csv"),
        encoding="utf-8",
    )
    return headers_dir


def run_memory_campaign(
    tflite_path: Path,
    subset_npz: Path,
    out_dir: Path,
    seed: int,
    confidence: float,
    margin: float,
    p_prior: float,
) -> Path:
    model_bytes = tflite_path.read_bytes()
    fb_model = get_model(model_bytes)

    tensor_records = collect_tensor_records(fb_model, model_bytes)
    uses = collect_tensor_uses(fb_model, tensor_records)

    tensor_df = pd.DataFrame([asdict(r) for r in tensor_records.values()])
    tensor_df = tensor_df[(tensor_df["is_constant"]) | (tensor_df["tensor_index"] == 0)]
    tensor_df.sort_values(["tensor_index"], inplace=True)
    tensor_df.to_csv(out_dir / "tensor_map.csv", index=False)

    (out_dir / "tensor_map.h").write_text(tensor_map_header(list(tensor_records.values())), encoding="utf-8")

    population_rows: List[Dict[str, object]] = []
    fault_id = 0
    for use in uses:
        if use.io_role != "input":
            continue
        if use.op_name not in TRAINABLE_OPS:
            continue

        tensor_rec = tensor_records[use.tensor_index]
        if not tensor_rec.is_constant or tensor_rec.length_bytes <= 0:
            continue

        for byte_in_tensor in range(tensor_rec.length_bytes):
            global_byte_offset = tensor_rec.global_offset + byte_in_tensor
            for bit_index in range(8):
                population_rows.append(
                    {
                        "fault_id": fault_id,
                        "op_index": use.op_index,
                        "op_name": use.op_name,
                        "tensor_index": use.tensor_index,
                        "tensor_name": use.tensor_name,
                        "tensor_buffer_index": tensor_rec.buffer_index,
                        "tensor_shape": tensor_rec.shape,
                        "tensor_length_bytes": tensor_rec.length_bytes,
                        "tensor_global_offset": tensor_rec.global_offset,
                        "byte_in_tensor": byte_in_tensor,
                        "global_byte_offset": global_byte_offset,
                        "bit_index": bit_index,
                    }
                )
                fault_id += 1

    population_df = pd.DataFrame(population_rows)
    population_df.to_csv(out_dir / "memory_population.csv", index=False)

    op_summary = (
        population_df.groupby(["op_index", "op_name", "tensor_index", "tensor_name"], as_index=False)
        .agg(population_items=("fault_id", "count"))
        .sort_values(["op_index", "tensor_index"])
    )
    op_summary.to_csv(out_dir / "population_by_operator.csv", index=False)

    rng = np.random.default_rng(seed)
    selected_parts = []
    strata_summary = []

    grouped = population_df.groupby(["op_index", "op_name"], dropna=False, sort=True)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_size = len(group_df)
        sample_size = finite_population_sample_size(group_size, confidence, margin, p_prior)
        group_seed = int(rng.integers(0, 2**31 - 1))
        sampled_group = sample_group(group_df, sample_size, group_seed)

        selected_parts.append(sampled_group)
        strata_summary.append(
            {
                "population_size": group_size,
                "selected_faults": len(sampled_group),
                "estimated_total_inferences_for_100_images": int(len(sampled_group) * 100),
                "sampling_seed": group_seed,
                "op_index": group_key[0],
                "op_name": group_key[1],
            }
        )

    if selected_parts:
        selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
        selected_df = selected_df.sort_values(["op_index", "fault_id"]).reset_index(drop=True)
    else:
        selected_df = population_df.iloc[0:0].copy()

    strata_df = pd.DataFrame(strata_summary)

    selected_csv = out_dir / "selected_faults.csv"
    selected_df.to_csv(selected_csv, index=False)
    strata_df.to_csv(out_dir / "selected_faults_by_layer.csv", index=False)
    strata_df[["op_index", "op_name", "selected_faults"]].sort_values(["op_index"]).to_csv(
        out_dir / "selected_faults_by_operator.csv",
        index=False,
    )

    sample_meta = {
        "method": "one_step_layer_wise_memoryfi",
        "population_size_total": int(len(population_df)),
        "selected_faults_total": int(len(selected_df)),
        "confidence": confidence,
        "margin": margin,
        "p": p_prior,
        "group_cols": ["op_index", "op_name"],
        "num_layers": int(len(strata_df)),
        "estimated_total_inferences_for_100_images": int(len(selected_df) * 100),
    }
    (selected_csv.with_suffix(".meta.json")).write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")

    headers_dir = out_dir / "headers_memoryfi"
    write_common_headers(headers_dir, tflite_path, subset_npz)
    (headers_dir / "fault_list.h").write_text(memory_fault_list_header(selected_csv), encoding="utf-8")
    (headers_dir / "tensor_map.h").write_text(tensor_map_header(list(tensor_records.values())), encoding="utf-8")
    return headers_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=["register", "memory"], required=True)
    parser.add_argument("--tflite", type=Path, required=True, help="Path to quantized .tflite model")
    parser.add_argument("--subset", type=Path, required=True, help="Path to mnist_subset.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--p-prior", type=float, default=0.5)
    parser.add_argument("--slot-count", type=int, default=3, help="Used only for register campaign")
    parser.add_argument(
        "--firmware-src",
        type=Path,
        default=None,
        help="Firmware src directory to receive generated headers (default: Register_FI/Register_FI/src)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Only generate artifacts, do not copy headers into firmware src",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    firmware_src = (args.firmware_src or (repo_root / "Register_FI" / "Register_FI" / "src")).resolve()

    if args.campaign == "register":
        headers_dir = run_register_campaign(
            tflite_path=args.tflite,
            subset_npz=args.subset,
            out_dir=args.out_dir,
            seed=args.seed,
            confidence=args.confidence,
            margin=args.margin,
            p_prior=args.p_prior,
            slot_count=args.slot_count,
        )
    else:
        headers_dir = run_memory_campaign(
            tflite_path=args.tflite,
            subset_npz=args.subset,
            out_dir=args.out_dir,
            seed=args.seed,
            confidence=args.confidence,
            margin=args.margin,
            p_prior=args.p_prior,
        )

    if not args.no_install:
        installed_headers = install_campaign_headers(args.campaign, headers_dir, firmware_src)
        print(f"Installed headers to: {firmware_src}")
        for header in installed_headers:
            print(f"  - {header.name}")

    print(f"Campaign '{args.campaign}' generated at: {args.out_dir}")


if __name__ == "__main__":
    main()
