"""Generate C/C++ header files for firmware consumption."""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tflmfi.population import TensorRecord


# ---------------------------------------------------------------------------
# Header file lists per campaign
# ---------------------------------------------------------------------------

CAMPAIGN_HEADERS: Dict[str, List[str]] = {
    "register": ["model_data.h", "mnist_subset.h", "register_fault_list.h", "register_operator_map.h"],
    "memory": ["model_data.h", "mnist_subset.h", "fault_list.h", "tensor_map.h"],
}


# ---------------------------------------------------------------------------
# Utility: C array formatting
# ---------------------------------------------------------------------------

def format_c_array_u8(name: str, data: bytes, cols: int = 12) -> str:
    lines = [f"const unsigned char {name}[] = {{"]
    for i in range(0, len(data), cols):
        chunk = data[i:i + cols]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {line},")
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(data)}u;")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual header generators
# ---------------------------------------------------------------------------

def generate_model_header(tflite_path: Path) -> str:
    data = tflite_path.read_bytes()
    body = format_c_array_u8("g_model_data", data)
    return "\n".join(["#pragma once", "#include <stdint.h>", "", body, ""])


def generate_subset_header(npz_path: Path) -> str:
    subset = np.load(npz_path)
    images = subset["images"].astype(np.float32)
    labels = subset["labels"].astype(np.int32)
    test_indices = subset["test_indices"].astype(np.int32)

    n, h, w, c = images.shape
    flat = images.reshape(-1)

    img_vals = ",\n  ".join(f"{v:.8f}f" for v in flat)
    label_vals = ", ".join(str(int(v)) for v in labels)
    index_vals = ", ".join(str(int(v)) for v in test_indices)

    return "\n".join([
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
    ])


def generate_register_fault_list_header(
    selected_csv: Path,
    groups_csv: Optional[Path] = None,
    multi_count: int = 0,
) -> str:
    df = pd.read_csv(selected_csv)
    is_multi = groups_csv is not None and multi_count > 1

    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    if is_multi:
        lines.append(f"#define MULTI_FI_MAX_COUNT {multi_count}")
        lines.append("")

    lines.extend([
        "typedef struct {",
        "  int32_t fault_id;",
        "  uint32_t op_index;",
        "  uint8_t time_slot_index;",
        "  uint8_t time_slot_count;",
        "  uint8_t register_id;",
        "  uint8_t bit_index;",
        "} register_fault_t;",
        "",
    ])

    pool_name = "g_fault_pool" if is_multi else "g_fault_list"
    lines.append(f"static const register_fault_t {pool_name}[] = {{")

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

    lines.extend(["};", f"static const uint32_t {pool_name}_count = {len(df)}u;", ""])

    if is_multi:
        groups_df = pd.read_csv(groups_csv)
        pool_idx_cols = [c for c in groups_df.columns if c.startswith("pool_index_")]
        lines.extend([
            "typedef struct {",
            "  int32_t group_id;",
            "  uint8_t fault_count;",
            f"  uint16_t pool_indices[MULTI_FI_MAX_COUNT];",
            "} fault_group_t;",
            "",
            "static const fault_group_t g_fault_groups[] = {",
        ])
        for _, grow in groups_df.iterrows():
            indices = ", ".join(str(int(grow[c])) for c in pool_idx_cols)
            lines.append(f"  {{{int(grow['group_id'])}, {len(pool_idx_cols)}u, {{{indices}}}}},")
        lines.extend(["};", f"static const uint32_t g_fault_group_count = {len(groups_df)}u;", ""])

        lines.append("#define g_fault_list g_fault_pool")
        lines.append("#define g_fault_list_count g_fault_pool_count")
        lines.append("")

    return "\n".join(lines)


def generate_register_operator_map_header(operators_csv: Path) -> str:
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


def generate_tensor_map_header(records: List[TensorRecord]) -> str:
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

    count = 0
    for rec in records:
        if not rec.is_constant or rec.length_bytes <= 0:
            continue
        safe = rec.name.replace('"', '\\"')
        lines.append(f'  {{"{safe}", {rec.tensor_index}u, {rec.global_offset}u, {rec.length_bytes}u}},')
        count += 1

    lines.extend(["};", f"static const uint32_t g_tensor_map_count = {count}u;", ""])
    return "\n".join(lines)


def generate_memory_fault_list_header(
    selected_csv: Path,
    groups_csv: Optional[Path] = None,
    multi_count: int = 0,
) -> str:
    df = pd.read_csv(selected_csv)
    is_multi = groups_csv is not None and multi_count > 1

    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]

    if is_multi:
        lines.append(f"#define MULTI_FI_MAX_COUNT {multi_count}")
        lines.append("")

    lines.extend([
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
    ])

    pool_name = "g_fault_pool" if is_multi else "g_fault_list"
    lines.append(f"static const memory_fault_t {pool_name}[] = {{")

    for _, row in df.iterrows():
        op_index = int(row["op_index"]) if not pd.isna(row.get("op_index")) else 0xFFFFFFFF
        tensor_index = int(row["tensor_index"]) if not pd.isna(row.get("tensor_index")) else 0xFFFFFFFF
        tensor_offset = int(row["tensor_global_offset"]) if not pd.isna(row.get("tensor_global_offset")) else 0
        tensor_length = int(row["tensor_length_bytes"]) if not pd.isna(row.get("tensor_length_bytes")) else 0

        fields = [
            f"{int(row['fault_id'])}",
            f"{op_index}u",
            f"{tensor_index}u",
            f"{tensor_offset}u",
            f"{tensor_length}u",
            f"{int(row['byte_in_tensor'])}u",
            f"{int(row['global_byte_offset'])}u",
            f"{int(row['bit_index'])}u",
        ]
        lines.append("  {" + ", ".join(fields) + "},")

    lines.extend(["};", f"static const uint32_t {pool_name}_count = {len(df)}u;", ""])

    if is_multi:
        groups_df = pd.read_csv(groups_csv)
        pool_idx_cols = [c for c in groups_df.columns if c.startswith("pool_index_")]
        lines.extend([
            "typedef struct {",
            "  int32_t group_id;",
            "  uint8_t fault_count;",
            f"  uint16_t pool_indices[MULTI_FI_MAX_COUNT];",
            "} fault_group_t;",
            "",
            "static const fault_group_t g_fault_groups[] = {",
        ])
        for _, grow in groups_df.iterrows():
            indices = ", ".join(str(int(grow[c])) for c in pool_idx_cols)
            lines.append(f"  {{{int(grow['group_id'])}, {len(pool_idx_cols)}u, {{{indices}}}}},")
        lines.extend(["};", f"static const uint32_t g_fault_group_count = {len(groups_df)}u;", ""])

        lines.append("#define g_fault_list g_fault_pool")
        lines.append("#define g_fault_list_count g_fault_pool_count")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write & install
# ---------------------------------------------------------------------------

def write_headers(
    campaign_type: str,
    headers_dir: Path,
    tflite_path: Path,
    subset_npz: Path,
    selected_csv: Path,
    trainable_ops_csv: Path | None = None,
    tensor_records: List[TensorRecord] | None = None,
    groups_csv: Path | None = None,
    multi_count: int = 0,
) -> List[Path]:
    """Generate all campaign headers into *headers_dir*."""
    headers_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # Common headers
    p = headers_dir / "model_data.h"
    p.write_text(generate_model_header(tflite_path), encoding="utf-8")
    written.append(p)

    p = headers_dir / "mnist_subset.h"
    p.write_text(generate_subset_header(subset_npz), encoding="utf-8")
    written.append(p)

    # Campaign-specific headers
    if campaign_type == "register":
        p = headers_dir / "register_fault_list.h"
        p.write_text(
            generate_register_fault_list_header(selected_csv, groups_csv, multi_count),
            encoding="utf-8",
        )
        written.append(p)

        if trainable_ops_csv is not None:
            p = headers_dir / "register_operator_map.h"
            p.write_text(generate_register_operator_map_header(trainable_ops_csv), encoding="utf-8")
            written.append(p)

    elif campaign_type == "memory":
        p = headers_dir / "fault_list.h"
        p.write_text(
            generate_memory_fault_list_header(selected_csv, groups_csv, multi_count),
            encoding="utf-8",
        )
        written.append(p)

        if tensor_records is not None:
            p = headers_dir / "tensor_map.h"
            p.write_text(generate_tensor_map_header(tensor_records), encoding="utf-8")
            written.append(p)

    return written


def install_headers(campaign_type: str, headers_dir: Path, firmware_src: Path) -> List[Path]:
    """Copy generated headers into the firmware src/ directory."""
    if not firmware_src.is_dir():
        raise FileNotFoundError(f"Firmware src dir not found: {firmware_src}")

    expected = CAMPAIGN_HEADERS.get(campaign_type, [])
    installed: List[Path] = []

    for name in expected:
        src = headers_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Header not found: {src}")
        dst = firmware_src / name
        shutil.copy2(src, dst)
        installed.append(dst)

    return installed
