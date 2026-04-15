"""Model-physical memory analysis for fault-injection campaigns.

Uses existing semantic + physical metadata (tensor_global_offset,
global_byte_offset, byte_in_tensor, op_index, tensor_index) to analyse
the spatial distribution of faults across the serialised model blob.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_ORDER = ["Masked", "Acceptable", "Good", "Observed", "Warning", "Critical", "Crash"]

CRITICAL_CLASSES = {"Critical", "Crash"}
DEGRADED_CLASSES = {"Warning", "Critical", "Crash"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class CampaignData:
    """All DataFrames from a single campaign run."""
    classified: pd.DataFrame
    selected: pd.DataFrame
    tensor_map: pd.DataFrame
    population: Optional[pd.DataFrame] = None
    groups: Optional[pd.DataFrame] = None
    is_multi: bool = False


def load_campaign_data(run_dir: str | Path) -> CampaignData:
    """Load all relevant CSVs from a run directory."""
    d = Path(run_dir)

    classified_path = d / "classified_results.csv"
    if not classified_path.exists():
        raise FileNotFoundError(f"classified_results.csv not found in {d}")
    classified = pd.read_csv(classified_path)

    selected_path = d / "selected_faults.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"selected_faults.csv not found in {d}")
    selected = pd.read_csv(selected_path)

    tensor_map_path = d / "tensor_map.csv"
    if not tensor_map_path.exists():
        raise FileNotFoundError(f"tensor_map.csv not found in {d}")
    tensor_map = pd.read_csv(tensor_map_path)

    pop_path = d / "memory_population.csv"
    population = pd.read_csv(pop_path) if pop_path.exists() else None

    groups_path = d / "memory_fault_groups.csv"
    groups = pd.read_csv(groups_path) if groups_path.exists() else None

    is_multi = groups is not None and "group_id" in classified.columns

    return CampaignData(
        classified=classified,
        selected=selected,
        tensor_map=tensor_map,
        population=population,
        groups=groups,
        is_multi=is_multi,
    )


# ---------------------------------------------------------------------------
# Enrichment — join fault metadata into classified results
# ---------------------------------------------------------------------------

def enrich_results(data: CampaignData) -> pd.DataFrame:
    """Merge physical/semantic fault details into classified results.

    For single-injection campaigns, joins directly via fault_id.
    For multi-injection, explodes each group into constituent faults so
    every row carries full tensor/offset metadata.
    """
    classified = data.classified.copy()
    selected = data.selected.copy()

    if not data.is_multi:
        if "fault_id" in classified.columns and "fault_id" in selected.columns:
            fault_cols = [c for c in selected.columns if c != "fault_id"]
            enriched = classified.merge(
                selected[["fault_id"] + fault_cols],
                on="fault_id", how="left", suffixes=("", "_sel"),
            )
            return enriched
        return classified

    pool_cols = sorted(
        [c for c in classified.columns if c.startswith("pool_index_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    if not pool_cols:
        return classified

    selected = selected.reset_index(drop=True)
    rows: List[pd.Series] = []
    for _, row in classified.iterrows():
        for pc in pool_cols:
            idx = row.get(pc)
            if pd.isna(idx):
                continue
            idx = int(idx)
            if idx < 0 or idx >= len(selected):
                continue
            fault_info = selected.iloc[idx]
            merged = row.copy()
            merged["_pool_col"] = pc
            for col in fault_info.index:
                if col not in merged.index or pd.isna(merged.get(col)):
                    merged[col] = fault_info[col]
                elif col not in ("fault_id",):
                    merged[f"{col}_fault"] = fault_info[col]
            rows.append(merged)

    if not rows:
        return classified

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mode 1 — Model Physical Analysis
# ---------------------------------------------------------------------------

def _rate(group: pd.DataFrame, classes: set) -> float:
    if len(group) == 0:
        return 0.0
    return group["fault_class"].isin(classes).sum() / len(group)


def bin_by_physical_offset(
    enriched: pd.DataFrame,
    bin_size: int = 128,
    offset_col: str = "global_byte_offset",
) -> pd.DataFrame:
    """Group faults into physical-address bins and compute criticality metrics.

    Returns a DataFrame with columns:
        bin_start, bin_end, total, masked, critical, crash, degraded,
        critical_rate, degraded_rate, crash_rate
    """
    if offset_col not in enriched.columns:
        raise KeyError(f"Column '{offset_col}' not found. Available: {list(enriched.columns)}")

    df = enriched.dropna(subset=[offset_col]).copy()
    df["_bin"] = (df[offset_col].astype(int) // bin_size) * bin_size

    records: List[Dict[str, Any]] = []
    for bin_start, grp in df.groupby("_bin"):
        total = len(grp)
        fc = grp["fault_class"].value_counts()
        records.append({
            "bin_start": int(bin_start),
            "bin_end": int(bin_start) + bin_size,
            "total": total,
            "masked": int(fc.get("Masked", 0)),
            "acceptable": int(fc.get("Acceptable", 0)),
            "warning": int(fc.get("Warning", 0)),
            "critical": int(fc.get("Critical", 0)),
            "crash": int(fc.get("Crash", 0)),
            "critical_rate": _rate(grp, CRITICAL_CLASSES),
            "degraded_rate": _rate(grp, DEGRADED_CLASSES),
        })

    return pd.DataFrame(records).sort_values("bin_start").reset_index(drop=True)


def criticality_by_tensor(enriched: pd.DataFrame) -> pd.DataFrame:
    """Compute criticality rates per tensor.

    Returns a DataFrame with per-tensor statistics sorted by critical_rate
    descending.
    """
    cols = ["tensor_index", "tensor_name"]
    available = [c for c in cols if c in enriched.columns]
    if not available:
        raise KeyError("No tensor columns found in enriched data.")

    group_cols = available
    records: List[Dict[str, Any]] = []
    for keys, grp in enriched.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["total"] = len(grp)
        fc = grp["fault_class"].value_counts()
        row["critical"] = int(fc.get("Critical", 0))
        row["crash"] = int(fc.get("Crash", 0))
        row["degraded"] = int(fc.get("Warning", 0)) + row["critical"] + row["crash"]
        row["critical_rate"] = _rate(grp, CRITICAL_CLASSES)
        row["degraded_rate"] = _rate(grp, DEGRADED_CLASSES)

        if "tensor_global_offset" in grp.columns:
            row["phys_start"] = int(grp["tensor_global_offset"].min())
        if "tensor_length_bytes" in grp.columns:
            row["tensor_size"] = int(grp["tensor_length_bytes"].iloc[0])
        records.append(row)

    return pd.DataFrame(records).sort_values("critical_rate", ascending=False).reset_index(drop=True)


def criticality_by_operator(enriched: pd.DataFrame) -> pd.DataFrame:
    """Compute criticality rates per operator."""
    cols = ["op_index", "op_name"]
    available = [c for c in cols if c in enriched.columns]
    if not available:
        raise KeyError("No operator columns found.")

    records: List[Dict[str, Any]] = []
    for keys, grp in enriched.groupby(available, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(available, keys))
        row["total"] = len(grp)
        fc = grp["fault_class"].value_counts()
        row["critical"] = int(fc.get("Critical", 0))
        row["crash"] = int(fc.get("Crash", 0))
        row["critical_rate"] = _rate(grp, CRITICAL_CLASSES)
        row["degraded_rate"] = _rate(grp, DEGRADED_CLASSES)
        records.append(row)

    return pd.DataFrame(records).sort_values("critical_rate", ascending=False).reset_index(drop=True)


def physical_heatmap_data(
    enriched: pd.DataFrame,
    bin_size: int = 64,
    metric: str = "critical_rate",
) -> pd.DataFrame:
    """Generate a 2D heatmap: tensor_name (Y) × physical offset bin (X).

    Returns a pivot table suitable for seaborn.heatmap or matplotlib imshow.
    """
    if "tensor_name" not in enriched.columns or "global_byte_offset" not in enriched.columns:
        raise KeyError("Need 'tensor_name' and 'global_byte_offset' columns.")

    df = enriched.dropna(subset=["global_byte_offset", "tensor_name"]).copy()
    df["offset_bin"] = (df["global_byte_offset"].astype(int) // bin_size) * bin_size

    agg: List[Dict[str, Any]] = []
    for (tensor, obin), grp in df.groupby(["tensor_name", "offset_bin"]):
        agg.append({
            "tensor_name": tensor,
            "offset_bin": int(obin),
            "total": len(grp),
            "critical_rate": _rate(grp, CRITICAL_CLASSES),
            "degraded_rate": _rate(grp, DEGRADED_CLASSES),
        })

    agg_df = pd.DataFrame(agg)
    if agg_df.empty:
        return agg_df

    pivot = agg_df.pivot_table(
        index="tensor_name", columns="offset_bin", values=metric, fill_value=0.0,
    )
    return pivot


def contiguity_analysis(enriched: pd.DataFrame, tensor_map: pd.DataFrame) -> pd.DataFrame:
    """Analyse whether physically adjacent tensors have correlated criticality.

    Returns a DataFrame with pairs of adjacent tensors and their
    criticality rates.
    """
    tmap = tensor_map[tensor_map["is_constant"] == True].copy()  # noqa: E712
    tmap = tmap.sort_values("global_offset").reset_index(drop=True)

    if len(tmap) < 2:
        return pd.DataFrame()

    tensor_crit: Dict[int, float] = {}
    if "tensor_index" in enriched.columns and "fault_class" in enriched.columns:
        for ti, grp in enriched.groupby("tensor_index"):
            tensor_crit[int(ti)] = _rate(grp, CRITICAL_CLASSES)

    pairs: List[Dict[str, Any]] = []
    for i in range(len(tmap) - 1):
        t_a = tmap.iloc[i]
        t_b = tmap.iloc[i + 1]
        gap = int(t_b["global_offset"]) - (int(t_a["global_offset"]) + int(t_a["length_bytes"]))
        pairs.append({
            "tensor_a_index": int(t_a["tensor_index"]),
            "tensor_a_name": str(t_a["name"]),
            "tensor_a_offset": int(t_a["global_offset"]),
            "tensor_a_size": int(t_a["length_bytes"]),
            "tensor_b_index": int(t_b["tensor_index"]),
            "tensor_b_name": str(t_b["name"]),
            "tensor_b_offset": int(t_b["global_offset"]),
            "tensor_b_size": int(t_b["length_bytes"]),
            "gap_bytes": gap,
            "tensor_a_critical_rate": tensor_crit.get(int(t_a["tensor_index"]), float("nan")),
            "tensor_b_critical_rate": tensor_crit.get(int(t_b["tensor_index"]), float("nan")),
        })

    result = pd.DataFrame(pairs)
    if not result.empty:
        result["crit_rate_diff"] = abs(result["tensor_a_critical_rate"] - result["tensor_b_critical_rate"])
    return result


def semantic_vs_physical(
    enriched: pd.DataFrame,
    bin_size: int = 256,
) -> Dict[str, Any]:
    """Compare how well semantic grouping (op/tensor) vs physical grouping
    (offset bins) explains the distribution of critical faults.

    Returns a dict with:
        - semantic_entropy: entropy of critical-rate across tensors
        - physical_entropy: entropy of critical-rate across offset bins
        - semantic_gini: Gini coefficient of criticality across tensors
        - physical_gini: Gini coefficient of criticality across offset bins
        - verdict: which grouping concentrates criticality more
    """
    if "fault_class" not in enriched.columns:
        raise KeyError("Need 'fault_class' column.")

    def _gini(values: np.ndarray) -> float:
        values = np.sort(values)
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float(((2 * index - n - 1) * values).sum() / (n * values.sum()))

    def _entropy(values: np.ndarray) -> float:
        values = values[values > 0]
        if len(values) == 0:
            return 0.0
        p = values / values.sum()
        return float(-np.sum(p * np.log2(p)))

    sem_rates = []
    if "tensor_index" in enriched.columns:
        for _, grp in enriched.groupby("tensor_index"):
            sem_rates.append(_rate(grp, CRITICAL_CLASSES))
    sem_rates = np.array(sem_rates)

    phys_rates = []
    if "global_byte_offset" in enriched.columns:
        df = enriched.dropna(subset=["global_byte_offset"]).copy()
        df["_bin"] = (df["global_byte_offset"].astype(int) // bin_size) * bin_size
        for _, grp in df.groupby("_bin"):
            phys_rates.append(_rate(grp, CRITICAL_CLASSES))
    phys_rates = np.array(phys_rates)

    sem_gini = _gini(sem_rates) if len(sem_rates) > 0 else 0.0
    phys_gini = _gini(phys_rates) if len(phys_rates) > 0 else 0.0
    sem_ent = _entropy(sem_rates) if len(sem_rates) > 0 else 0.0
    phys_ent = _entropy(phys_rates) if len(phys_rates) > 0 else 0.0

    if sem_gini > phys_gini:
        verdict = "semantic"
        explanation = "Criticality is more concentrated by tensor/operator than by physical address range."
    elif phys_gini > sem_gini:
        verdict = "physical"
        explanation = "Criticality is more concentrated by physical address range than by tensor/operator."
    else:
        verdict = "balanced"
        explanation = "Criticality concentration is similar for both semantic and physical groupings."

    return {
        "semantic_gini": round(sem_gini, 4),
        "physical_gini": round(phys_gini, 4),
        "semantic_entropy": round(sem_ent, 4),
        "physical_entropy": round(phys_ent, 4),
        "num_tensors": len(sem_rates),
        "num_physical_bins": len(phys_rates),
        "bin_size": bin_size,
        "verdict": verdict,
        "explanation": explanation,
    }


def find_critical_regions(
    enriched: pd.DataFrame,
    bin_size: int = 64,
    top_n: int = 10,
    min_faults: int = 5,
) -> pd.DataFrame:
    """Identify the top-N most critical physical regions of the model.

    Returns a DataFrame of the hottest bins sorted by critical_rate
    descending.
    """
    binned = bin_by_physical_offset(enriched, bin_size=bin_size)
    binned = binned[binned["total"] >= min_faults]
    return binned.nlargest(top_n, "critical_rate").reset_index(drop=True)


def byte_level_vulnerability(
    enriched: pd.DataFrame,
    tensor_index: Optional[int] = None,
) -> pd.DataFrame:
    """Per-byte vulnerability within a tensor or across all tensors.

    Useful for identifying which byte positions within tensors are most
    vulnerable (e.g. MSB vs LSB of quantised weights).
    """
    df = enriched.copy()
    if tensor_index is not None and "tensor_index" in df.columns:
        df = df[df["tensor_index"] == tensor_index]

    if "byte_in_tensor" not in df.columns or "bit_index" not in df.columns:
        raise KeyError("Need 'byte_in_tensor' and 'bit_index' columns.")

    records: List[Dict[str, Any]] = []
    for bit_idx, grp in df.groupby("bit_index"):
        total = len(grp)
        fc = grp["fault_class"].value_counts()
        records.append({
            "bit_index": int(bit_idx),
            "total": total,
            "critical": int(fc.get("Critical", 0)),
            "crash": int(fc.get("Crash", 0)),
            "critical_rate": _rate(grp, CRITICAL_CLASSES),
            "degraded_rate": _rate(grp, DEGRADED_CLASSES),
        })

    return pd.DataFrame(records).sort_values("bit_index").reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level report generation
# ---------------------------------------------------------------------------

@dataclass
class PhysicalAnalysisReport:
    """Complete model-physical analysis report."""
    binned_offsets: pd.DataFrame
    by_tensor: pd.DataFrame
    by_operator: pd.DataFrame
    contiguity: pd.DataFrame
    semantic_vs_physical: Dict[str, Any]
    critical_regions: pd.DataFrame
    bit_vulnerability: pd.DataFrame
    enriched: pd.DataFrame

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Model-Physical Analysis Report",
            "=" * 60,
            "",
            f"  Total enriched fault results: {len(self.enriched):,}",
            "",
        ]

        if not self.by_tensor.empty:
            top = self.by_tensor.iloc[0]
            name = top.get("tensor_name", top.get("tensor_index", "?"))
            lines.append(f"  Most critical tensor: {name} (rate={top['critical_rate']:.3f})")

        if not self.by_operator.empty:
            top = self.by_operator.iloc[0]
            name = top.get("op_name", top.get("op_index", "?"))
            lines.append(f"  Most critical operator: {name} (rate={top['critical_rate']:.3f})")

        svp = self.semantic_vs_physical
        lines.extend([
            "",
            f"  Semantic vs Physical verdict: {svp['verdict']}",
            f"    Semantic  Gini={svp['semantic_gini']:.4f}  Entropy={svp['semantic_entropy']:.4f}",
            f"    Physical  Gini={svp['physical_gini']:.4f}  Entropy={svp['physical_entropy']:.4f}",
            f"    {svp['explanation']}",
            "",
        ])

        if not self.critical_regions.empty:
            top = self.critical_regions.iloc[0]
            lines.append(
                f"  Hottest physical region: [{top['bin_start']:#x}, {top['bin_end']:#x})"
                f" rate={top['critical_rate']:.3f} ({top['total']} faults)"
            )

        if not self.bit_vulnerability.empty:
            worst_bit = self.bit_vulnerability.loc[self.bit_vulnerability["critical_rate"].idxmax()]
            lines.append(
                f"  Most vulnerable bit position: bit {int(worst_bit['bit_index'])}"
                f" (rate={worst_bit['critical_rate']:.3f})"
            )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> List[Path]:
        """Save all report DataFrames as CSVs."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for name, df in [
            ("physical_bins", self.binned_offsets),
            ("criticality_by_tensor", self.by_tensor),
            ("criticality_by_operator", self.by_operator),
            ("contiguity_analysis", self.contiguity),
            ("critical_regions", self.critical_regions),
            ("bit_vulnerability", self.bit_vulnerability),
        ]:
            p = d / f"analysis_{name}.csv"
            df.to_csv(p, index=True if isinstance(df.index, pd.MultiIndex) or name == "physical_heatmap" else False)
            saved.append(p)

        import json
        p = d / "analysis_semantic_vs_physical.json"
        p.write_text(json.dumps(self.semantic_vs_physical, indent=2), encoding="utf-8")
        saved.append(p)

        p = d / "analysis_summary.txt"
        p.write_text(self.summary(), encoding="utf-8")
        saved.append(p)

        return saved


def generate_physical_report(
    run_dir: str | Path,
    bin_size: int = 128,
    top_n_regions: int = 10,
) -> PhysicalAnalysisReport:
    """Run the full model-physical analysis pipeline on a campaign run.

    Parameters
    ----------
    run_dir : path
        Directory containing classified_results.csv, selected_faults.csv,
        tensor_map.csv.
    bin_size : int
        Size of physical address bins in bytes.
    top_n_regions : int
        Number of top critical regions to report.

    Returns
    -------
    PhysicalAnalysisReport
        Comprehensive analysis with all tables and a summary.
    """
    data = load_campaign_data(run_dir)
    enriched = enrich_results(data)

    return PhysicalAnalysisReport(
        binned_offsets=bin_by_physical_offset(enriched, bin_size=bin_size),
        by_tensor=criticality_by_tensor(enriched),
        by_operator=criticality_by_operator(enriched),
        contiguity=contiguity_analysis(enriched, data.tensor_map),
        semantic_vs_physical=semantic_vs_physical(enriched, bin_size=bin_size),
        critical_regions=find_critical_regions(enriched, bin_size=bin_size // 2, top_n=top_n_regions),
        bit_vulnerability=byte_level_vulnerability(enriched),
        enriched=enriched,
    )
