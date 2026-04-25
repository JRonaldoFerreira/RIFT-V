"""RIFT-V — Runtime-aware Injection of Faults in TinyML on RISC-V.

A statistical fault injection framework for evaluating TinyML inference
reliability on RISC-V microcontrollers.
"""

__version__ = "0.2.0"

from riftv.config import ExperimentConfig, load_config
from riftv.experiment import Experiment
from riftv.model_utils import quantize_model, prepare_test_subset
from riftv.analysis import (
    CampaignData,
    PhysicalAnalysisReport,
    generate_physical_report,
    load_campaign_data,
    enrich_results,
    bin_by_physical_offset,
    criticality_by_tensor,
    criticality_by_operator,
    physical_heatmap_data,
    contiguity_analysis,
    semantic_vs_physical,
    find_critical_regions,
    byte_level_vulnerability,
)

__all__ = [
    "ExperimentConfig",
    "Experiment",
    "load_config",
    # Model utilities
    "quantize_model",
    "prepare_test_subset",
    # Analysis
    "CampaignData",
    "PhysicalAnalysisReport",
    "generate_physical_report",
    "load_campaign_data",
    "enrich_results",
    "bin_by_physical_offset",
    "criticality_by_tensor",
    "criticality_by_operator",
    "physical_heatmap_data",
    "contiguity_analysis",
    "semantic_vs_physical",
    "find_critical_regions",
    "byte_level_vulnerability",
]
