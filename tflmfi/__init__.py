"""tflmfi — Unified Fault Injection Framework for TFLite-Micro on ESP32."""

__version__ = "0.2.0"

from tflmfi.config import ExperimentConfig, load_config
from tflmfi.experiment import Experiment
from tflmfi.model_utils import quantize_model, prepare_test_subset
from tflmfi.analysis import (
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
