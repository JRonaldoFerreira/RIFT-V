"""Smoke tests for RIFT-V prepare step (no hardware required)."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from riftv.config import load_config, ExperimentConfig
from riftv.population import ModelIntrospector, generate_register_population, generate_memory_population


REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "lenet5_mnist_int8.tflite"
SUBSET = REPO / "models" / "mnist_subset.npz"


def _skip_if_no_model():
    if not MODEL.exists() or not SUBSET.exists():
        import pytest
        pytest.skip("Model files not present")


def test_load_config():
    cfg = load_config(REPO / "configs" / "registerfi_example.yaml")
    assert cfg.name
    assert cfg.campaign.type == "register"
    assert cfg.sampling.confidence == 0.95


def test_load_memory_config():
    cfg = load_config(REPO / "configs" / "memoryfi_example.yaml")
    assert cfg.campaign.type == "memory"
    assert cfg.campaign.memory is not None


def test_model_introspector():
    _skip_if_no_model()
    m = ModelIntrospector(MODEL)
    ops = m.collect_operators()
    assert len(ops) > 0
    assert ops[0].op_name == "CONV_2D"

    recs = m.collect_tensor_records()
    assert len(recs) > 0
    const = [r for r in recs.values() if r.is_constant]
    assert len(const) > 0


def test_register_population():
    _skip_if_no_model()
    m = ModelIntrospector(MODEL)
    pop, trainable, all_ops = generate_register_population(m, slot_count=3)
    assert len(pop) > 0
    assert "fault_id" in pop.columns
    assert "register_id" in pop.columns
    assert len(trainable) > 0


def test_memory_population():
    _skip_if_no_model()
    m = ModelIntrospector(MODEL)
    pop, tensor_records = generate_memory_population(m)
    assert len(pop) > 0
    assert "global_byte_offset" in pop.columns
    assert "bit_index" in pop.columns


def test_full_prepare_register():
    _skip_if_no_model()
    from riftv.experiment import Experiment

    cfg = load_config(REPO / "configs" / "registerfi_example.yaml")
    exp = Experiment(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        exp.out_dir = Path(tmpdir)
        artifacts = exp.prepare()
        assert artifacts.selected_csv.exists()
        assert artifacts.headers_dir.exists()
        assert (artifacts.headers_dir / "register_fault_list.h").exists()
        assert (artifacts.headers_dir / "model_data.h").exists()

        manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text())
        assert manifest["campaign_type"] == "register"


def test_full_prepare_memory():
    _skip_if_no_model()
    from riftv.experiment import Experiment

    cfg = load_config(REPO / "configs" / "memoryfi_example.yaml")
    exp = Experiment(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        exp.out_dir = Path(tmpdir)
        artifacts = exp.prepare()
        assert artifacts.selected_csv.exists()
        assert (artifacts.headers_dir / "fault_list.h").exists()
        assert (artifacts.headers_dir / "tensor_map.h").exists()


def test_load_hybrid_config():
    cfg = load_config(REPO / "configs" / "hybrid_campaign.yaml")
    assert cfg.campaign.type == "hybrid"
    assert cfg.campaign.register is not None
    assert cfg.campaign.memory is not None


def test_full_prepare_hybrid():
    _skip_if_no_model()
    from riftv.experiment import Experiment

    cfg = load_config(REPO / "configs" / "hybrid_campaign.yaml")
    exp = Experiment(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        exp.out_dir = Path(tmpdir)
        artifacts = exp.prepare()

        # Both sub-campaign artifacts should be stored
        assert exp._reg_artifacts is not None
        assert exp._mem_artifacts is not None

        # Register sub-campaign
        reg_dir = Path(tmpdir) / "register"
        assert reg_dir.exists()
        assert (reg_dir / "selected_register_faults.csv").exists()
        assert (reg_dir / "headers" / "register_fault_list.h").exists()
        assert (reg_dir / "headers" / "model_data.h").exists()

        # Memory sub-campaign
        mem_dir = Path(tmpdir) / "memory"
        assert mem_dir.exists()
        assert (mem_dir / "selected_faults.csv").exists()
        assert (mem_dir / "headers" / "fault_list.h").exists()
        assert (mem_dir / "headers" / "tensor_map.h").exists()

        manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text())
        assert manifest["campaign_type"] == "hybrid"


if __name__ == "__main__":
    test_load_config()
    test_load_memory_config()
    print("Config tests passed.")

    test_model_introspector()
    print("Introspector test passed.")

    test_register_population()
    print("Register population test passed.")

    test_memory_population()
    print("Memory population test passed.")

    test_full_prepare_register()
    print("Full register prepare test passed.")

    test_full_prepare_memory()
    print("Full memory prepare test passed.")

    test_load_hybrid_config()
    print("Hybrid config test passed.")

    test_full_prepare_hybrid()
    print("Full hybrid prepare test passed.")

    print("\nAll tests passed!")
