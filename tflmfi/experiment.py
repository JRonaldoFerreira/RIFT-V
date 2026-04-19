"""Experiment lifecycle: prepare → build → flash → run → analyze."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from tflmfi.config import ExperimentConfig, load_config
from tflmfi.population import (
    CampaignArtifacts,
    ModelIntrospector,
    prepare_memory_campaign,
    prepare_register_campaign,
)
from tflmfi.headers import install_headers, write_headers
from tflmfi import builder as pio_builder
from tflmfi.uart import capture as uart_capture
from tflmfi.classify import classify_results
from tflmfi.analysis import generate_physical_report, PhysicalAnalysisReport
from tflmfi.model_utils import (
    quantize_model as _quantize_model,
    prepare_test_subset as _prepare_test_subset,
)


class Experiment:
    """High-level orchestrator for a fault-injection experiment.

    Usage::

        from tflmfi import Experiment, load_config

        cfg = load_config("configs/memoryfi.yaml")
        exp = Experiment(cfg)

        exp.prepare()       # generate population, sample, headers
        exp.build()         # pio build
        exp.flash()         # pio upload
        exp.run()           # UART capture
        exp.analyze()       # classify results

        # Or run the full pipeline:
        exp.execute()
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.out_dir: Path = cfg.resolve_output_dir()
        self.artifacts: Optional[CampaignArtifacts] = None
        self._reg_artifacts: Optional[CampaignArtifacts] = None
        self._mem_artifacts: Optional[CampaignArtifacts] = None
        self._introspector: Optional[ModelIntrospector] = None

    # ------------------------------------------------------------------ #
    # Step 0a: QUANTIZE — convert Keras model to int8 TFLite
    # ------------------------------------------------------------------ #

    def quantize_model(
        self,
        model_path: str | Path,
        output_path: Optional[str | Path] = None,
        representative_data: Optional["np.ndarray"] = None,
        dataset_name: Optional[str] = None,
        num_calibration_samples: int = 500,
        preprocess_fn=None,
    ) -> Path:
        """Quantize a Keras model to int8 TFLite and save it.

        The resulting ``.tflite`` is placed at *output_path* (defaults to the
        path declared in ``model.tflite`` in the config), ready to be used by
        ``prepare()``.

        Parameters
        ----------
        model_path:
            ``.h5`` / ``.keras`` file or SavedModel directory.
        output_path:
            Where to write the ``.tflite`` file.  Defaults to
            ``cfg.model.tflite``.
        representative_data:
            NumPy array for calibration.  When *None* and *dataset_name* is
            provided, the training split is loaded automatically.
        dataset_name:
            ``"mnist"``, ``"fashion_mnist"``, ``"cifar10"``, or
            ``"cifar100"``.  Used for auto-loading calibration data.
        num_calibration_samples:
            How many samples to use for calibration (default 500).
        preprocess_fn:
            Optional callable ``(np.ndarray) -> np.ndarray`` applied to
            calibration data.  Must match training preprocessing.  When
            *None*, divides by 255 if data looks like uint8.

        Returns
        -------
        Path
            The written ``.tflite`` file.
        """
        if output_path is None:
            output_path = self.cfg.tflite_path
        result = _quantize_model(
            model_path=model_path,
            output_path=output_path,
            representative_data=representative_data,
            dataset_name=dataset_name,
            num_calibration_samples=num_calibration_samples,
            preprocess_fn=preprocess_fn,
        )
        print(
            f"[experiment] Quantized model ready at {result}",
            file=sys.stderr,
        )
        return result

    # ------------------------------------------------------------------ #
    # Step 0b: SUBSET — build random test-subset .npz
    # ------------------------------------------------------------------ #

    def prepare_test_subset(
        self,
        subset_size: int = 100,
        output_path: Optional[str | Path] = None,
        x: Optional["np.ndarray"] = None,
        y: Optional["np.ndarray"] = None,
        dataset_name: Optional[str] = None,
        test_data_path: Optional[Union[str, Path]] = None,
        seed: int = 42,
        preprocess_fn=None,
    ) -> Path:
        """Create a random test-subset ``.npz`` for the firmware.

        The file is saved at *output_path* (defaults to ``model.subset_npz``
        in the config) with the keys ``images``, ``labels``, and
        ``test_indices`` expected by the header generator.

        Parameters
        ----------
        subset_size:
            Number of test samples to draw (default 100).
        output_path:
            Destination ``.npz`` file.  Defaults to ``cfg.model.subset_npz``.
        x, y:
            Full test images and labels.  If omitted, *test_data_path* or
            *dataset_name* is used.
        dataset_name:
            ``"mnist"``, ``"fashion_mnist"``, ``"cifar10"``, or
            ``"cifar100"``.  The test split is loaded automatically.
        test_data_path:
            Path to an ``.npz`` file with the full test set.  Must contain
            one of these key pairs: ``(images, labels)``,
            ``(x_test, y_test)``, or ``(x, y)``.
        seed:
            Random seed for reproducibility.
        preprocess_fn:
            Optional callable ``(np.ndarray) -> np.ndarray`` applied to
            the image batch.  Must match training preprocessing.  When
            *None*, divides by 255 if data looks like uint8.

        Returns
        -------
        Path
            The written ``.npz`` file.
        """
        if output_path is None:
            output_path = self.cfg.subset_npz_path
        result = _prepare_test_subset(
            output_path=output_path,
            subset_size=subset_size,
            x=x,
            y=y,
            dataset_name=dataset_name,
            test_data_path=test_data_path,
            seed=seed,
            preprocess_fn=preprocess_fn,
        )
        print(
            f"[experiment] Test subset ready at {result}",
            file=sys.stderr,
        )
        return result

    # ------------------------------------------------------------------ #
    # Step 1: PREPARE — population + sampling + header generation
    # ------------------------------------------------------------------ #

    def prepare(self) -> CampaignArtifacts:
        """Generate population, sample, write headers, install into firmware."""
        print(f"[experiment] Preparing campaign '{self.cfg.campaign.type}' → {self.out_dir}", file=sys.stderr)

        self._introspector = ModelIntrospector(self.cfg.tflite_path)
        model_op_names = [op.op_name for op in self._introspector.collect_operators()]
        ctype = self.cfg.campaign.type

        if ctype == "register":
            self.artifacts = prepare_register_campaign(
                self._introspector, self.cfg.campaign, self.cfg.sampling, self.out_dir,
            )
            trainable_csv = self.out_dir / "trainable_operator_instances.csv"
            mi = self.artifacts.multi_injection
            write_headers(
                "register",
                self.artifacts.headers_dir,
                self.cfg.tflite_path,
                self.cfg.subset_npz_path,
                self.artifacts.selected_csv,
                trainable_ops_csv=trainable_csv,
                groups_csv=self.artifacts.groups_csv,
                multi_count=mi.count if mi and mi.enabled else 0,
                model_op_names=model_op_names,
            )

        elif ctype == "memory":
            self.artifacts = prepare_memory_campaign(
                self._introspector, self.cfg.campaign, self.cfg.sampling, self.out_dir,
            )
            tensor_records = list(self._introspector.collect_tensor_records().values())
            mi = self.artifacts.multi_injection
            write_headers(
                "memory",
                self.artifacts.headers_dir,
                self.cfg.tflite_path,
                self.cfg.subset_npz_path,
                self.artifacts.selected_csv,
                tensor_records=tensor_records,
                groups_csv=self.artifacts.groups_csv,
                multi_count=mi.count if mi and mi.enabled else 0,
                model_op_names=model_op_names,
            )

        elif ctype == "hybrid":
            # Run both campaigns into sub-directories
            reg_dir = self.out_dir / "register"
            mem_dir = self.out_dir / "memory"

            reg_artifacts = prepare_register_campaign(
                self._introspector, self.cfg.campaign, self.cfg.sampling, reg_dir,
            )
            trainable_csv = reg_dir / "trainable_operator_instances.csv"
            mi_reg = reg_artifacts.multi_injection
            write_headers(
                "register", reg_artifacts.headers_dir,
                self.cfg.tflite_path, self.cfg.subset_npz_path,
                reg_artifacts.selected_csv, trainable_ops_csv=trainable_csv,
                groups_csv=reg_artifacts.groups_csv,
                multi_count=mi_reg.count if mi_reg and mi_reg.enabled else 0,
                model_op_names=model_op_names,
            )

            mem_artifacts = prepare_memory_campaign(
                self._introspector, self.cfg.campaign, self.cfg.sampling, mem_dir,
            )
            tensor_records = list(self._introspector.collect_tensor_records().values())
            mi_mem = mem_artifacts.multi_injection
            write_headers(
                "memory", mem_artifacts.headers_dir,
                self.cfg.tflite_path, self.cfg.subset_npz_path,
                mem_artifacts.selected_csv, tensor_records=tensor_records,
                groups_csv=mem_artifacts.groups_csv,
                multi_count=mi_mem.count if mi_mem and mi_mem.enabled else 0,
                model_op_names=model_op_names,
            )

            self._reg_artifacts = reg_artifacts
            self._mem_artifacts = mem_artifacts
            # Use register as primary artifacts for header installation below
            self.artifacts = reg_artifacts
            print("[experiment] Hybrid: both register/ and memory/ artifacts generated.", file=sys.stderr)
        else:
            raise ValueError(f"Unknown campaign type: {ctype}")

        # Install headers into firmware
        installed = install_headers(
            self.artifacts.campaign_type,
            self.artifacts.headers_dir,
            self.cfg.firmware_src_path,
        )
        print(f"[experiment] Installed {len(installed)} headers → {self.cfg.firmware_src_path}", file=sys.stderr)

        # Write run manifest
        manifest = {
            "experiment": self.cfg.name,
            "campaign_type": ctype,
            "fault_duration": self.cfg.campaign.fault_duration,
            "output_dir": str(self.out_dir),
            "tflite": str(self.cfg.tflite_path),
            "subset_npz": str(self.cfg.subset_npz_path),
            "sampling": {
                "confidence": self.cfg.sampling.confidence,
                "margin": self.cfg.sampling.margin,
                "p_prior": self.cfg.sampling.p_prior,
                "seed": self.cfg.sampling.seed,
            },
        }
        manifest_path = self.out_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[experiment] Manifest → {manifest_path}", file=sys.stderr)

        return self.artifacts

    # ------------------------------------------------------------------ #
    # Step 2: BUILD
    # ------------------------------------------------------------------ #

    def build(self, verbose: bool = False) -> None:
        """Build firmware for the configured campaign type."""
        if self.cfg.campaign.type == "hybrid":
            for ctype in ("register", "memory"):
                print(f"[experiment] Building firmware (env={ctype})...", file=sys.stderr)
                pio_builder.build(self.cfg, ctype, verbose=verbose)
            print("[experiment] Both builds succeeded.", file=sys.stderr)
            return
        ctype = self.cfg.campaign.type
        print(f"[experiment] Building firmware (env={ctype})...", file=sys.stderr)
        pio_builder.build(self.cfg, ctype, verbose=verbose)
        print("[experiment] Build succeeded.", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Step 3: FLASH
    # ------------------------------------------------------------------ #

    def flash(self, upload_port: Optional[str] = None, verbose: bool = False) -> None:
        """Flash firmware onto the target board."""
        if self.cfg.campaign.type == "hybrid":
            print("[experiment] Note: standalone flash with hybrid only flashes register.", file=sys.stderr)
            print("[experiment]       Use execute() for the full hybrid pipeline.", file=sys.stderr)
        ctype = self.cfg.campaign.type if self.cfg.campaign.type != "hybrid" else "register"
        port = upload_port or self.cfg.serial.port
        print(f"[experiment] Flashing via {port} (env={ctype})...", file=sys.stderr)
        pio_builder.flash(self.cfg, ctype, upload_port=port, verbose=verbose)
        print("[experiment] Flash succeeded.", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Step 4: RUN — UART capture
    # ------------------------------------------------------------------ #

    def run(self, stop_after: Optional[int] = None) -> Path:
        """Capture campaign results via UART. Returns the CSV path."""
        import numpy as np
        import pandas as pd

        jsonl = self.out_dir / "results.jsonl"
        csv = self.out_dir / "results.csv"
        ctype = self.cfg.campaign.type
        mi = self.cfg.campaign.multi_injection
        is_multi = bool(mi and mi.enabled)
        groups_csv = self.artifacts.groups_csv if self.artifacts else None

        # Compute fault_count
        fault_count = 0
        if self.artifacts:
            if is_multi and groups_csv and groups_csv.exists():
                fault_count = len(pd.read_csv(groups_csv))
            elif self.artifacts.selected_csv.exists():
                fault_count = len(pd.read_csv(self.artifacts.selected_csv))

        # Compute image_count from subset .npz
        image_count = 0
        subset_path = self.cfg.subset_npz_path
        if subset_path.exists():
            subset = np.load(subset_path)
            image_count = len(subset["labels"])

        print(f"[experiment] Capturing UART → {csv}", file=sys.stderr)
        uart_capture(
            self.cfg.serial, jsonl, csv,
            fault_count=fault_count,
            image_count=image_count,
            stop_after=stop_after,
            campaign_type=ctype,
            is_multi=is_multi,
            groups_csv=groups_csv,
        )
        print(f"[experiment] Capture complete → {csv}", file=sys.stderr)
        return csv

    # ------------------------------------------------------------------ #
    # Step 4b: REBUILD-CSV — regenerate results.csv from JSONL
    # ------------------------------------------------------------------ #

    def rebuild_csv(self, run_dir: Optional[Path] = None) -> Path:
        """Re-generate results.csv from results.jsonl in an existing run directory.

        Useful when the CSV column schema changed after a capture.
        """
        from tflmfi.uart import build_csv

        target = run_dir or self.out_dir
        jsonl = target / "results.jsonl"
        csv = target / "results.csv"
        ctype = self.cfg.campaign.type
        mi = self.cfg.campaign.multi_injection
        is_multi = bool(mi and mi.enabled)
        groups_csv = next(target.glob("*_fault_groups.csv"), None)

        if not jsonl.exists():
            raise FileNotFoundError(f"No JSONL found at {jsonl}")

        print(f"[experiment] Rebuilding CSV from {jsonl}...", file=sys.stderr)
        build_csv(jsonl, csv, campaign_type=ctype, is_multi=is_multi, groups_csv=groups_csv)
        print(f"[experiment] CSV rebuilt → {csv}", file=sys.stderr)
        return csv

    # ------------------------------------------------------------------ #
    # Step 5: ANALYZE — classify
    # ------------------------------------------------------------------ #

    def analyze(self, results_csv: Optional[Path] = None) -> Path | Dict[str, Path]:
        """Classify captured results. Returns the classified CSV path(s)."""
        if self.cfg.campaign.type == "hybrid" and results_csv is None:
            classified: Dict[str, Path] = {}
            for sub_type in ("register", "memory"):
                sub_dir = self.out_dir / sub_type
                csv = sub_dir / "results.csv"
                out = sub_dir / "classified_results.csv"
                if csv.exists():
                    print(f"[experiment] Classifying {sub_type}: {csv}...", file=sys.stderr)
                    classify_results(csv, out)
                    classified[sub_type] = out
                    print(f"[experiment] Classified {sub_type} → {out}", file=sys.stderr)
                else:
                    print(f"[experiment] Skipping {sub_type}: no results at {csv}", file=sys.stderr)
            return classified
        if results_csv is None:
            results_csv = self.out_dir / "results.csv"
        classified_csv = self.out_dir / "classified_results.csv"
        print(f"[experiment] Classifying {results_csv}...", file=sys.stderr)
        classify_results(results_csv, classified_csv)
        print(f"[experiment] Classified → {classified_csv}", file=sys.stderr)
        return classified_csv

    # ------------------------------------------------------------------ #
    # Step 5b: PHYSICAL ANALYSIS — model/RAM spatial analysis
    # ------------------------------------------------------------------ #

    def physical_analysis(
        self,
        run_dir: Optional[Path] = None,
        bin_size: int = 128,
    ) -> PhysicalAnalysisReport:
        """Run model-physical analysis on classified results.

        Returns a :class:`PhysicalAnalysisReport` with all tables.
        """
        target_dir = run_dir or self.out_dir
        print(f"[experiment] Running physical memory analysis on {target_dir}...", file=sys.stderr)
        report = generate_physical_report(target_dir, bin_size=bin_size)

        saved = report.save(target_dir)
        print(f"[experiment] Physical analysis: {len(saved)} files saved.", file=sys.stderr)
        print(report.summary(), file=sys.stderr)
        return report

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #

    def execute(
        self,
        skip_build: bool = False,
        skip_flash: bool = False,
        stop_after: Optional[int] = None,
        verbose: bool = False,
    ) -> Path | Dict[str, Path]:
        """Run the full experiment pipeline: prepare → build → flash → run → analyze.

        Returns the path to the classified results CSV.
        For hybrid campaigns, returns a dict mapping sub-campaign name to CSV path.
        """
        self.prepare()

        if self.cfg.campaign.type == "hybrid":
            return self._execute_hybrid(
                skip_build=skip_build, skip_flash=skip_flash,
                stop_after=stop_after, verbose=verbose,
            )

        if not skip_build:
            self.build(verbose=verbose)

        if not skip_flash:
            self.flash(verbose=verbose)

        self.run(stop_after=stop_after)
        return self.analyze()

    # ------------------------------------------------------------------ #
    # Hybrid: sequential register + memory sub-campaigns
    # ------------------------------------------------------------------ #

    def _execute_hybrid(
        self,
        skip_build: bool = False,
        skip_flash: bool = False,
        stop_after: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Path]:
        """Run register and memory sub-campaigns sequentially on the target."""
        import numpy as np
        import pandas as pd

        results: Dict[str, Path] = {}

        for sub_type, artifacts in [
            ("register", self._reg_artifacts),
            ("memory", self._mem_artifacts),
        ]:
            sub_dir = self.out_dir / sub_type
            print(f"\n[experiment] {'='*50}", file=sys.stderr)
            print(f"[experiment]  Hybrid sub-campaign: {sub_type.upper()}", file=sys.stderr)
            print(f"[experiment] {'='*50}", file=sys.stderr)

            # Install sub-campaign headers into firmware src
            installed = install_headers(
                sub_type, artifacts.headers_dir, self.cfg.firmware_src_path,
            )
            print(f"[experiment] Installed {len(installed)} headers → {self.cfg.firmware_src_path}", file=sys.stderr)

            # Build
            if not skip_build:
                print(f"[experiment] Building firmware (env={sub_type})...", file=sys.stderr)
                pio_builder.build(self.cfg, sub_type, verbose=verbose)
                print(f"[experiment] Build succeeded.", file=sys.stderr)

            # Flash
            if not skip_flash:
                port = self.cfg.serial.port
                print(f"[experiment] Flashing via {port} (env={sub_type})...", file=sys.stderr)
                pio_builder.flash(self.cfg, sub_type, verbose=verbose)
                print(f"[experiment] Flash succeeded.", file=sys.stderr)

            # UART capture
            jsonl = sub_dir / "results.jsonl"
            csv = sub_dir / "results.csv"
            mi = self.cfg.campaign.multi_injection
            is_multi = bool(mi and mi.enabled)
            groups_csv_path = artifacts.groups_csv if artifacts else None

            # Compute fault_count / image_count for host-driven scheduler
            sub_fault_count = 0
            if artifacts:
                if is_multi and groups_csv_path and groups_csv_path.exists():
                    sub_fault_count = len(pd.read_csv(groups_csv_path))
                elif artifacts.selected_csv.exists():
                    sub_fault_count = len(pd.read_csv(artifacts.selected_csv))
            sub_image_count = 0
            subset_path = self.cfg.subset_npz_path
            if subset_path.exists():
                sub_image_count = len(np.load(subset_path)["labels"])

            print(f"[experiment] Capturing UART → {csv}", file=sys.stderr)
            uart_capture(
                self.cfg.serial, jsonl, csv,
                fault_count=sub_fault_count,
                image_count=sub_image_count,
                stop_after=stop_after,
                campaign_type=sub_type,
                is_multi=is_multi,
                groups_csv=groups_csv_path,
            )
            print(f"[experiment] Capture complete → {csv}", file=sys.stderr)

            # Classify
            classified = sub_dir / "classified_results.csv"
            print(f"[experiment] Classifying {csv}...", file=sys.stderr)
            classify_results(csv, classified)
            results[sub_type] = classified
            print(f"[experiment] {sub_type} classified → {classified}", file=sys.stderr)

        print(f"\n[experiment] {'='*50}", file=sys.stderr)
        print(f"[experiment]  Hybrid campaign complete", file=sys.stderr)
        print(f"[experiment] {'='*50}", file=sys.stderr)
        print(f"[experiment]   Register → {results.get('register')}", file=sys.stderr)
        print(f"[experiment]   Memory   → {results.get('memory')}", file=sys.stderr)
        return results
