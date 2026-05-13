"""Utilities for model quantization and test-subset preparation.

These helpers turn a trained Keras / SavedModel into a ready-to-use int8
TFLite file and build the random test-subset ``.npz`` that the firmware
header generator expects (keys: ``images``, ``labels``, ``test_indices``).

TensorFlow is imported lazily so the rest of the framework stays lightweight.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np


# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #

_KNOWN_DATASETS = ("mnist", "fashion_mnist", "cifar10", "cifar100")


def _default_preprocess(data: np.ndarray) -> np.ndarray:
    """Default preprocessing: normalise uint8 images to [0, 1]."""
    if data.max() > 1.0:
        data = data / 255.0
    return data


def quantize_model(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    representative_data: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
    calibration_data_path: Optional[Union[str, Path]] = None,
    num_calibration_samples: int = 500,
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Path:
    """Quantize a Keras model to full int8 TFLite and save it.

    Parameters
    ----------
    model_path:
        Path to a ``.h5`` / ``.keras`` file **or** a SavedModel directory.
    output_path:
        Destination ``.tflite`` file (parent dirs are created automatically).
    representative_data:
        Optional NumPy array used as calibration data.  When *None* and
        *dataset_name* is given, the **training** split is loaded
        automatically and used for calibration.
    dataset_name:
        One of ``"mnist"``, ``"fashion_mnist"``, ``"cifar10"``,
        ``"cifar100"``.  Used to load calibration data when
        *representative_data* is not provided.
    calibration_data_path:
        Path to a ``.npz`` file containing calibration images.  The file
        must have image arrays stored under one of the recognised key
        pairs: ``(images, labels)``, ``(x_test, y_test)``, or ``(x, y)``.
        Only the images are used for calibration.  Takes priority over
        *dataset_name* when *representative_data* is not provided.
    num_calibration_samples:
        Maximum number of samples drawn from the calibration set.
    preprocess_fn:
        Optional callable ``(np.ndarray) -> np.ndarray`` applied to the
        calibration batch **after** reshaping to 4-D.  Must reproduce the
        exact same preprocessing used during training (e.g.
        ``lambda x: x / 127.5 - 1.0`` for ``[-1, 1]`` models).  When
        *None*, the default heuristic divides by 255 if the data looks
        like uint8.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    tf = _import_tensorflow()

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    if model_path.is_dir():
        converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    elif model_path.suffix in (".h5", ".keras"):
        model = tf.keras.models.load_model(str(model_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        raise ValueError(
            f"Unsupported model format: {model_path}. "
            "Expected .h5, .keras, or a SavedModel directory."
        )

    # Resolve calibration data
    if representative_data is None and calibration_data_path is not None:
        x_cal, _ = _load_test_npz(calibration_data_path)
        representative_data = x_cal
    if representative_data is None and dataset_name is not None:
        (x_train, _), _ = _load_dataset(dataset_name)
        representative_data = x_train

    # Int8 quantization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    if representative_data is not None:
        cal = representative_data[:num_calibration_samples].astype(np.float32)
        # Ensure 4-D (N, H, W, C) for image models
        if cal.ndim == 3:
            cal = cal[..., np.newaxis]
        # Apply preprocessing — must match the training pipeline so that
        # the quantisation parameters (scale / zero-point) are correct.
        cal = (preprocess_fn or _default_preprocess)(cal)

        def _rep_gen():
            for i in range(len(cal)):
                yield [cal[i : i + 1]]

        converter.representative_dataset = _rep_gen

    tflite_bytes = converter.convert()
    output_path.write_bytes(tflite_bytes)
    print(
        f"[model_utils] Quantized model saved → {output_path} "
        f"({len(tflite_bytes):,} bytes)",
        file=sys.stderr,
    )
    return output_path


def prepare_test_subset(
    output_path: Union[str, Path],
    subset_size: int = 100,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None,
    test_data_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Path:
    """Create a random test-subset ``.npz`` ready for firmware headers.

    The saved file contains three arrays:

    * ``images``  — shape ``(N, H, W, C)``, float32, preprocessed.
    * ``labels``  — shape ``(N,)``, int32.
    * ``test_indices`` — shape ``(N,)``, int32, original indices in the full
      test set.

    Parameters
    ----------
    output_path:
        Destination ``.npz`` file.
    subset_size:
        Number of samples to draw (default 100).
    x, y:
        Raw test images and labels.  If omitted, *test_data_path* or
        *dataset_name* is used.
    dataset_name:
        One of ``"mnist"``, ``"fashion_mnist"``, ``"cifar10"``,
        ``"cifar100"``.  The **test** split is loaded automatically.
    test_data_path:
        Path to an ``.npz`` file containing the full test set.  The file
        must have image and label arrays stored under one of the
        recognised key pairs: ``(images, labels)``, ``(x_test, y_test)``,
        or ``(x, y)``.
    seed:
        Random seed for reproducibility.
    preprocess_fn:
        Optional callable ``(np.ndarray) -> np.ndarray`` applied to the
        image batch **after** reshaping to 4-D.  Must match the
        preprocessing used during training.  When *None*, the default
        heuristic divides by 255 if the data looks like uint8.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if x is not None and y is not None:
        x_test, y_test = x, y
    elif test_data_path is not None:
        x_test, y_test = _load_test_npz(test_data_path)
    elif dataset_name is not None:
        _, (x_test, y_test) = _load_dataset(dataset_name)
    else:
        raise ValueError(
            "Provide either (x, y) arrays, a test_data_path (.npz), "
            f"or a dataset_name (one of {_KNOWN_DATASETS})."
        )

    total = len(x_test)
    if subset_size > total:
        raise ValueError(
            f"subset_size ({subset_size}) exceeds dataset size ({total})"
        )

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=subset_size, replace=False))

    images = x_test[indices].astype(np.float32)
    # Ensure 4-D (N, H, W, C)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    # Apply preprocessing — must match the training pipeline.
    images = (preprocess_fn or _default_preprocess)(images)

    labels = y_test[indices].astype(np.int32)
    test_indices = indices.astype(np.int32)

    np.savez(
        output_path,
        images=images,
        labels=labels,
        test_indices=test_indices,
    )
    print(
        f"[model_utils] Test subset saved → {output_path} "
        f"({subset_size}/{total} samples, seed={seed})",
        file=sys.stderr,
    )
    return output_path


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _import_tensorflow():
    """Lazy TensorFlow import with a clear error message."""
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for this operation. "
            "Install it with:  pip install tensorflow"
        )


def _load_test_npz(path: Union[str, Path]):
    """Load test images and labels from a ``.npz`` file.

    Recognised key pairs (checked in order):
    ``(images, labels)``, ``(x_test, y_test)``, ``(x, y)``.

    Returns ``(x_test, y_test)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")

    data = np.load(path)
    _KEY_PAIRS = [
        ("images", "labels"),
        ("x_test", "y_test"),
        ("x", "y"),
    ]
    for img_key, lbl_key in _KEY_PAIRS:
        if img_key in data and lbl_key in data:
            return data[img_key], data[lbl_key]

    raise ValueError(
        f"Cannot find image/label arrays in '{path}'. "
        f"Available keys: {list(data.keys())}. "
        f"Expected one of these pairs: {_KEY_PAIRS}"
    )


def _load_dataset(name: str):
    """Load a well-known dataset via ``tf.keras.datasets``.

    Returns ``(x_train, y_train), (x_test, y_test)``.
    """
    tf = _import_tensorflow()

    loaders = {
        "mnist": tf.keras.datasets.mnist,
        "fashion_mnist": tf.keras.datasets.fashion_mnist,
        "cifar10": tf.keras.datasets.cifar10,
        "cifar100": tf.keras.datasets.cifar100,
    }

    key = name.lower()
    if key not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(loaders.keys())}"
        )

    return loaders[key].load_data()
