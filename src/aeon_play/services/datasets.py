"""
Dataset loading and management service for aeon-play.

Handles:
- Built-in aeon dataset loading
- File uploads (CSV, .ts, .tsf)
- Dataset caching
- Data validation and limits
"""

import os
import io
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import diskcache

# aeon imports
AEON_AVAILABLE = False
load_classification = None
load_regression = None
load_from_tsfile = None
load_from_tsf_file = None
tsc_univariate = []
tsc_multivariate = []
tser_all = []

try:
    from aeon.datasets import load_classification, load_regression
    from aeon.datasets.tsc_datasets import univariate as tsc_univariate, multivariate as tsc_multivariate
    AEON_AVAILABLE = True
except ImportError:
    pass

# Optional imports that may not be available in all aeon versions
try:
    from aeon.datasets import load_from_tsfile, load_from_tsf_file
except ImportError:
    pass

try:
    from aeon.datasets.tser_datasets import tser_all
except ImportError:
    pass

# Configuration
MAX_CASES = int(os.getenv("AEON_PLAY_MAX_CASES", "10000"))
MAX_TIMEPOINTS = int(os.getenv("AEON_PLAY_MAX_TIMEPOINTS", "5000"))
CACHE_DIR = os.getenv("AEON_PLAY_CACHE_DIR", "./.cache")


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset."""
    name: str
    task_type: str  # classification, regression, forecasting, anomaly
    data_type: str  # collection, single_series
    n_cases: int
    n_channels: int
    n_timepoints: Union[int, str]  # int or "variable" for unequal length
    has_labels: bool
    n_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    is_univariate: bool = True
    is_equal_length: bool = True
    missing_values: int = 0
    memory_mb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class DatasetService:
    """Service for loading and managing datasets."""

    # Built-in dataset catalogs
    CLASSIFICATION_DATASETS = {
        "univariate": [
            "GunPoint", "ItalyPowerDemand", "ECG200", "TwoLeadECG",
            "Wafer", "FaceFour", "Coffee", "Beef", "ArrowHead",
            "CBF", "Chinatown", "SmoothSubspace", "UMD",
        ],
        "multivariate": [
            "BasicMotions", "Epilepsy", "NATOPS", "RacketSports",
            "ArticularyWordRecognition", "AtrialFibrillation",
        ],
    }

    REGRESSION_DATASETS = [
        "Covid3Month", "FloodModeling1", "FloodModeling2",
        "FloodModeling3", "HouseholdPowerConsumption1",
    ]

    FORECASTING_DATASETS = [
        "airline", "lynx", "shampoo_sales", "sunspots",
    ]

    def __init__(self, cache_dir: str = CACHE_DIR):
        """Initialize dataset service with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir / "datasets"))

    def get_available_datasets(self, task_type: str = "all") -> Dict[str, List[str]]:
        """Get list of available built-in datasets by task type."""
        datasets = {}

        if task_type in ("all", "classification"):
            datasets["classification"] = {
                "univariate": self.CLASSIFICATION_DATASETS["univariate"],
                "multivariate": self.CLASSIFICATION_DATASETS["multivariate"],
            }

        if task_type in ("all", "regression"):
            datasets["regression"] = self.REGRESSION_DATASETS

        if task_type in ("all", "forecasting"):
            datasets["forecasting"] = self.FORECASTING_DATASETS

        return datasets

    def load_builtin_dataset(
        self,
        name: str,
        task_type: str,
        split: Optional[str] = None,
    ) -> Tuple[Any, Optional[Any], DatasetInfo]:
        """
        Load a built-in aeon dataset.

        Args:
            name: Dataset name
            task_type: Type of task (classification, regression, forecasting)
            split: Optional split (train, test, None for both)

        Returns:
            Tuple of (X, y, DatasetInfo)
        """
        if not AEON_AVAILABLE:
            raise RuntimeError("aeon library not available")

        cache_key = f"builtin_{task_type}_{name}_{split}"

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            if task_type == "classification":
                if split:
                    X, y = load_classification(name, split=split)
                else:
                    X_train, y_train = load_classification(name, split="train")
                    X_test, y_test = load_classification(name, split="test")
                    X = np.concatenate([X_train, X_test], axis=0)
                    y = np.concatenate([y_train, y_test], axis=0)

            elif task_type == "regression":
                if split:
                    X, y = load_regression(name, split=split)
                else:
                    X_train, y_train = load_regression(name, split="train")
                    X_test, y_test = load_regression(name, split="test")
                    X = np.concatenate([X_train, X_test], axis=0)
                    y = np.concatenate([y_train, y_test], axis=0)

            elif task_type == "forecasting":
                # For forecasting, load as single series
                X, y = self._load_forecasting_dataset(name)

            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Apply limits
            X, y = self._apply_limits(X, y)

            # Generate info
            info = self._generate_info(X, y, name, task_type)

            result = (X, y, info)
            self.cache[cache_key] = result

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {name}: {str(e)}")

    def _load_forecasting_dataset(self, name: str) -> Tuple[np.ndarray, None]:
        """Load a forecasting dataset as a single series."""
        # Simple synthetic forecasting datasets
        np.random.seed(42)

        if name == "airline":
            # Classic airline passengers dataset (simulated)
            t = np.arange(144)
            trend = t * 2.5
            seasonal = 50 * np.sin(2 * np.pi * t / 12)
            noise = np.random.normal(0, 10, 144)
            X = 100 + trend + seasonal + noise

        elif name == "lynx":
            # Simulated lynx trappings
            t = np.arange(114)
            X = 1000 + 500 * np.sin(2 * np.pi * t / 10) + np.random.normal(0, 100, 114)
            X = np.maximum(X, 0)

        elif name == "shampoo_sales":
            # Simulated shampoo sales
            t = np.arange(36)
            X = 200 + t * 10 + np.random.normal(0, 20, 36)

        elif name == "sunspots":
            # Simulated sunspot data
            t = np.arange(200)
            X = 50 + 40 * np.sin(2 * np.pi * t / 11) + np.random.normal(0, 15, 200)
            X = np.maximum(X, 0)

        else:
            # Default synthetic series
            t = np.arange(100)
            X = 100 + 10 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 5, 100)

        return X.reshape(1, 1, -1), None

    def load_from_file(
        self,
        file_content: bytes,
        filename: str,
        file_format: str = "auto",
    ) -> Tuple[Any, Optional[Any], DatasetInfo]:
        """
        Load dataset from uploaded file.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            file_format: Format (auto, csv, ts, tsf)

        Returns:
            Tuple of (X, y, DatasetInfo)
        """
        if file_format == "auto":
            file_format = self._detect_format(filename)

        if file_format == "csv":
            return self._load_csv(file_content, filename)
        elif file_format == "ts":
            return self._load_ts(file_content, filename)
        elif file_format == "tsf":
            return self._load_tsf(file_content, filename)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def _detect_format(self, filename: str) -> str:
        """Detect file format from filename."""
        ext = Path(filename).suffix.lower()
        format_map = {
            ".csv": "csv",
            ".ts": "ts",
            ".tsf": "tsf",
            ".arff": "arff",
        }
        return format_map.get(ext, "csv")

    def _load_csv(
        self,
        content: bytes,
        filename: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], DatasetInfo]:
        """Load data from CSV file."""
        df = pd.read_csv(io.BytesIO(content))

        # Detect if last column is labels
        last_col = df.columns[-1]
        has_labels = df[last_col].dtype == object or df[last_col].nunique() < 20

        if has_labels:
            y = df[last_col].values
            X = df.drop(columns=[last_col]).values
        else:
            y = None
            X = df.values

        # Reshape to (n_cases, n_channels, n_timepoints)
        X = X.reshape(X.shape[0], 1, -1)

        # Apply limits
        X, y = self._apply_limits(X, y)

        info = self._generate_info(X, y, filename, "unknown")
        return X, y, info

    def _load_ts(
        self,
        content: bytes,
        filename: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], DatasetInfo]:
        """Load data from aeon .ts file."""
        if not AEON_AVAILABLE:
            raise RuntimeError("aeon library not available")

        # Write to temp file for aeon loader
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            X, y = load_from_tsfile(temp_path)
            X, y = self._apply_limits(X, y)
            info = self._generate_info(X, y, filename, "classification")
            return X, y, info
        finally:
            os.unlink(temp_path)

    def _load_tsf(
        self,
        content: bytes,
        filename: str,
    ) -> Tuple[Any, Optional[Any], DatasetInfo]:
        """Load data from aeon .tsf file."""
        if not AEON_AVAILABLE:
            raise RuntimeError("aeon library not available")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tsf", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            data, metadata = load_from_tsf_file(temp_path)
            # Convert to numpy array
            if isinstance(data, pd.DataFrame):
                X = data.values
            else:
                X = data
            X, _ = self._apply_limits(X, None)
            info = self._generate_info(X, None, filename, "forecasting")
            return X, None, info
        finally:
            os.unlink(temp_path)

    def _apply_limits(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply size limits to protect the server."""
        if X is None:
            return X, y

        # Check if X is a list (unequal length)
        if isinstance(X, list):
            if len(X) > MAX_CASES:
                X = X[:MAX_CASES]
                if y is not None:
                    y = y[:MAX_CASES]
            return X, y

        # Limit cases
        if X.shape[0] > MAX_CASES:
            indices = np.random.choice(X.shape[0], MAX_CASES, replace=False)
            X = X[indices]
            if y is not None:
                y = y[indices]

        # Limit timepoints
        if len(X.shape) >= 3 and X.shape[2] > MAX_TIMEPOINTS:
            X = X[:, :, :MAX_TIMEPOINTS]

        return X, y

    def _generate_info(
        self,
        X: Any,
        y: Optional[Any],
        name: str,
        task_type: str,
    ) -> DatasetInfo:
        """Generate DatasetInfo from data."""
        # Handle list (unequal length) format
        if isinstance(X, list):
            n_cases = len(X)
            n_channels = X[0].shape[0] if len(X[0].shape) > 1 else 1
            n_timepoints = "variable"
            is_equal_length = False
        else:
            n_cases = X.shape[0]
            n_channels = X.shape[1] if len(X.shape) > 2 else 1
            n_timepoints = X.shape[-1] if len(X.shape) > 1 else X.shape[0]
            is_equal_length = True

        # Determine data type
        if task_type in ("forecasting", "anomaly", "segmentation"):
            data_type = "single_series"
        else:
            data_type = "collection"

        # Class distribution
        class_dist = None
        n_classes = None
        if y is not None and task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            class_dist = {str(k): int(v) for k, v in zip(unique, counts)}
            n_classes = len(unique)

        # Missing values
        if isinstance(X, list):
            missing = sum(np.isnan(x).sum() for x in X)
        else:
            missing = int(np.isnan(X).sum()) if np.issubdtype(X.dtype, np.floating) else 0

        # Memory estimate
        if isinstance(X, list):
            memory_mb = sum(x.nbytes for x in X) / (1024 * 1024)
        else:
            memory_mb = X.nbytes / (1024 * 1024)

        return DatasetInfo(
            name=name,
            task_type=task_type,
            data_type=data_type,
            n_cases=n_cases,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            has_labels=y is not None,
            n_classes=n_classes,
            class_distribution=class_dist,
            is_univariate=n_channels == 1,
            is_equal_length=is_equal_length,
            missing_values=missing,
            memory_mb=round(memory_mb, 2),
        )

    def save_to_session(
        self,
        session_id: str,
        dataset_key: str,
        X: Any,
        y: Optional[Any],
        info: DatasetInfo,
    ) -> str:
        """Save dataset to session cache."""
        cache_key = f"session_{session_id}_{dataset_key}"
        self.cache[cache_key] = {
            "X": X,
            "y": y,
            "info": info.to_dict(),
        }
        return cache_key

    def load_from_session(
        self,
        session_id: str,
        dataset_key: str,
    ) -> Optional[Tuple[Any, Optional[Any], DatasetInfo]]:
        """Load dataset from session cache."""
        cache_key = f"session_{session_id}_{dataset_key}"
        if cache_key in self.cache:
            data = self.cache[cache_key]
            info = DatasetInfo(**data["info"])
            return data["X"], data["y"], info
        return None

    def get_session_datasets(self, session_id: str) -> List[str]:
        """Get list of datasets in a session."""
        prefix = f"session_{session_id}_"
        return [
            k.replace(prefix, "")
            for k in self.cache.iterkeys()
            if k.startswith(prefix)
        ]

    def export_dataset(
        self,
        X: Any,
        y: Optional[Any],
        info: DatasetInfo,
        format: str = "npz",
    ) -> bytes:
        """Export dataset to bytes."""
        if format == "npz":
            buffer = io.BytesIO()
            if y is not None:
                np.savez(buffer, X=X, y=y)
            else:
                np.savez(buffer, X=X)
            return buffer.getvalue()
        elif format == "json":
            metadata = info.to_dict()
            return json.dumps(metadata, indent=2).encode()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Singleton instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get singleton DatasetService instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
