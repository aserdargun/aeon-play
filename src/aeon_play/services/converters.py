"""
Data conversion and validation utilities for aeon-play.

Handles detection and conversion between different time series data formats.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataFormatError(Exception):
    """Exception for data format issues."""
    pass


class DataConverter:
    """Service for detecting and converting time series data formats."""

    @staticmethod
    def detect_format(data: Any) -> Dict[str, Any]:
        """
        Detect the format of input data.

        Returns a dict with:
        - format: str (numpy_3d, numpy_2d, list_of_arrays, pandas_df, series)
        - n_cases: int
        - n_channels: int
        - n_timepoints: int or "variable"
        - is_univariate: bool
        - is_equal_length: bool
        - dtype: str
        """
        result = {
            "format": "unknown",
            "n_cases": 0,
            "n_channels": 1,
            "n_timepoints": 0,
            "is_univariate": True,
            "is_equal_length": True,
            "dtype": str(type(data)),
        }

        if isinstance(data, np.ndarray):
            result["dtype"] = str(data.dtype)

            if data.ndim == 3:
                result["format"] = "numpy_3d"
                result["n_cases"] = data.shape[0]
                result["n_channels"] = data.shape[1]
                result["n_timepoints"] = data.shape[2]
                result["is_univariate"] = data.shape[1] == 1

            elif data.ndim == 2:
                result["format"] = "numpy_2d"
                result["n_cases"] = data.shape[0]
                result["n_timepoints"] = data.shape[1]

            elif data.ndim == 1:
                result["format"] = "series"
                result["n_cases"] = 1
                result["n_timepoints"] = len(data)

        elif isinstance(data, list):
            if len(data) == 0:
                result["format"] = "empty_list"
            elif isinstance(data[0], np.ndarray):
                result["format"] = "list_of_arrays"
                result["n_cases"] = len(data)

                # Check for equal length
                lengths = [arr.shape[-1] if arr.ndim > 1 else len(arr) for arr in data]
                result["is_equal_length"] = len(set(lengths)) == 1
                result["n_timepoints"] = lengths[0] if result["is_equal_length"] else "variable"

                # Check channels
                if data[0].ndim > 1:
                    result["n_channels"] = data[0].shape[0]
                    result["is_univariate"] = data[0].shape[0] == 1

        elif isinstance(data, pd.DataFrame):
            result["format"] = "pandas_df"
            result["n_cases"] = len(data)
            result["n_timepoints"] = len(data.columns)
            result["dtype"] = str(data.dtypes.iloc[0]) if len(data.columns) > 0 else "unknown"

        elif isinstance(data, pd.Series):
            result["format"] = "pandas_series"
            result["n_cases"] = 1
            result["n_timepoints"] = len(data)

        return result

    @staticmethod
    def to_numpy_3d(
        data: Any,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Convert data to numpy 3D format (n_cases, n_channels, n_timepoints).

        Args:
            data: Input data in various formats
            copy: Whether to copy the data

        Returns:
            numpy array with shape (n_cases, n_channels, n_timepoints)
        """
        format_info = DataConverter.detect_format(data)
        fmt = format_info["format"]

        if fmt == "numpy_3d":
            return data.copy() if copy else data

        elif fmt == "numpy_2d":
            # Assume (n_cases, n_timepoints), add channel dimension
            result = data.reshape(data.shape[0], 1, data.shape[1])
            return result.copy() if copy else result

        elif fmt == "series":
            # Single series
            return data.reshape(1, 1, -1).copy() if copy else data.reshape(1, 1, -1)

        elif fmt == "list_of_arrays":
            if format_info["is_equal_length"]:
                # Can stack into 3D array
                arrays = []
                for arr in data:
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    arrays.append(arr)
                return np.stack(arrays)
            else:
                raise DataFormatError(
                    "Cannot convert unequal-length list to numpy 3D. "
                    "Use list format or resample to equal length."
                )

        elif fmt == "pandas_df":
            # Assume rows are cases, columns are timepoints
            values = data.values
            return values.reshape(values.shape[0], 1, values.shape[1])

        elif fmt == "pandas_series":
            return data.values.reshape(1, 1, -1)

        else:
            raise DataFormatError(f"Cannot convert format '{fmt}' to numpy 3D")

    @staticmethod
    def to_single_series(data: Any) -> np.ndarray:
        """
        Extract or convert to a single 1D series.

        Args:
            data: Input data

        Returns:
            1D numpy array
        """
        format_info = DataConverter.detect_format(data)
        fmt = format_info["format"]

        if fmt == "series":
            return data.flatten()

        elif fmt == "numpy_3d":
            if data.shape[0] == 1 and data.shape[1] == 1:
                return data[0, 0, :]
            raise DataFormatError(
                f"Data has {data.shape[0]} cases and {data.shape[1]} channels. "
                "Select a single case and channel first."
            )

        elif fmt == "numpy_2d":
            if data.shape[0] == 1:
                return data[0, :]
            raise DataFormatError(
                f"Data has {data.shape[0]} cases. Select a single case first."
            )

        elif fmt == "pandas_series":
            return data.values

        elif fmt == "pandas_df":
            if len(data) == 1:
                return data.iloc[0].values
            raise DataFormatError("DataFrame has multiple rows. Select a single row first.")

        else:
            raise DataFormatError(f"Cannot extract single series from format '{fmt}'")

    @staticmethod
    def extract_case(
        data: Any,
        case_idx: int,
        channel_idx: int = 0,
    ) -> np.ndarray:
        """
        Extract a single case/channel from collection data.

        Args:
            data: Input collection data
            case_idx: Index of case to extract
            channel_idx: Index of channel to extract

        Returns:
            1D numpy array
        """
        format_info = DataConverter.detect_format(data)
        fmt = format_info["format"]

        if fmt == "numpy_3d":
            return data[case_idx, channel_idx, :]

        elif fmt == "numpy_2d":
            return data[case_idx, :]

        elif fmt == "list_of_arrays":
            arr = data[case_idx]
            if arr.ndim > 1:
                return arr[channel_idx, :]
            return arr

        else:
            raise DataFormatError(f"Cannot extract case from format '{fmt}'")

    @staticmethod
    def validate_for_task(
        data: Any,
        task_type: str,
    ) -> Tuple[bool, str]:
        """
        Validate data format for a specific task.

        Args:
            data: Input data
            task_type: Task type (classification, regression, forecasting, etc.)

        Returns:
            Tuple of (is_valid, message)
        """
        format_info = DataConverter.detect_format(data)

        collection_tasks = {"classification", "regression", "clustering", "similarity"}
        single_series_tasks = {"forecasting", "anomaly", "segmentation"}

        if task_type in collection_tasks:
            if format_info["n_cases"] < 2:
                return False, f"Task '{task_type}' requires a collection with multiple cases"

        if task_type in single_series_tasks:
            if format_info["n_cases"] > 1:
                return True, f"Data has {format_info['n_cases']} cases; will use first case"

        if format_info["n_timepoints"] == 0:
            return False, "Data has no timepoints"

        return True, "Data format is valid"

    @staticmethod
    def resample_to_length(
        data: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Resample series to a target length using linear interpolation.

        Args:
            data: Input data (1D, 2D, or 3D)
            target_length: Target number of timepoints

        Returns:
            Resampled data with same shape except last dimension
        """
        from scipy import interpolate

        original_shape = data.shape
        target_shape = list(original_shape)
        target_shape[-1] = target_length

        # Flatten to 2D for processing
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])

        result = np.zeros((data.shape[0], target_length))

        for i in range(data.shape[0]):
            x_old = np.linspace(0, 1, len(data[i]))
            x_new = np.linspace(0, 1, target_length)
            f = interpolate.interp1d(x_old, data[i], kind='linear')
            result[i] = f(x_new)

        # Reshape back
        result = result.reshape(target_shape)

        return result

    @staticmethod
    def pad_to_length(
        data: np.ndarray,
        target_length: int,
        pad_value: float = 0.0,
        pad_mode: str = "end",
    ) -> np.ndarray:
        """
        Pad series to a target length.

        Args:
            data: Input data
            target_length: Target number of timepoints
            pad_value: Value to use for padding
            pad_mode: Where to pad ("end", "start", "both")

        Returns:
            Padded data
        """
        current_length = data.shape[-1]

        if current_length >= target_length:
            return data[..., :target_length]

        pad_amount = target_length - current_length

        if data.ndim == 1:
            if pad_mode == "end":
                pad_width = (0, pad_amount)
            elif pad_mode == "start":
                pad_width = (pad_amount, 0)
            else:
                pad_width = (pad_amount // 2, pad_amount - pad_amount // 2)
            return np.pad(data, pad_width, mode='constant', constant_values=pad_value)

        elif data.ndim == 2:
            if pad_mode == "end":
                pad_width = ((0, 0), (0, pad_amount))
            elif pad_mode == "start":
                pad_width = ((0, 0), (pad_amount, 0))
            else:
                pad_width = ((0, 0), (pad_amount // 2, pad_amount - pad_amount // 2))
            return np.pad(data, pad_width, mode='constant', constant_values=pad_value)

        elif data.ndim == 3:
            if pad_mode == "end":
                pad_width = ((0, 0), (0, 0), (0, pad_amount))
            elif pad_mode == "start":
                pad_width = ((0, 0), (0, 0), (pad_amount, 0))
            else:
                pad_width = ((0, 0), (0, 0), (pad_amount // 2, pad_amount - pad_amount // 2))
            return np.pad(data, pad_width, mode='constant', constant_values=pad_value)

        return data

    @staticmethod
    def truncate_to_length(data: np.ndarray, target_length: int) -> np.ndarray:
        """Truncate series to target length."""
        return data[..., :target_length]

    @staticmethod
    def normalize_series(
        data: np.ndarray,
        method: str = "zscore",
    ) -> np.ndarray:
        """
        Normalize time series data.

        Args:
            data: Input data
            method: Normalization method ("zscore", "minmax", "robust")

        Returns:
            Normalized data
        """
        if method == "zscore":
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            std = np.where(std == 0, 1, std)
            return (data - mean) / std

        elif method == "minmax":
            min_val = np.min(data, axis=-1, keepdims=True)
            max_val = np.max(data, axis=-1, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            return (data - min_val) / range_val

        elif method == "robust":
            median = np.median(data, axis=-1, keepdims=True)
            q1 = np.percentile(data, 25, axis=-1, keepdims=True)
            q3 = np.percentile(data, 75, axis=-1, keepdims=True)
            iqr = q3 - q1
            iqr = np.where(iqr == 0, 1, iqr)
            return (data - median) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def get_data_contract(data: Any, y: Any = None) -> Dict[str, Any]:
        """
        Generate a 'data contract' describing the dataset format and compatibility.

        Args:
            data: Input data (X)
            y: Optional labels/targets

        Returns:
            Dict describing the data contract
        """
        format_info = DataConverter.detect_format(data)

        contract = {
            "format": format_info["format"],
            "shape": {
                "n_cases": format_info["n_cases"],
                "n_channels": format_info["n_channels"],
                "n_timepoints": format_info["n_timepoints"],
            },
            "properties": {
                "is_univariate": format_info["is_univariate"],
                "is_equal_length": format_info["is_equal_length"],
                "dtype": format_info["dtype"],
            },
            "compatible_tasks": [],
            "warnings": [],
        }

        # Determine compatible tasks
        if format_info["n_cases"] > 1:
            contract["compatible_tasks"].extend([
                "classification", "regression", "clustering", "similarity_search"
            ])

        if format_info["n_cases"] >= 1:
            contract["compatible_tasks"].extend([
                "forecasting", "anomaly_detection", "segmentation"
            ])

        # Add warnings
        if not format_info["is_equal_length"]:
            contract["warnings"].append(
                "Unequal length series detected. Some algorithms may not support this."
            )

        if format_info["n_cases"] > 5000:
            contract["warnings"].append(
                f"Large dataset ({format_info['n_cases']} cases). Some algorithms may be slow."
            )

        if isinstance(format_info["n_timepoints"], int) and format_info["n_timepoints"] > 2000:
            contract["warnings"].append(
                f"Long series ({format_info['n_timepoints']} timepoints). Consider resampling."
            )

        # Add label info
        if y is not None:
            contract["labels"] = {
                "present": True,
                "n_unique": len(np.unique(y)) if isinstance(y, np.ndarray) else None,
                "dtype": str(y.dtype) if isinstance(y, np.ndarray) else str(type(y)),
            }
        else:
            contract["labels"] = {"present": False}

        return contract
