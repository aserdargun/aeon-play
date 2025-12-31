"""Tests for data converter utilities."""

import pytest
import numpy as np

from aeon_play.services.converters import DataConverter, DataFormatError


class TestDataConverter:
    """Tests for DataConverter class."""

    def test_detect_format_numpy_3d(self):
        """Test detection of 3D numpy array format."""
        X = np.random.randn(10, 2, 50)
        result = DataConverter.detect_format(X)

        assert result["format"] == "numpy_3d"
        assert result["n_cases"] == 10
        assert result["n_channels"] == 2
        assert result["n_timepoints"] == 50
        assert result["is_univariate"] == False
        assert result["is_equal_length"] == True

    def test_detect_format_numpy_2d(self):
        """Test detection of 2D numpy array format."""
        X = np.random.randn(10, 50)
        result = DataConverter.detect_format(X)

        assert result["format"] == "numpy_2d"
        assert result["n_cases"] == 10
        assert result["n_timepoints"] == 50

    def test_detect_format_1d(self):
        """Test detection of 1D numpy array (single series)."""
        X = np.random.randn(100)
        result = DataConverter.detect_format(X)

        assert result["format"] == "series"
        assert result["n_cases"] == 1
        assert result["n_timepoints"] == 100

    def test_detect_format_list_of_arrays(self):
        """Test detection of list of arrays format."""
        X = [np.random.randn(50) for _ in range(10)]
        result = DataConverter.detect_format(X)

        assert result["format"] == "list_of_arrays"
        assert result["n_cases"] == 10
        assert result["is_equal_length"] == True

    def test_detect_format_unequal_length(self):
        """Test detection of unequal length series."""
        X = [np.random.randn(50 + i) for i in range(10)]
        result = DataConverter.detect_format(X)

        assert result["format"] == "list_of_arrays"
        assert result["is_equal_length"] == False
        assert result["n_timepoints"] == "variable"

    def test_to_numpy_3d_from_2d(self):
        """Test conversion from 2D to 3D."""
        X = np.random.randn(10, 50)
        result = DataConverter.to_numpy_3d(X)

        assert result.shape == (10, 1, 50)

    def test_to_numpy_3d_from_1d(self):
        """Test conversion from 1D to 3D."""
        X = np.random.randn(100)
        result = DataConverter.to_numpy_3d(X)

        assert result.shape == (1, 1, 100)

    def test_to_numpy_3d_from_3d(self):
        """Test that 3D stays 3D."""
        X = np.random.randn(10, 2, 50)
        result = DataConverter.to_numpy_3d(X)

        assert result.shape == (10, 2, 50)

    def test_to_single_series(self):
        """Test extraction of single series."""
        X = np.random.randn(1, 1, 100)
        result = DataConverter.to_single_series(X)

        assert result.shape == (100,)

    def test_to_single_series_error(self):
        """Test error when multiple cases/channels."""
        X = np.random.randn(10, 1, 100)

        with pytest.raises(DataFormatError):
            DataConverter.to_single_series(X)

    def test_extract_case(self):
        """Test extraction of specific case."""
        X = np.random.randn(10, 2, 50)
        result = DataConverter.extract_case(X, 5, 1)

        assert result.shape == (50,)
        np.testing.assert_array_equal(result, X[5, 1, :])

    def test_validate_for_task_classification(self):
        """Test validation for classification task."""
        X = np.random.randn(10, 1, 50)
        valid, msg = DataConverter.validate_for_task(X, "classification")

        assert valid == True

    def test_validate_for_task_single_case(self):
        """Test validation fails for classification with single case."""
        X = np.random.randn(1, 1, 50)
        valid, msg = DataConverter.validate_for_task(X, "classification")

        assert valid == False
        assert "multiple cases" in msg

    def test_resample_to_length(self):
        """Test resampling series to target length."""
        X = np.random.randn(10, 1, 100)
        result = DataConverter.resample_to_length(X, 50)

        assert result.shape == (10, 1, 50)

    def test_pad_to_length(self):
        """Test padding series to target length."""
        X = np.random.randn(10, 1, 50)
        result = DataConverter.pad_to_length(X, 100)

        assert result.shape == (10, 1, 100)
        # Check padding is zeros
        np.testing.assert_array_equal(result[:, :, 50:], 0)

    def test_truncate_to_length(self):
        """Test truncating series."""
        X = np.random.randn(10, 1, 100)
        result = DataConverter.truncate_to_length(X, 50)

        assert result.shape == (10, 1, 50)

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        X = np.random.randn(10, 1, 100) * 10 + 50
        result = DataConverter.normalize_series(X, method="zscore")

        # Check mean is approximately 0 and std is approximately 1
        assert np.abs(np.mean(result)) < 0.1
        assert np.abs(np.std(result) - 1) < 0.1

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        X = np.random.randn(10, 1, 100) * 10 + 50
        result = DataConverter.normalize_series(X, method="minmax")

        # Check values are in [0, 1] range
        assert result.min() >= 0
        assert result.max() <= 1

    def test_get_data_contract(self):
        """Test data contract generation."""
        X = np.random.randn(10, 2, 50)
        y = np.array(["A"] * 5 + ["B"] * 5)

        contract = DataConverter.get_data_contract(X, y)

        assert contract["format"] == "numpy_3d"
        assert contract["shape"]["n_cases"] == 10
        assert contract["shape"]["n_channels"] == 2
        assert contract["properties"]["is_univariate"] == False
        assert "classification" in contract["compatible_tasks"]
        assert contract["labels"]["present"] == True
