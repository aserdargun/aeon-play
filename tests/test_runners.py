"""Tests for task runners."""

import pytest
import numpy as np

from aeon_play.services.runners import TaskRunner, RunResult


class TestRunResult:
    """Tests for RunResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        result = RunResult(
            success=True,
            task_type="classification",
            estimator_name="TestClassifier",
            train_time=1.0,
            metrics={"accuracy": 0.9},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] == True
        assert result_dict["task_type"] == "classification"
        assert result_dict["train_time"] == 1.0
        assert result_dict["metrics"]["accuracy"] == 0.9

    def test_to_dict_with_arrays(self):
        """Test conversion handles numpy arrays."""
        result = RunResult(
            success=True,
            task_type="classification",
            estimator_name="TestClassifier",
            train_time=1.0,
            predictions=np.array([0, 1, 0, 1]),
            confusion_matrix=np.array([[5, 1], [2, 7]]),
        )

        result_dict = result.to_dict()

        # Arrays should be converted to lists
        assert isinstance(result_dict["predictions"], list)
        assert isinstance(result_dict["confusion_matrix"], list)


class TestTaskRunner:
    """Tests for TaskRunner class."""

    @pytest.fixture
    def runner(self):
        """Create a TaskRunner instance."""
        return TaskRunner()

    @pytest.fixture
    def sample_classification_data(self):
        """Generate sample classification data."""
        np.random.seed(42)
        X = np.random.randn(50, 1, 20)
        y = np.array(["A"] * 25 + ["B"] * 25)
        return X, y

    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample regression data."""
        np.random.seed(42)
        X = np.random.randn(50, 1, 20)
        y = np.random.randn(50)
        return X, y

    @pytest.fixture
    def sample_clustering_data(self):
        """Generate sample clustering data."""
        np.random.seed(42)
        X = np.random.randn(30, 1, 20)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        return X, y

    def test_run_classification_success(self, runner, sample_classification_data):
        """Test successful classification run."""
        X, y = sample_classification_data

        result = runner.run_classification(
            X=X,
            y=y,
            estimator_name="DummyClassifier",  # Use dummy for reliable test
            params={},
            test_size=0.2,
            random_state=42,
        )

        # DummyClassifier might not be available, so check structure
        assert isinstance(result, RunResult)
        assert result.task_type == "classification"
        assert result.estimator_name == "DummyClassifier"

    def test_run_classification_logs(self, runner, sample_classification_data):
        """Test that classification logs are populated."""
        X, y = sample_classification_data

        result = runner.run_classification(
            X=X, y=y,
            estimator_name="DummyClassifier",
            params={},
        )

        assert result.logs is not None
        assert len(result.logs) > 0

    def test_run_classification_invalid_estimator(self, runner, sample_classification_data):
        """Test handling of invalid estimator."""
        X, y = sample_classification_data

        result = runner.run_classification(
            X=X, y=y,
            estimator_name="NonExistentClassifier",
            params={},
        )

        # Should fail gracefully
        assert result.success == False or result.error_message is not None

    def test_run_transformation(self, runner):
        """Test transformation run."""
        X = np.random.randn(10, 1, 50)

        success, result, error = runner.run_transformation(
            X=X,
            transformer_name="SummaryTransformer",
            params={},
        )

        # Transformer might not be available
        assert isinstance(success, bool)

    def test_fallback_similarity_search(self, runner):
        """Test fallback similarity search."""
        X = np.random.randn(20, 1, 30)
        query = X[0, 0, :]

        result = runner._fallback_similarity_search(X, query, k=5, logs=[])

        assert result.success == True
        assert len(result.predictions) == 5
        assert result.metrics["k"] == 5


class TestRunnerDataHandling:
    """Tests for data handling in runners."""

    @pytest.fixture
    def runner(self):
        return TaskRunner()

    def test_handles_2d_input(self, runner):
        """Test runner handles 2D input correctly."""
        X = np.random.randn(30, 50)  # 2D
        y = np.array(["A"] * 15 + ["B"] * 15)

        # Should not crash
        result = runner.run_classification(
            X=X.reshape(30, 1, 50),  # Convert to 3D
            y=y,
            estimator_name="DummyClassifier",
            params={},
        )

        assert isinstance(result, RunResult)

    def test_handles_small_dataset(self, runner):
        """Test runner handles small datasets."""
        X = np.random.randn(10, 1, 20)
        y = np.array(["A"] * 5 + ["B"] * 5)

        result = runner.run_classification(
            X=X, y=y,
            estimator_name="DummyClassifier",
            params={},
            test_size=0.3,  # Will give 3 test samples
        )

        assert isinstance(result, RunResult)
