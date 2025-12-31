"""Smoke tests for aeon-play application.

These tests verify that the application can start and basic
functionality works across all major modules.
"""

import pytest
import numpy as np


class TestSmoke:
    """Basic smoke tests to verify app functionality."""

    def test_app_imports(self):
        """Test that the app can be imported."""
        from aeon_play.app import app, server

        assert app is not None
        assert server is not None

    def test_services_import(self):
        """Test that services can be imported."""
        from aeon_play.services import (
            DatasetService,
            AeonDiscovery,
            DataConverter,
            FormGenerator,
            TaskRunner,
            CodeExporter,
        )

        assert DatasetService is not None
        assert AeonDiscovery is not None
        assert DataConverter is not None
        assert FormGenerator is not None
        assert TaskRunner is not None
        assert CodeExporter is not None

    def test_components_import(self):
        """Test that components can be imported."""
        from aeon_play.components import (
            create_learning_section,
            create_param_form,
            level_gate,
            create_data_preview,
            create_metrics_display,
        )

        assert create_learning_section is not None
        assert create_param_form is not None
        assert level_gate is not None
        assert create_data_preview is not None
        assert create_metrics_display is not None

    def test_dataset_service_basics(self):
        """Test basic dataset service functionality."""
        from aeon_play.services.datasets import get_dataset_service

        service = get_dataset_service()

        # Get available datasets
        datasets = service.get_available_datasets()
        assert "classification" in datasets

    def test_aeon_discovery_basics(self):
        """Test basic discovery functionality."""
        from aeon_play.services.aeon_discovery import get_aeon_discovery

        discovery = get_aeon_discovery()

        # Should be able to discover classifiers
        classifiers = discovery.discover_estimators("classifier", "beginner")
        assert len(classifiers) > 0

    def test_data_converter_basics(self):
        """Test basic converter functionality."""
        from aeon_play.services.converters import DataConverter

        X = np.random.randn(10, 1, 50)
        info = DataConverter.detect_format(X)

        assert info["format"] == "numpy_3d"
        assert info["n_cases"] == 10

    def test_task_runner_smoke(self):
        """Test that task runner can be instantiated."""
        from aeon_play.services.runners import TaskRunner

        runner = TaskRunner()

        # Simple transformation test
        X = np.random.randn(5, 1, 20)
        success, result, error = runner.run_transformation(
            X=X,
            transformer_name="SummaryTransformer",
            params={},
        )

        # Just check it doesn't crash
        assert isinstance(success, bool)

    def test_code_exporter_smoke(self):
        """Test that code exporter works."""
        from aeon_play.services.exporters import CodeExporter

        exporter = CodeExporter()

        code = exporter.generate_classification_code(
            estimator_name="KNeighborsTimeSeriesClassifier",
            estimator_module="aeon.classification.distance_based",
            params={"n_neighbors": 5},
            dataset_name="GunPoint",
        )

        assert "KNeighborsTimeSeriesClassifier" in code
        assert "GunPoint" in code

    def test_level_gating(self):
        """Test level gating functionality."""
        from aeon_play.components.level_gate import level_gate, get_level_features

        # Test level gating
        content = "Test content"
        visible = level_gate(content, "beginner", "intermediate")
        assert visible == content

        hidden = level_gate(content, "advanced", "beginner")
        assert hidden is None or hidden != content

        # Test feature flags
        beginner_features = get_level_features("beginner")
        advanced_features = get_level_features("advanced")

        assert beginner_features["builtin_datasets"] == True
        assert advanced_features["all_estimators"] == True

    def test_form_generator(self):
        """Test form generator."""
        from aeon_play.services.forms import FormGenerator, create_quick_form

        generator = FormGenerator()

        # Test quick form creation
        params = {"n_neighbors": 5, "metric": "dtw"}
        form = create_quick_form(params, prefix="test")

        assert len(form) > 0

    def test_learning_section_component(self):
        """Test learning section component."""
        from aeon_play.components.learning_section import create_learning_section

        section = create_learning_section(
            title="Test Title",
            content="Test content",
            level="beginner",
        )

        assert section is not None

    def test_metrics_display_component(self):
        """Test metrics display component."""
        from aeon_play.components.metrics_display import create_metrics_display

        metrics = {"accuracy": 0.95, "f1_weighted": 0.93}
        display = create_metrics_display(metrics, "classification")

        assert display is not None


class TestTaskModulesSmoke:
    """Smoke tests for task module functionality."""

    def test_classification_pipeline(self):
        """Test classification pipeline end-to-end."""
        from aeon_play.services.runners import TaskRunner

        np.random.seed(42)
        X = np.random.randn(30, 1, 20)
        y = np.array(["A"] * 15 + ["B"] * 15)

        runner = TaskRunner()
        result = runner.run_classification(
            X=X, y=y,
            estimator_name="DummyClassifier",
            params={},
        )

        assert result.task_type == "classification"

    def test_clustering_pipeline(self):
        """Test clustering pipeline."""
        from aeon_play.services.runners import TaskRunner

        np.random.seed(42)
        X = np.random.randn(20, 1, 20)

        runner = TaskRunner()
        result = runner.run_clustering(
            X=X,
            estimator_name="TimeSeriesKMeans",
            params={"n_clusters": 3},
        )

        assert result.task_type == "clustering"

    def test_similarity_search_fallback(self):
        """Test similarity search with fallback."""
        from aeon_play.services.runners import TaskRunner

        np.random.seed(42)
        X = np.random.randn(20, 1, 30)
        query = X[0]

        runner = TaskRunner()
        result = runner.run_similarity_search(
            X=X,
            query=query,
            estimator_name="NonExistent",
            params={},
            k=5,
        )

        # Should fall back to Euclidean NN
        assert result.success == True
        assert len(result.predictions) == 5


class TestExporterSmoke:
    """Smoke tests for code export functionality."""

    def test_all_code_generators(self):
        """Test all code generation methods."""
        from aeon_play.services.exporters import CodeExporter

        exporter = CodeExporter()

        # Classification
        code = exporter.generate_classification_code(
            "TestClassifier", "aeon.classification", {}, "TestData"
        )
        assert "TestClassifier" in code

        # Regression
        code = exporter.generate_regression_code(
            "TestRegressor", "aeon.regression", {}, "TestData"
        )
        assert "TestRegressor" in code

        # Clustering
        code = exporter.generate_clustering_code(
            "TestClusterer", "aeon.clustering", {}
        )
        assert "TestClusterer" in code

        # Forecasting
        code = exporter.generate_forecasting_code(
            "TestForecaster", "aeon.forecasting", {}, horizon=10
        )
        assert "TestForecaster" in code

        # Anomaly detection
        code = exporter.generate_anomaly_detection_code(
            "TestDetector", "aeon.anomaly_detection", {}
        )
        assert "TestDetector" in code

        # Segmentation
        code = exporter.generate_segmentation_code(
            "TestSegmenter", "aeon.segmentation", {}
        )
        assert "TestSegmenter" in code

        # Transformation
        code = exporter.generate_transformation_code(
            "TestTransformer", "aeon.transformations", {}
        )
        assert "TestTransformer" in code

        # Distance
        code = exporter.generate_distance_code("dtw", {"window": 0.1})
        assert "dtw" in code
