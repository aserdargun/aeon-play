"""Tests for aeon discovery service."""

import pytest

from aeon_play.services.aeon_discovery import (
    AeonDiscovery,
    get_aeon_discovery,
    get_distance_functions,
    DISTANCE_FUNCTIONS,
)


class TestAeonDiscovery:
    """Tests for AeonDiscovery class."""

    def test_singleton(self):
        """Test singleton pattern."""
        discovery1 = get_aeon_discovery()
        discovery2 = get_aeon_discovery()
        assert discovery1 is discovery2

    def test_discover_classifiers(self):
        """Test classifier discovery."""
        discovery = AeonDiscovery()
        classifiers = discovery.discover_estimators("classifier", "beginner")

        assert len(classifiers) > 0
        assert all(c.estimator_type == "classifier" for c in classifiers)

    def test_discover_regressors(self):
        """Test regressor discovery."""
        discovery = AeonDiscovery()
        regressors = discovery.discover_estimators("regressor", "beginner")

        assert len(regressors) > 0
        assert all(r.estimator_type == "regressor" for r in regressors)

    def test_discover_clusterers(self):
        """Test clusterer discovery."""
        discovery = AeonDiscovery()
        clusterers = discovery.discover_estimators("clusterer", "beginner")

        assert len(clusterers) > 0
        assert all(c.estimator_type == "clusterer" for c in clusterers)

    def test_level_filtering(self):
        """Test that level filtering works."""
        discovery = AeonDiscovery()

        beginner = discovery.discover_estimators("classifier", "beginner")
        advanced = discovery.discover_estimators("classifier", "advanced")

        # Advanced should have at least as many as beginner
        assert len(advanced) >= len(beginner)

    def test_return_names_only(self):
        """Test returning only names."""
        discovery = AeonDiscovery()
        names = discovery.discover_estimators("classifier", "beginner", return_names_only=True)

        assert all(isinstance(n, str) for n in names)

    def test_estimator_info_fields(self):
        """Test EstimatorInfo has required fields."""
        discovery = AeonDiscovery()
        estimators = discovery.discover_estimators("classifier", "beginner")

        if estimators:
            est = estimators[0]
            assert hasattr(est, "name")
            assert hasattr(est, "class_name")
            assert hasattr(est, "module")
            assert hasattr(est, "estimator_type")
            assert hasattr(est, "tags")
            assert hasattr(est, "description")
            assert hasattr(est, "level")

    def test_get_estimator_params(self):
        """Test getting estimator parameters."""
        discovery = AeonDiscovery()
        params = discovery.get_estimator_params("KNeighborsTimeSeriesClassifier", "classifier")

        # Should return dict with param info
        assert isinstance(params, dict)

    def test_caching(self):
        """Test that discovery results are cached."""
        discovery = AeonDiscovery()

        # First call
        result1 = discovery.discover_estimators("classifier", "beginner")

        # Second call should use cache
        result2 = discovery.discover_estimators("classifier", "beginner")

        # Cache key should exist
        assert "classifier_beginner" in discovery._cache


class TestDistanceFunctions:
    """Tests for distance function utilities."""

    def test_get_distance_functions_beginner(self):
        """Test getting beginner distances."""
        distances = get_distance_functions("beginner")

        assert len(distances) >= 2
        names = [d["name"] for d in distances]
        assert "euclidean" in names
        assert "dtw" in names

    def test_get_distance_functions_advanced(self):
        """Test getting advanced distances."""
        distances = get_distance_functions("advanced")

        # Should have more distances
        assert len(distances) > len(get_distance_functions("beginner"))

    def test_distance_function_structure(self):
        """Test distance function info structure."""
        distances = get_distance_functions("beginner")

        for d in distances:
            assert "name" in d
            assert "display_name" in d
            assert "description" in d
            assert "params" in d

    def test_distance_functions_dict_structure(self):
        """Test DISTANCE_FUNCTIONS dict structure."""
        assert "beginner" in DISTANCE_FUNCTIONS
        assert "intermediate" in DISTANCE_FUNCTIONS
        assert "advanced" in DISTANCE_FUNCTIONS
