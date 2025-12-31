"""
Aeon estimator discovery service.

Uses aeon's registry utilities to dynamically discover available
estimators, transformers, and their capabilities.
"""

from typing import Dict, List, Optional, Any, Tuple, Type
from dataclasses import dataclass
import inspect

try:
    from aeon.registry import all_estimators
    from aeon.base import BaseEstimator
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False
    all_estimators = None


@dataclass
class EstimatorInfo:
    """Information about an aeon estimator."""
    name: str
    class_name: str
    module: str
    estimator_type: str
    tags: Dict[str, Any]
    description: str
    level: str  # beginner, intermediate, advanced
    is_available: bool = True

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "module": self.module,
            "estimator_type": self.estimator_type,
            "tags": self.tags,
            "description": self.description,
            "level": self.level,
            "is_available": self.is_available,
        }


class AeonDiscovery:
    """Service for discovering aeon estimators and their capabilities."""

    # Curated lists for beginner level
    BEGINNER_CLASSIFIERS = [
        "KNeighborsTimeSeriesClassifier",
        "TimeSeriesForestClassifier",
        "RocketClassifier",
        "DummyClassifier",
    ]

    BEGINNER_REGRESSORS = [
        "KNeighborsTimeSeriesRegressor",
        "TimeSeriesForestRegressor",
        "RocketRegressor",
    ]

    BEGINNER_CLUSTERERS = [
        "TimeSeriesKMeans",
        "TimeSeriesKMedoids",
    ]

    BEGINNER_FORECASTERS = [
        "NaiveForecaster",
        "TrendForecaster",
    ]

    BEGINNER_TRANSFORMERS = [
        "Catch22",
        "SummaryTransformer",
        "TSFreshRelevantFeatureExtractor",
    ]

    BEGINNER_ANOMALY_DETECTORS = [
        "STRAY",
        "IsolationForest",
    ]

    BEGINNER_SEGMENTERS = [
        "ClaSPSegmenter",
    ]

    # Intermediate additions
    INTERMEDIATE_CLASSIFIERS = [
        "ShapeDTW",
        "Arsenal",
        "HIVECOTEV2",
        "MiniRocket",
    ]

    INTERMEDIATE_REGRESSORS = [
        "MiniRocketRegressor",
    ]

    # Type mappings for aeon
    TYPE_FILTERS = {
        "classifier": "classifier",
        "regressor": "regressor",
        "clusterer": "clusterer",
        "forecaster": "forecaster",
        "transformer": "transformer",
        "anomaly_detector": "anomaly-detector",
        "segmenter": "segmenter",
        "similarity_search": "similarity-search",
    }

    def __init__(self):
        """Initialize the discovery service."""
        self._cache: Dict[str, List[EstimatorInfo]] = {}

    def discover_estimators(
        self,
        estimator_type: str,
        level: str = "advanced",
        return_names_only: bool = False,
    ) -> List[EstimatorInfo]:
        """
        Discover available estimators of a given type.

        Args:
            estimator_type: Type of estimator (classifier, regressor, etc.)
            level: User level (beginner, intermediate, advanced)
            return_names_only: If True, return list of names instead of EstimatorInfo

        Returns:
            List of EstimatorInfo or names
        """
        cache_key = f"{estimator_type}_{level}"

        if cache_key not in self._cache:
            self._cache[cache_key] = self._discover_type(estimator_type, level)

        if return_names_only:
            return [e.name for e in self._cache[cache_key]]

        return self._cache[cache_key]

    def _discover_type(self, estimator_type: str, level: str) -> List[EstimatorInfo]:
        """Discover estimators of a specific type."""
        estimators = []

        if not AEON_AVAILABLE:
            return self._get_fallback_estimators(estimator_type, level)

        try:
            type_filter = self.TYPE_FILTERS.get(estimator_type, estimator_type)

            # Get all estimators of this type
            all_ests = all_estimators(type_filter=type_filter)

            for name, est_class in all_ests:
                info = self._create_estimator_info(name, est_class, estimator_type)
                if info and self._matches_level(info, level, estimator_type):
                    estimators.append(info)

        except Exception as e:
            print(f"Error discovering {estimator_type}: {e}")
            return self._get_fallback_estimators(estimator_type, level)

        # Sort by name
        estimators.sort(key=lambda x: x.name)

        return estimators

    def _create_estimator_info(
        self,
        name: str,
        est_class: Type,
        estimator_type: str,
    ) -> Optional[EstimatorInfo]:
        """Create EstimatorInfo from an estimator class."""
        try:
            # Get tags
            tags = {}
            if hasattr(est_class, "get_class_tags"):
                try:
                    tags = est_class.get_class_tags()
                except Exception:
                    pass
            elif hasattr(est_class, "_tags"):
                tags = est_class._tags.copy() if est_class._tags else {}

            # Get docstring for description
            doc = est_class.__doc__ or ""
            description = doc.split("\n")[0].strip() if doc else f"{name} estimator"

            # Determine level
            level = self._determine_level(name, estimator_type, tags)

            return EstimatorInfo(
                name=name,
                class_name=est_class.__name__,
                module=est_class.__module__,
                estimator_type=estimator_type,
                tags=tags,
                description=description,
                level=level,
                is_available=True,
            )

        except Exception as e:
            print(f"Error creating info for {name}: {e}")
            return None

    def _determine_level(
        self,
        name: str,
        estimator_type: str,
        tags: Dict,
    ) -> str:
        """Determine the appropriate level for an estimator."""
        # Check beginner lists
        beginner_lists = {
            "classifier": self.BEGINNER_CLASSIFIERS,
            "regressor": self.BEGINNER_REGRESSORS,
            "clusterer": self.BEGINNER_CLUSTERERS,
            "forecaster": self.BEGINNER_FORECASTERS,
            "transformer": self.BEGINNER_TRANSFORMERS,
            "anomaly_detector": self.BEGINNER_ANOMALY_DETECTORS,
            "segmenter": self.BEGINNER_SEGMENTERS,
        }

        intermediate_lists = {
            "classifier": self.INTERMEDIATE_CLASSIFIERS,
            "regressor": self.INTERMEDIATE_REGRESSORS,
        }

        if estimator_type in beginner_lists:
            if name in beginner_lists[estimator_type]:
                return "beginner"

        if estimator_type in intermediate_lists:
            if name in intermediate_lists[estimator_type]:
                return "intermediate"

        # Check complexity indicators in tags
        if tags.get("python_dependencies"):
            return "advanced"

        if tags.get("algorithm_type") in ["deep_learning", "ensemble"]:
            return "advanced"

        return "intermediate"

    def _matches_level(
        self,
        info: EstimatorInfo,
        user_level: str,
        estimator_type: str,
    ) -> bool:
        """Check if estimator matches user level."""
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
        return level_order.get(info.level, 2) <= level_order.get(user_level, 2)

    def _get_fallback_estimators(
        self,
        estimator_type: str,
        level: str,
    ) -> List[EstimatorInfo]:
        """Return fallback list when aeon is not available."""
        fallback_map = {
            "classifier": [
                ("KNeighborsTimeSeriesClassifier", "Distance-based classifier using k-nearest neighbors"),
                ("TimeSeriesForestClassifier", "Ensemble classifier using interval features"),
                ("RocketClassifier", "Fast classifier using random convolutional kernels"),
            ],
            "regressor": [
                ("KNeighborsTimeSeriesRegressor", "Distance-based regressor using k-nearest neighbors"),
                ("TimeSeriesForestRegressor", "Ensemble regressor using interval features"),
            ],
            "clusterer": [
                ("TimeSeriesKMeans", "K-means clustering for time series"),
                ("TimeSeriesKMedoids", "K-medoids clustering for time series"),
            ],
            "forecaster": [
                ("NaiveForecaster", "Simple baseline forecaster"),
                ("TrendForecaster", "Trend-based forecaster"),
            ],
            "transformer": [
                ("Catch22", "Catch22 feature extraction"),
                ("SummaryTransformer", "Summary statistics transformer"),
            ],
            "anomaly_detector": [
                ("STRAY", "Search TRace AnomalY detector"),
            ],
            "segmenter": [
                ("ClaSPSegmenter", "ClaSP-based segmenter"),
            ],
        }

        estimators = []
        for name, desc in fallback_map.get(estimator_type, []):
            estimators.append(EstimatorInfo(
                name=name,
                class_name=name,
                module="aeon",
                estimator_type=estimator_type,
                tags={},
                description=desc,
                level="beginner",
                is_available=False,
            ))

        return estimators

    def get_estimator_class(self, name: str, estimator_type: str) -> Optional[Type]:
        """Get the estimator class by name."""
        if not AEON_AVAILABLE:
            return None

        try:
            type_filter = self.TYPE_FILTERS.get(estimator_type, estimator_type)
            all_ests = all_estimators(type_filter=type_filter)

            for est_name, est_class in all_ests:
                if est_name == name:
                    return est_class

        except Exception:
            pass

        return None

    def get_estimator_params(self, name: str, estimator_type: str) -> Dict[str, Any]:
        """Get default parameters for an estimator."""
        est_class = self.get_estimator_class(name, estimator_type)

        if est_class is None:
            return {}

        try:
            # Get signature of __init__
            sig = inspect.signature(est_class.__init__)
            params = {}

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "args", "kwargs"):
                    continue

                default = param.default
                if default is inspect.Parameter.empty:
                    default = None

                # Get type annotation if available
                annotation = param.annotation
                if annotation is inspect.Parameter.empty:
                    annotation = None
                else:
                    annotation = str(annotation)

                params[param_name] = {
                    "default": default if not callable(default) else None,
                    "type": annotation,
                    "required": param.default is inspect.Parameter.empty,
                }

            return params

        except Exception as e:
            print(f"Error getting params for {name}: {e}")
            return {}

    def get_estimator_docstring(self, name: str, estimator_type: str) -> str:
        """Get the docstring for an estimator."""
        est_class = self.get_estimator_class(name, estimator_type)

        if est_class is None:
            return ""

        return est_class.__doc__ or ""

    def get_param_docstrings(self, name: str, estimator_type: str) -> Dict[str, str]:
        """Extract parameter descriptions from docstring."""
        docstring = self.get_estimator_docstring(name, estimator_type)

        if not docstring:
            return {}

        param_docs = {}
        in_params = False
        current_param = None
        current_desc = []

        for line in docstring.split("\n"):
            stripped = line.strip()

            if stripped.startswith("Parameters"):
                in_params = True
                continue

            if in_params:
                if stripped.startswith("---") or stripped.startswith("==="):
                    continue

                if stripped.startswith("Attributes") or stripped.startswith("Returns"):
                    break

                # Check for new parameter
                if ":" in stripped and not stripped.startswith(" "):
                    if current_param:
                        param_docs[current_param] = " ".join(current_desc).strip()

                    parts = stripped.split(":", 1)
                    current_param = parts[0].strip()
                    current_desc = [parts[1].strip()] if len(parts) > 1 else []

                elif current_param and stripped:
                    current_desc.append(stripped)

        if current_param:
            param_docs[current_param] = " ".join(current_desc).strip()

        return param_docs


# Distance functions catalog
DISTANCE_FUNCTIONS = {
    "beginner": [
        {
            "name": "euclidean",
            "display_name": "Euclidean",
            "description": "Standard Euclidean distance (L2 norm)",
            "params": {},
        },
        {
            "name": "dtw",
            "display_name": "Dynamic Time Warping (DTW)",
            "description": "Elastic distance that aligns series optimally",
            "params": {
                "window": {"type": "float", "default": None, "description": "Warping window as fraction of series length"},
            },
        },
    ],
    "intermediate": [
        {
            "name": "squared",
            "display_name": "Squared Euclidean",
            "description": "Squared Euclidean distance",
            "params": {},
        },
        {
            "name": "manhattan",
            "display_name": "Manhattan",
            "description": "Manhattan distance (L1 norm)",
            "params": {},
        },
        {
            "name": "erp",
            "display_name": "Edit Distance with Real Penalty (ERP)",
            "description": "Edit distance variant for time series",
            "params": {
                "g": {"type": "float", "default": 0.0, "description": "Gap penalty"},
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
        {
            "name": "lcss",
            "display_name": "Longest Common Subsequence (LCSS)",
            "description": "Distance based on longest common subsequence",
            "params": {
                "epsilon": {"type": "float", "default": 1.0, "description": "Matching threshold"},
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
    ],
    "advanced": [
        {
            "name": "ddtw",
            "display_name": "Derivative DTW",
            "description": "DTW on derivative of series",
            "params": {
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
        {
            "name": "wdtw",
            "display_name": "Weighted DTW",
            "description": "DTW with distance-dependent weights",
            "params": {
                "g": {"type": "float", "default": 0.05, "description": "Weight parameter"},
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
        {
            "name": "twe",
            "display_name": "Time Warp Edit (TWE)",
            "description": "Combines DTW and edit distance concepts",
            "params": {
                "nu": {"type": "float", "default": 0.001, "description": "Stiffness parameter"},
                "lmbda": {"type": "float", "default": 1.0, "description": "Penalty for delete operation"},
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
        {
            "name": "msm",
            "display_name": "Move-Split-Merge (MSM)",
            "description": "Edit-based distance using move/split/merge operations",
            "params": {
                "c": {"type": "float", "default": 1.0, "description": "Cost parameter"},
                "window": {"type": "float", "default": None, "description": "Warping window"},
            },
        },
    ],
}


def get_distance_functions(level: str = "advanced") -> List[Dict]:
    """Get available distance functions for the given level."""
    result = DISTANCE_FUNCTIONS["beginner"].copy()

    if level in ("intermediate", "advanced"):
        result.extend(DISTANCE_FUNCTIONS["intermediate"])

    if level == "advanced":
        result.extend(DISTANCE_FUNCTIONS["advanced"])

    return result


# Singleton instance
_aeon_discovery: Optional[AeonDiscovery] = None


def get_aeon_discovery() -> AeonDiscovery:
    """Get singleton AeonDiscovery instance."""
    global _aeon_discovery
    if _aeon_discovery is None:
        _aeon_discovery = AeonDiscovery()
    return _aeon_discovery
