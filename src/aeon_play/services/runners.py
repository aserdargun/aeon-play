"""
Task runners for executing ML operations.

Provides wrappers for training, prediction, and evaluation of aeon estimators.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import traceback

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score,
)

from .aeon_discovery import get_aeon_discovery
from .converters import DataConverter


@dataclass
class RunResult:
    """Result of a task run."""
    success: bool
    task_type: str
    estimator_name: str
    train_time: float
    predict_time: Optional[float] = None
    metrics: Dict[str, float] = None
    predictions: Any = None
    probabilities: Any = None
    confusion_matrix: Any = None
    feature_importances: Any = None
    cluster_labels: Any = None
    anomaly_scores: Any = None
    forecast_values: Any = None
    change_points: Any = None
    error_message: Optional[str] = None
    logs: List[str] = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for key in ["predictions", "probabilities", "confusion_matrix",
                    "feature_importances", "cluster_labels", "anomaly_scores",
                    "forecast_values", "change_points"]:
            if result[key] is not None and isinstance(result[key], np.ndarray):
                result[key] = result[key].tolist()
        return result


class TaskRunner:
    """Executes ML tasks using aeon estimators."""

    def __init__(self):
        """Initialize the task runner."""
        self.discovery = get_aeon_discovery()
        self.converter = DataConverter()

    def run_classification(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
        cross_validate: bool = False,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> RunResult:
        """
        Run a classification task.

        Args:
            X: Feature data (n_cases, n_channels, n_timepoints)
            y: Labels
            estimator_name: Name of the classifier
            params: Estimator parameters
            test_size: Fraction for test split
            cross_validate: Whether to run cross-validation
            cv_folds: Number of CV folds
            random_state: Random seed

        Returns:
            RunResult with metrics and predictions
        """
        logs = []

        try:
            # Get estimator class
            est_class = self.discovery.get_estimator_class(estimator_name, "classifier")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="classification",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Classifier '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using classifier: {estimator_name}")

            # Clean params
            clean_params = {k: v for k, v in params.items() if v is not None}
            if "random_state" not in clean_params:
                clean_params["random_state"] = random_state

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logs.append(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

            # Create and train estimator
            estimator = est_class(**clean_params)
            logs.append("Training model...")

            start_time = time.time()
            estimator.fit(X_train, y_train)
            train_time = time.time() - start_time
            logs.append(f"Training completed in {train_time:.2f}s")

            # Predict
            start_time = time.time()
            predictions = estimator.predict(X_test)
            predict_time = time.time() - start_time

            # Get probabilities if available
            probabilities = None
            if hasattr(estimator, "predict_proba"):
                try:
                    probabilities = estimator.predict_proba(X_test)
                except Exception:
                    pass

            # Calculate metrics
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            cm = confusion_matrix(y_test, predictions)

            metrics = {
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
            }

            logs.append(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Cross-validation if requested
            if cross_validate:
                logs.append(f"Running {cv_folds}-fold cross-validation...")
                cv_scores = cross_val_score(
                    est_class(**clean_params), X, y, cv=cv_folds, scoring='accuracy'
                )
                metrics["cv_mean"] = round(cv_scores.mean(), 4)
                metrics["cv_std"] = round(cv_scores.std(), 4)
                logs.append(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

            return RunResult(
                success=True,
                task_type="classification",
                estimator_name=estimator_name,
                train_time=train_time,
                predict_time=predict_time,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                confusion_matrix=cm,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="classification",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> RunResult:
        """Run a regression task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "regressor")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="regression",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Regressor '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using regressor: {estimator_name}")

            clean_params = {k: v for k, v in params.items() if v is not None}
            if "random_state" not in clean_params:
                clean_params["random_state"] = random_state

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logs.append(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

            estimator = est_class(**clean_params)
            logs.append("Training model...")

            start_time = time.time()
            estimator.fit(X_train, y_train)
            train_time = time.time() - start_time
            logs.append(f"Training completed in {train_time:.2f}s")

            start_time = time.time()
            predictions = estimator.predict(X_test)
            predict_time = time.time() - start_time

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            metrics = {
                "mae": round(mae, 4),
                "mse": round(mse, 4),
                "rmse": round(np.sqrt(mse), 4),
                "r2": round(r2, 4),
            }

            logs.append(f"MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}, R2: {r2:.4f}")

            return RunResult(
                success=True,
                task_type="regression",
                estimator_name=estimator_name,
                train_time=train_time,
                predict_time=predict_time,
                metrics=metrics,
                predictions=predictions,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="regression",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_clustering(
        self,
        X: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        y_true: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> RunResult:
        """Run a clustering task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "clusterer")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="clustering",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Clusterer '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using clusterer: {estimator_name}")

            clean_params = {k: v for k, v in params.items() if v is not None}
            if "random_state" not in clean_params:
                clean_params["random_state"] = random_state

            estimator = est_class(**clean_params)
            logs.append("Fitting clusters...")

            start_time = time.time()
            cluster_labels = estimator.fit_predict(X)
            train_time = time.time() - start_time

            n_clusters = len(np.unique(cluster_labels))
            logs.append(f"Found {n_clusters} clusters in {train_time:.2f}s")

            metrics = {"n_clusters": n_clusters}

            # Calculate silhouette if we have enough samples
            if len(X) > n_clusters and n_clusters > 1:
                try:
                    # Flatten X for silhouette calculation
                    X_flat = X.reshape(X.shape[0], -1)
                    sil = silhouette_score(X_flat, cluster_labels)
                    metrics["silhouette"] = round(sil, 4)
                    logs.append(f"Silhouette score: {sil:.4f}")
                except Exception:
                    pass

            # Calculate ARI if true labels provided
            if y_true is not None:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(y_true, cluster_labels)
                metrics["ari"] = round(ari, 4)
                logs.append(f"Adjusted Rand Index: {ari:.4f}")

            return RunResult(
                success=True,
                task_type="clustering",
                estimator_name=estimator_name,
                train_time=train_time,
                metrics=metrics,
                cluster_labels=cluster_labels,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="clustering",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_forecasting(
        self,
        series: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        horizon: int = 10,
        test_size: Optional[int] = None,
        random_state: int = 42,
    ) -> RunResult:
        """Run a forecasting task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "forecaster")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="forecasting",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Forecaster '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using forecaster: {estimator_name}")

            # Ensure series is 1D
            series = self.converter.to_single_series(series)

            # Split for backtesting
            if test_size is None:
                test_size = min(horizon, len(series) // 5)

            train_series = series[:-test_size]
            test_series = series[-test_size:]

            logs.append(f"Train length: {len(train_series)}, Test length: {len(test_series)}")

            clean_params = {k: v for k, v in params.items() if v is not None}

            estimator = est_class(**clean_params)
            logs.append("Fitting forecaster...")

            start_time = time.time()
            estimator.fit(train_series)
            train_time = time.time() - start_time
            logs.append(f"Training completed in {train_time:.2f}s")

            # Forecast
            start_time = time.time()
            fh = np.arange(1, test_size + 1)
            forecast = estimator.predict(fh=fh)
            predict_time = time.time() - start_time

            # Calculate metrics
            mae = mean_absolute_error(test_series, forecast)
            mse = mean_squared_error(test_series, forecast)
            mape = np.mean(np.abs((test_series - forecast) / (test_series + 1e-10))) * 100

            metrics = {
                "mae": round(mae, 4),
                "rmse": round(np.sqrt(mse), 4),
                "mape": round(mape, 2),
            }

            logs.append(f"MAE: {mae:.4f}, RMSE: {np.sqrt(mse):.4f}, MAPE: {mape:.2f}%")

            return RunResult(
                success=True,
                task_type="forecasting",
                estimator_name=estimator_name,
                train_time=train_time,
                predict_time=predict_time,
                metrics=metrics,
                forecast_values=forecast,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="forecasting",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_anomaly_detection(
        self,
        series: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        y_true: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> RunResult:
        """Run an anomaly detection task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "anomaly_detector")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="anomaly_detection",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Anomaly detector '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using anomaly detector: {estimator_name}")

            # Ensure proper format
            if series.ndim == 1:
                series = series.reshape(1, 1, -1)

            clean_params = {k: v for k, v in params.items() if v is not None}

            estimator = est_class(**clean_params)
            logs.append("Detecting anomalies...")

            start_time = time.time()
            estimator.fit(series)
            anomaly_scores = estimator.predict(series)
            train_time = time.time() - start_time

            logs.append(f"Detection completed in {train_time:.2f}s")

            metrics = {
                "n_anomalies": int(np.sum(anomaly_scores > 0)) if anomaly_scores is not None else 0,
            }

            # Calculate metrics if true labels provided
            if y_true is not None:
                from sklearn.metrics import precision_score, recall_score
                binary_preds = (anomaly_scores > 0).astype(int)
                precision = precision_score(y_true, binary_preds, zero_division=0)
                recall = recall_score(y_true, binary_preds, zero_division=0)
                metrics["precision"] = round(precision, 4)
                metrics["recall"] = round(recall, 4)

            return RunResult(
                success=True,
                task_type="anomaly_detection",
                estimator_name=estimator_name,
                train_time=train_time,
                metrics=metrics,
                anomaly_scores=anomaly_scores,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="anomaly_detection",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_segmentation(
        self,
        series: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        random_state: int = 42,
    ) -> RunResult:
        """Run a segmentation task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "segmenter")
            if est_class is None:
                return RunResult(
                    success=False,
                    task_type="segmentation",
                    estimator_name=estimator_name,
                    train_time=0,
                    error_message=f"Segmenter '{estimator_name}' not found",
                    logs=logs,
                )

            logs.append(f"Using segmenter: {estimator_name}")

            # Ensure series is 1D
            series = self.converter.to_single_series(series)

            clean_params = {k: v for k, v in params.items() if v is not None}

            estimator = est_class(**clean_params)
            logs.append("Finding change points...")

            start_time = time.time()
            estimator.fit(series)
            change_points = estimator.predict(series)
            train_time = time.time() - start_time

            n_segments = len(change_points) + 1 if change_points is not None else 1
            logs.append(f"Found {n_segments} segments in {train_time:.2f}s")

            metrics = {
                "n_change_points": len(change_points) if change_points is not None else 0,
                "n_segments": n_segments,
            }

            return RunResult(
                success=True,
                task_type="segmentation",
                estimator_name=estimator_name,
                train_time=train_time,
                metrics=metrics,
                change_points=change_points,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="segmentation",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def run_similarity_search(
        self,
        X: np.ndarray,
        query: np.ndarray,
        estimator_name: str,
        params: Dict[str, Any],
        k: int = 5,
    ) -> RunResult:
        """Run a similarity search task."""
        logs = []

        try:
            est_class = self.discovery.get_estimator_class(estimator_name, "similarity_search")
            if est_class is None:
                # Fallback to basic kNN-style search
                logs.append("Using fallback nearest neighbor search")
                return self._fallback_similarity_search(X, query, k, logs)

            logs.append(f"Using similarity searcher: {estimator_name}")

            clean_params = {k: v for k, v in params.items() if v is not None}
            clean_params["k"] = k

            estimator = est_class(**clean_params)
            logs.append("Building index...")

            start_time = time.time()
            estimator.fit(X)
            train_time = time.time() - start_time

            logs.append("Searching for similar series...")
            indices = estimator.predict(query)

            metrics = {
                "k": k,
                "n_database": len(X),
            }

            return RunResult(
                success=True,
                task_type="similarity_search",
                estimator_name=estimator_name,
                train_time=train_time,
                metrics=metrics,
                predictions=indices,
                logs=logs,
            )

        except Exception as e:
            logs.append(f"Error: {str(e)}")
            return RunResult(
                success=False,
                task_type="similarity_search",
                estimator_name=estimator_name,
                train_time=0,
                error_message=str(e),
                logs=logs,
            )

    def _fallback_similarity_search(
        self,
        X: np.ndarray,
        query: np.ndarray,
        k: int,
        logs: List[str],
    ) -> RunResult:
        """Fallback similarity search using Euclidean distance."""
        start_time = time.time()

        # Flatten for distance calculation
        X_flat = X.reshape(X.shape[0], -1)
        query_flat = query.flatten()

        # Calculate distances
        distances = np.linalg.norm(X_flat - query_flat, axis=1)
        indices = np.argsort(distances)[:k]

        train_time = time.time() - start_time

        logs.append(f"Found {k} nearest neighbors")

        return RunResult(
            success=True,
            task_type="similarity_search",
            estimator_name="EuclideanNN",
            train_time=train_time,
            metrics={"k": k, "distances": distances[indices].tolist()},
            predictions=indices,
            logs=logs,
        )

    def run_transformation(
        self,
        X: np.ndarray,
        transformer_name: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, Any, str]:
        """
        Run a transformation on data.

        Returns:
            Tuple of (success, transformed_data, error_message)
        """
        try:
            est_class = self.discovery.get_estimator_class(transformer_name, "transformer")
            if est_class is None:
                return False, None, f"Transformer '{transformer_name}' not found"

            clean_params = {k: v for k, v in params.items() if v is not None}

            transformer = est_class(**clean_params)
            transformed = transformer.fit_transform(X)

            return True, transformed, ""

        except Exception as e:
            return False, None, str(e)
