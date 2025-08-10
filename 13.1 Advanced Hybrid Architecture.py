import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

class HybridForecastingFramework:
    """
    Advanced hybrid forecasting framework combining statistical and 
    machine learning approaches for enhanced prediction accuracy.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Component models
        self.statistical_models = {}
        self.ml_models = {}
        self.meta_learner = None
        
        # Fitted states
        self.is_fitted = False
        self.training_history = []
        self.component_weights = {}
        self.performance_metrics = {}
        
        # Decomposition components
        self.decomposition_method = self.config.get('decomposition_method', 'adaptive')
        self.decomposed_components = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for hybrid framework."""
        return {
            'statistical_models': ['arima', 'exp_smoothing', 'prophet'],
            'ml_models': ['lstm', 'random_forest', 'xgboost'],
            'meta_learning_method': 'dynamic_weighted_average',
            'decomposition_method': 'adaptive',
            'ensemble_strategy': 'hierarchical',
            'validation_method': 'time_series_cv',
            'optimization_objective': 'balanced_accuracy_diversity'
        }
    
    def add_statistical_model(self, name: str, model: BaseEstimator, 
                            config: Dict[str, Any] = None) -> 'HybridForecastingFramework':
        """Add statistical forecasting model to the ensemble."""
        
        model_wrapper = StatisticalModelWrapper(model, config or {})
        self.statistical_models[name] = model_wrapper
        
        return self
    
    def add_ml_model(self, name: str, model: BaseEstimator, 
                    config: Dict[str, Any] = None) -> 'HybridForecastingFramework':
        """Add machine learning model to the ensemble."""
        
        model_wrapper = MLModelWrapper(model, config or {})
        self.ml_models[name] = model_wrapper
        
        return self
    
    def fit(self, data: pd.DataFrame, target_column: str,
            validation_data: pd.DataFrame = None) -> 'HybridForecastingFramework':
        """
        Fit hybrid forecasting framework using hierarchical ensemble strategy.
        """
        
        self._validate_input_data(data, target_column)
        
        # Step 1: Data preprocessing and decomposition
        processed_data = self._preprocess_data(data, target_column)
        
        # Step 2: Decompose time series based on chosen method
        if self.decomposition_method != 'none':
            self.decomposed_components = self._decompose_time_series(
                processed_data[target_column]
            )
        
        # Step 3: Fit statistical models
        self._fit_statistical_models(processed_data, target_column)
        
        # Step 4: Fit machine learning models
        self._fit_ml_models(processed_data, target_column)
        
        # Step 5: Train meta-learner for ensemble combination
        self._train_meta_learner(processed_data, target_column, validation_data)
        
        # Step 6: Calculate component weights and performance metrics
        self._calculate_component_performance(processed_data, target_column)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame, horizon: int = 1, 
               return_components: bool = False, 
               return_uncertainty: bool = False) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Generate hybrid forecasts combining all component models.
        """
        
        if not self.is_fitted:
            raise RuntimeError("Framework must be fitted before prediction")
        
        # Get predictions from all component models
        statistical_predictions = self._get_statistical_predictions(data, horizon)
        ml_predictions = self._get_ml_predictions(data, horizon)
        
        # Combine predictions using meta-learner
        ensemble_prediction = self._combine_predictions(
            statistical_predictions, ml_predictions, horizon
        )
        
        # Generate uncertainty estimates if requested
        uncertainty_estimates = None
        if return_uncertainty:
            uncertainty_estimates = self._estimate_uncertainty(
                statistical_predictions, ml_predictions, ensemble_prediction, horizon
            )
        
        # Prepare return format
        if return_components or return_uncertainty:
            result = {
                'prediction': ensemble_prediction,
                'statistical_components': statistical_predictions if return_components else None,
                'ml_components': ml_predictions if return_components else None,
                'uncertainty': uncertainty_estimates if return_uncertainty else None
            }
            return result
        else:
            return ensemble_prediction
    
    def _preprocess_data(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Advanced preprocessing for hybrid modeling."""
        
        processed_data = data.copy()
        
        # Handle missing values
        if processed_data[target_column].isnull().any():
            processed_data[target_column] = processed_data[target_column].interpolate(
                method='time' if hasattr(processed_data.index, 'freq') else 'linear'
            )
        
        # Create additional features for ML models
        processed_data = self._create_temporal_features(processed_data)
        
        # Apply scaling if needed
        if self.config.get('scale_features', True):
            processed_data = self._scale_features(processed_data, target_column)
        
        return processed_data
    
    def _decompose_time_series(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Adaptive time series decomposition."""
        
        if self.decomposition_method == 'adaptive':
            return self._adaptive_decomposition(series)
        elif self.decomposition_method == 'stl':
            return self._stl_decomposition(series)
        elif self.decomposition_method == 'x13':
            return self._x13_decomposition(series)
        else:
            return {'original': series}
    
    def _adaptive_decomposition(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Adaptive decomposition choosing best method based on data characteristics."""
        
        # Analyze series characteristics
        seasonality_strength = self._measure_seasonality_strength(series)
        trend_strength = self._measure_trend_strength(series)
        noise_level = self._measure_noise_level(series)
        
        # Choose decomposition method based on characteristics
        if seasonality_strength > 0.6 and len(series) > 104:  # Strong seasonality, enough data
            return self._stl_decomposition(series)
        elif trend_strength > 0.5:
            return self._hp_filter_decomposition(series)
        else:
            return self._simple_ma_decomposition(series)
    
    def _fit_statistical_models(self, data: pd.DataFrame, target_column: str) -> None:
        """Fit all statistical models with optimal configurations."""
        
        for name, model_wrapper in self.statistical_models.items():
            try:
                # Prepare data specific to statistical model requirements
                model_data = self._prepare_statistical_data(data, target_column, name)
                
                # Fit model
                model_wrapper.fit(model_data)
                
                # Store fitting information
                self.training_history.append({
                    'model_name': name,
                    'model_type': 'statistical',
                    'status': 'success',
                    'data_shape': model_data.shape
                })
                
            except Exception as e:
                warnings.warn(f"Failed to fit statistical model {name}: {str(e)}")
                self.training_history.append({
                    'model_name': name,
                    'model_type': 'statistical',
                    'status': 'failed',
                    'error': str(e)
                })
    
    def _fit_ml_models(self, data: pd.DataFrame, target_column: str) -> None:
        """Fit all machine learning models with feature engineering."""
        
        for name, model_wrapper in self.ml_models.items():
            try:
                # Prepare features for ML model
                ml_features = self._prepare_ml_features(data, target_column, name)
                
                # Fit model
                model_wrapper.fit(ml_features['X'], ml_features['y'])
                
                self.training_history.append({
                    'model_name': name,
                    'model_type': 'ml',
                    'status': 'success',
                    'feature_count': ml_features['X'].shape[1]
                })
                
            except Exception as e:
                warnings.warn(f"Failed to fit ML model {name}: {str(e)}")
                self.training_history.append({
                    'model_name': name,
                    'model_type': 'ml',
                    'status': 'failed',
                    'error': str(e)
                })
    
    def _train_meta_learner(self, data: pd.DataFrame, target_column: str,
                          validation_data: pd.DataFrame = None) -> None:
        """Train meta-learner for optimal ensemble combination."""
        
        method = self.config['meta_learning_method']
        
        if method == 'dynamic_weighted_average':
            self.meta_learner = DynamicWeightedAverageMetaLearner()
        elif method == 'stacked_regression':
            self.meta_learner = StackedRegressionMetaLearner()
        elif method == 'neural_ensemble':
            self.meta_learner = NeuralEnsembleMetaLearner()
        else:
            self.meta_learner = SimpleAverageMetaLearner()
        
        # Prepare meta-learning data
        meta_features = self._prepare_meta_learning_data(data, target_column, validation_data)
        
        # Train meta-learner
        if meta_features is not None:
            self.meta_learner.fit(meta_features['predictions'], meta_features['targets'])
    
    def _get_statistical_predictions(self, data: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        """Get predictions from all statistical models."""
        
        predictions = {}
        
        for name, model_wrapper in self.statistical_models.items():
            try:
                model_data = self._prepare_statistical_data(data, None, name)
                pred = model_wrapper.predict(model_data, horizon)
                predictions[name] = pred
            except Exception as e:
                warnings.warn(f"Statistical model {name} prediction failed: {str(e)}")
                predictions[name] = np.full(horizon, np.nan)
        
        return predictions
    
    def _get_ml_predictions(self, data: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        """Get predictions from all ML models."""
        
        predictions = {}
        
        for name, model_wrapper in self.ml_models.items():
            try:
                ml_features = self._prepare_ml_features(data, None, name, for_prediction=True)
                pred = model_wrapper.predict(ml_features['X'], horizon)
                predictions[name] = pred
            except Exception as e:
                warnings.warn(f"ML model {name} prediction failed: {str(e)}")
                predictions[name] = np.full(horizon, np.nan)
        
        return predictions
    
    def _combine_predictions(self, statistical_preds: Dict[str, np.ndarray],
                           ml_preds: Dict[str, np.ndarray], horizon: int) -> np.ndarray:
        """Combine predictions using trained meta-learner."""
        
        # Prepare prediction matrix
        all_predictions = []
        model_names = []
        
        # Add statistical predictions
        for name, pred in statistical_preds.items():
            if not np.isnan(pred).all():
                all_predictions.append(pred)
                model_names.append(f"stat_{name}")
        
        # Add ML predictions
        for name, pred in ml_preds.items():
            if not np.isnan(pred).all():
                all_predictions.append(pred)
                model_names.append(f"ml_{name}")
        
        if not all_predictions:
            return np.full(horizon, np.nan)
        
        prediction_matrix = np.array(all_predictions).T  # Shape: (horizon, n_models)
        
        # Use meta-learner to combine predictions
        if self.meta_learner and hasattr(self.meta_learner, 'predict'):
            return self.meta_learner.predict(prediction_matrix)
        else:
            # Fallback to weighted average
            return np.nanmean(prediction_matrix, axis=1)
    
    def _estimate_uncertainty(self, statistical_preds: Dict[str, np.ndarray],
                            ml_preds: Dict[str, np.ndarray], 
                            ensemble_pred: np.ndarray, horizon: int) -> Dict[str, np.ndarray]:
        """Estimate prediction uncertainty using ensemble variance."""
        
        all_predictions = []
        
        # Collect all valid predictions
        for pred in statistical_preds.values():
            if not np.isnan(pred).all():
                all_predictions.append(pred)
        
        for pred in ml_preds.values():
            if not np.isnan(pred).all():
                all_predictions.append(pred)
        
        if len(all_predictions) < 2:
            # Not enough models for uncertainty estimation
            return {
                'prediction_variance': np.full(horizon, np.nan),
                'prediction_std': np.full(horizon, np.nan),
                'confidence_intervals': {
                    '68': (np.full(horizon, np.nan), np.full(horizon, np.nan)),
                    '95': (np.full(horizon, np.nan), np.full(horizon, np.nan))
                }
            }
        
        prediction_matrix = np.array(all_predictions)
        
        # Calculate ensemble variance
        prediction_variance = np.var(prediction_matrix, axis=0)
        prediction_std = np.sqrt(prediction_variance)
        
        # Generate confidence intervals assuming normal distribution
        ci_68_lower = ensemble_pred - 1.0 * prediction_std
        ci_68_upper = ensemble_pred + 1.0 * prediction_std
        ci_95_lower = ensemble_pred - 1.96 * prediction_std
        ci_95_upper = ensemble_pred + 1.96 * prediction_std
        
        return {
            'prediction_variance': prediction_variance,
            'prediction_std': prediction_std,
            'confidence_intervals': {
                '68': (ci_68_lower, ci_68_upper),
                '95': (ci_95_lower, ci_95_upper)
            }
        }
    
    # Helper methods for data preparation and feature engineering
    def _validate_input_data(self, data: pd.DataFrame, target_column: str) -> None:
        """Validate input data format and content."""
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if data[target_column].isnull().all():
            raise ValueError("Target column contains only null values")
        
        if len(data) < 10:
            raise ValueError("Insufficient data points for modeling (minimum 10 required)")
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features for ML models."""
        
        result = data.copy()
        
        # If index is datetime, create time-based features
        if isinstance(result.index, pd.DatetimeIndex):
            result['hour'] = result.index.hour
            result['day_of_week'] = result.index.dayofweek
            result['day_of_month'] = result.index.day
            result['month'] = result.index.month
            result['quarter'] = result.index.quarter
            result['year'] = result.index.year
            
            # Cyclical encoding for periodic features
            result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
            result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
            result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
            result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
            result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        return result
    
    def _measure_seasonality_strength(self, series: pd.Series) -> float:
        """Measure strength of seasonal patterns in the series."""
        
        # Simple seasonality detection using autocorrelation
        if len(series) < 24:
            return 0.0
        
        # Test common seasonal periods
        seasonal_periods = [7, 12, 24, 52] if len(series) > 104 else [7, 12]
        max_autocorr = 0.0
        
        for period in seasonal_periods:
            if len(series) > 2 * period:
                autocorr = series.autocorr(lag=period)
                if not np.isnan(autocorr):
                    max_autocorr = max(max_autocorr, abs(autocorr))
        
        return max_autocorr
    
    def _measure_trend_strength(self, series: pd.Series) -> float:
        """Measure strength of trend in the series."""
        
        if len(series) < 10:
            return 0.0
        
        # Calculate linear trend using least squares
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            return 0.0
        
        # Calculate correlation coefficient as trend strength
        correlation = np.corrcoef(x_valid, y_valid)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _measure_noise_level(self, series: pd.Series) -> float:
        """Measure noise level in the series."""
        
        if len(series) < 5:
            return 1.0
        
        # Calculate coefficient of variation as noise measure
        mean_val = series.mean()
        std_val = series.std()
        
        if mean_val == 0:
            return 1.0
        
        return std_val / abs(mean_val)


class DynamicWeightedAverageMetaLearner:
    """
    Dynamic weighted average meta-learner that adapts weights based on 
    recent model performance.
    """
    
    def __init__(self, window_size: int = 50, decay_factor: float = 0.95):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.model_weights = None
        self.performance_history = []
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Fit meta-learner using prediction matrix and targets."""
        
        n_models = predictions.shape[1]
        
        # Initialize weights
        self.model_weights = np.ones(n_models) / n_models
        
        # Calculate performance for each model
        model_errors = []
        for i in range(n_models):
            model_pred = predictions[:, i]
            valid_mask = ~np.isnan(model_pred) & ~np.isnan(targets)
            
            if np.sum(valid_mask) > 0:
                error = mean_absolute_error(targets[valid_mask], model_pred[valid_mask])
                model_errors.append(error)
            else:
                model_errors.append(float('inf'))
        
        # Convert errors to weights (inverse relationship)
        model_errors = np.array(model_errors)
        model_errors[model_errors == float('inf')] = np.nanmax(model_errors[model_errors != float('inf')]) * 10
        
        # Calculate weights (higher weight for lower error)
        inverse_errors = 1.0 / (model_errors + 1e-8)
        self.model_weights = inverse_errors / np.sum(inverse_errors)
        
        # Store performance history
        self.performance_history.append({
            'errors': model_errors,
            'weights': self.model_weights.copy()
        })
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Generate weighted average predictions."""
        
        if self.model_weights is None:
            return np.nanmean(predictions, axis=1)
        
        # Apply weights to predictions
        weighted_pred = np.zeros(predictions.shape[0])
        
        for i in range(predictions.shape[0]):
            valid_mask = ~np.isnan(predictions[i, :])
            if np.any(valid_mask):
                valid_weights = self.model_weights[valid_mask]
                valid_weights = valid_weights / np.sum(valid_weights)  # Renormalize
                weighted_pred[i] = np.sum(predictions[i, valid_mask] * valid_weights)
            else:
                weighted_pred[i] = np.nan
        
        return weighted_pred
