import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.base import BaseEstimator
from scipy import stats
import warnings

class TimeSeriesConformalPredictor:
    """
    Advanced conformal prediction framework for time series forecasting
    with support for multi-step predictions and temporal dependencies.
    """
    
    def __init__(self, base_predictor: BaseEstimator, 
                 method: str = 'adaptive_copula',
                 coverage_level: float = 0.9,
                 window_size: int = 100,
                 update_frequency: int = 10):
        """
        Initialize conformal predictor.
        
        Args:
            base_predictor: Underlying forecasting model
            method: Conformal prediction method 
                   ('adaptive_copula', 'multi_dim_spci', 'autocorrelated_mcp')
            coverage_level: Target coverage probability
            window_size: Size of calibration window
            update_frequency: Frequency of quantile updates
        """
        
        self.base_predictor = base_predictor
        self.method = method
        self.coverage_level = coverage_level
        self.alpha = 1 - coverage_level
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        # Calibration data storage
        self.calibration_scores = []
        self.prediction_history = []
        self.true_values_history = []
        
        # Adaptive parameters
        self.current_quantile = None
        self.update_counter = 0
        
        # Method-specific components
        if method == 'adaptive_copula':
            self.copula_model = CopulaConformalModel()
        elif method == 'multi_dim_spci':
            self.ellipsoid_calculator = EllipsoidCalculator()
        elif method == 'autocorrelated_mcp':
            self.autocorr_analyzer = AutocorrelationAnalyzer()
        
        # Uncertainty decomposer
        self.uncertainty_decomposer = UncertaintyDecomposer()
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                 horizon: int = 1) -> None:
        """
        Calibrate conformal predictor using historical data.
        
        Args:
            X_cal: Calibration input features
            y_cal: Calibration target values  
            horizon: Forecasting horizon
        """
        
        # Generate predictions on calibration set
        predictions = self._generate_base_predictions(X_cal, horizon)
        
        # Calculate non-conformity scores
        if self.method == 'adaptive_copula':
            scores = self._calculate_copula_scores(predictions, y_cal, horizon)
        elif self.method == 'multi_dim_spci':
            scores = self._calculate_multidim_scores(predictions, y_cal, horizon)
        elif self.method == 'autocorrelated_mcp':
            scores = self._calculate_autocorr_scores(predictions, y_cal, horizon)
        else:
            scores = self._calculate_standard_scores(predictions, y_cal)
        
        # Store calibration scores
        self.calibration_scores.extend(scores)
        
        # Limit calibration window size
        if len(self.calibration_scores) > self.window_size:
            self.calibration_scores = self.calibration_scores[-self.window_size:]
        
        # Calculate initial quantile
        self._update_quantile()
        
    def predict_interval(self, X_new: np.ndarray, horizon: int = 1,
                        return_decomposition: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Dict]:
        """
        Generate prediction intervals for new data.
        
        Args:
            X_new: New input features
            horizon: Forecasting horizon
            return_decomposition: Whether to return uncertainty decomposition
            
        Returns:
            Prediction intervals or comprehensive results dictionary
        """
        
        # Generate base predictions
        predictions = self._generate_base_predictions(X_new, horizon)
        
        # Calculate prediction intervals based on method
        if self.method == 'adaptive_copula':
            intervals = self._copula_prediction_intervals(predictions, horizon)
        elif self.method == 'multi_dim_spci':
            intervals = self._multidim_prediction_intervals(predictions, horizon)
        elif self.method == 'autocorrelated_mcp':
            intervals = self._autocorr_prediction_intervals(predictions, horizon)
        else:
            intervals = self._standard_prediction_intervals(predictions)
        
        lower_bounds, upper_bounds = intervals
        
        if return_decomposition:
            # Decompose uncertainty sources
            uncertainty_decomp = self.uncertainty_decomposer.decompose(
                predictions, lower_bounds, upper_bounds, X_new
            )
            
            return {
                'predictions': predictions,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'uncertainty_decomposition': uncertainty_decomp,
                'coverage_level': self.coverage_level,
                'method': self.method
            }
        
        return lower_bounds, upper_bounds
    
    def update_online(self, X_new: np.ndarray, y_new: np.ndarray,
                     horizon: int = 1) -> None:
        """
        Update conformal predictor online with new observations.
        
        Args:
            X_new: New input features
            y_new: New observed values
            horizon: Forecasting horizon
        """
        
        # Generate prediction for the new data
        prediction = self._generate_base_predictions(X_new, horizon)
        
        # Calculate non-conformity score
        if self.method == 'adaptive_copula':
            score = self._calculate_copula_scores(prediction, y_new, horizon)
        elif self.method == 'multi_dim_spci':
            score = self._calculate_multidim_scores(prediction, y_new, horizon)
        elif self.method == 'autocorrelated_mcp':
            score = self._calculate_autocorr_scores(prediction, y_new, horizon)
        else:
            score = self._calculate_standard_scores(prediction, y_new)
        
        # Update calibration scores
        self.calibration_scores.extend(score if isinstance(score, list) else [score])
        
        # Maintain window size
        if len(self.calibration_scores) > self.window_size:
            self.calibration_scores = self.calibration_scores[-self.window_size:]
        
        # Update history
        self.prediction_history.append(prediction)
        self.true_values_history.append(y_new)
        
        # Periodically update quantile
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            self._update_quantile()
    
    def _generate_base_predictions(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Generate predictions using base predictor."""
        
        if hasattr(self.base_predictor, 'predict_multi_step'):
            return self.base_predictor.predict_multi_step(X, horizon)
        else:
            # Fallback to single-step prediction
            return self.base_predictor.predict(X)
    
    def _calculate_copula_scores(self, predictions: np.ndarray, 
                               actuals: np.ndarray, horizon: int) -> List[float]:
        """Calculate non-conformity scores using copula approach."""
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if actuals.ndim == 1:
            actuals = actuals.reshape(-1, 1)
        
        scores = []
        
        for i in range(len(predictions)):
            pred_vec = predictions[i]
            actual_vec = actuals[i]
            
            # Multi-dimensional non-conformity using copula
            if len(pred_vec.shape) > 0 and pred_vec.shape[0] > 1:
                # Multivariate case
                score = self.copula_model.calculate_copula_score(pred_vec, actual_vec)
            else:
                # Univariate case
                score = abs(float(pred_vec) - float(actual_vec))
            
            scores.append(score)
        
        return scores
    
    def _calculate_multidim_scores(self, predictions: np.ndarray, 
                                 actuals: np.ndarray, horizon: int) -> List[float]:
        """Calculate scores for multi-dimensional conformal prediction."""
        
        scores = []
        
        for i in range(len(predictions)):
            pred_vec = predictions[i] if predictions.ndim > 1 else [predictions[i]]
            actual_vec = actuals[i] if actuals.ndim > 1 else [actuals[i]]
            
            # Mahalanobis-like distance for multi-dimensional residuals
            residual = np.array(actual_vec) - np.array(pred_vec)
            
            if len(residual) > 1:
                # Multi-dimensional score
                score = np.sqrt(np.sum(residual ** 2))
            else:
                # Univariate score
                score = abs(residual[0])
            
            scores.append(score)
        
        return scores
    
    def _calculate_autocorr_scores(self, predictions: np.ndarray, 
                                 actuals: np.ndarray, horizon: int) -> List[float]:
        """Calculate scores accounting for autocorrelation in residuals."""
        
        scores = []
        residuals = actuals.flatten() - predictions.flatten()
        
        # Account for temporal autocorrelation
        if len(self.prediction_history) > horizon:
            # Calculate autocorrelation-adjusted scores
            recent_residuals = []
            for i in range(min(horizon, len(self.true_values_history))):
                if len(self.true_values_history) > i and len(self.prediction_history) > i:
                    hist_residual = (
                        np.array(self.true_values_history[-(i+1)]).flatten() - 
                        np.array(self.prediction_history[-(i+1)]).flatten()
                    )
                    recent_residuals.extend(hist_residual)
            
            if recent_residuals:
                # Adjust current residuals based on historical autocorrelation
                autocorr_adjustment = self.autocorr_analyzer.calculate_adjustment(
                    residuals, recent_residuals, horizon
                )
                adjusted_residuals = residuals + autocorr_adjustment
                scores = [abs(r) for r in adjusted_residuals]
            else:
                scores = [abs(r) for r in residuals]
        else:
            scores = [abs(r) for r in residuals]
        
        return scores
    
    def _calculate_standard_scores(self, predictions: np.ndarray, 
                                 actuals: np.ndarray) -> List[float]:
        """Calculate standard absolute residual scores."""
        
        residuals = np.array(actuals).flatten() - np.array(predictions).flatten()
        return [abs(r) for r in residuals]
    
    def _update_quantile(self) -> None:
        """Update the quantile threshold for prediction intervals."""
        
        if not self.calibration_scores:
            return
        
        # Calculate empirical quantile
        n = len(self.calibration_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        self.current_quantile = np.quantile(
            self.calibration_scores, min(quantile_level, 1.0)
        )
    
    def _copula_prediction_intervals(self, predictions: np.ndarray, 
                                   horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals using copula method."""
        
        if self.current_quantile is None:
            self._update_quantile()
        
        interval_width = self.current_quantile
        
        # Apply copula-based adjustment for multi-step predictions
        if horizon > 1:
            copula_adjustment = self.copula_model.get_multi_step_adjustment(
                horizon, self.calibration_scores
            )
            interval_width *= copula_adjustment
        
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width
        
        return lower_bounds, upper_bounds
    
    def _multidim_prediction_intervals(self, predictions: np.ndarray, 
                                     horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multi-dimensional prediction intervals."""
        
        if self.current_quantile is None:
            self._update_quantile()
        
        # Create ellipsoidal prediction regions
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            ellipsoid_params = self.ellipsoid_calculator.calculate_ellipsoid(
                self.calibration_scores, self.coverage_level
            )
            
            lower_bounds, upper_bounds = self.ellipsoid_calculator.get_bounds(
                predictions, ellipsoid_params
            )
        else:
            # Fallback to standard intervals
            interval_width = self.current_quantile
            lower_bounds = predictions - interval_width
            upper_bounds = predictions + interval_width
        
        return lower_bounds, upper_bounds
    
    def _autocorr_prediction_intervals(self, predictions: np.ndarray, 
                                     horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate intervals accounting for autocorrelation."""
        
        if self.current_quantile is None:
            self._update_quantile()
        
        base_width = self.current_quantile
        
        # Adjust for autocorrelation effects
        if len(self.calibration_scores) > horizon * 2:
            autocorr_factor = self.autocorr_analyzer.get_horizon_adjustment(
                self.calibration_scores, horizon
            )
            adjusted_width = base_width * autocorr_factor
        else:
            adjusted_width = base_width
        
        lower_bounds = predictions - adjusted_width
        upper_bounds = predictions + adjusted_width
        
        return lower_bounds, upper_bounds
    
    def _standard_prediction_intervals(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate standard prediction intervals."""
        
        if self.current_quantile is None:
            self._update_quantile()
        
        interval_width = self.current_quantile
        
        lower_bounds = predictions - interval_width
        upper_bounds = predictions + interval_width
        
        return lower_bounds, upper_bounds


class CopulaConformalModel:
    """Copula-based conformal prediction model for multi-step forecasting."""
    
    def __init__(self):
        self.copula_params = {}
        
    def calculate_copula_score(self, prediction: np.ndarray, 
                             actual: np.ndarray) -> float:
        """Calculate copula-based non-conformity score."""
        
        if len(prediction) == 1:
            return abs(float(prediction[0]) - float(actual[0]))
        
        # Multi-dimensional copula score
        residuals = np.array(actual) - np.array(prediction)
        
        # Use empirical copula approach
        # Transform residuals to uniform margins
        uniform_residuals = self._transform_to_uniform(residuals)
        
        # Calculate copula-based distance
        copula_distance = np.sqrt(np.sum(uniform_residuals ** 2))
        
        return copula_distance
    
    def get_multi_step_adjustment(self, horizon: int, 
                                calibration_scores: List[float]) -> float:
        """Get adjustment factor for multi-step predictions."""
        
        if horizon == 1:
            return 1.0
        
        # Simple horizon-based scaling (can be improved with actual copula modeling)
        base_adjustment = np.sqrt(horizon)
        
        # Empirical adjustment based on historical performance
        if len(calibration_scores) > horizon * 10:
            recent_scores = calibration_scores[-horizon*10:]
            score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            trend_adjustment = 1 + abs(score_trend) * horizon
            return base_adjustment * trend_adjustment
        
        return base_adjustment
    
    def _transform_to_uniform(self, residuals: np.ndarray) -> np.ndarray:
        """Transform residuals to uniform margins using empirical CDF."""
        
        uniform_residuals = np.zeros_like(residuals)
        
        for i in range(len(residuals)):
            # Empirical CDF transformation
            uniform_residuals[i] = stats.norm.cdf(residuals[i])
        
        return uniform_residuals


class EllipsoidCalculator:
    """Calculator for ellipsoidal prediction regions."""
    
    def calculate_ellipsoid(self, scores: List[float], 
                          coverage_level: float) -> Dict:
        """Calculate ellipsoid parameters for prediction regions."""
        
        # Simple implementation - can be enhanced with proper multivariate analysis
        quantile = np.quantile(scores, coverage_level)
        
        return {
            'radius': quantile,
            'center_adjustment': 0.0,
            'shape_matrix': np.eye(1)  # Identity for simplicity
        }
    
    def get_bounds(self, predictions: np.ndarray, 
                  ellipsoid_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds from ellipsoid parameters."""
        
        radius = ellipsoid_params['radius']
        
        lower_bounds = predictions - radius
        upper_bounds = predictions + radius
        
        return lower_bounds, upper_bounds


class AutocorrelationAnalyzer:
    """Analyzer for autocorrelation effects in residuals."""
    
    def calculate_adjustment(self, current_residuals: np.ndarray,
                           historical_residuals: List[float], 
                           horizon: int) -> np.ndarray:
        """Calculate autocorrelation-based adjustment."""
        
        if not historical_residuals or len(current_residuals) == 0:
            return np.zeros_like(current_residuals)
        
        # Calculate autocorrelation coefficient
        if len(historical_residuals) > horizon:
            autocorr = np.corrcoef(
                historical_residuals[:-horizon], 
                historical_residuals[horizon:]
            )[0, 1]
            
            if not np.isnan(autocorr):
                # Apply autocorrelation adjustment
                adjustment = autocorr * np.mean(historical_residuals[-horizon:])
                return np.full_like(current_residuals, adjustment)
        
        return np.zeros_like(current_residuals)
    
    def get_horizon_adjustment(self, scores: List[float], horizon: int) -> float:
        """Get adjustment factor for different forecasting horizons."""
        
        if len(scores) < horizon * 2:
            return 1.0
        
        # Calculate empirical variance scaling with horizon
        short_term_var = np.var(scores[-horizon:])
        long_term_var = np.var(scores[-horizon*2:-horizon])
        
        if long_term_var > 0:
            variance_ratio = short_term_var / long_term_var
            return np.sqrt(max(variance_ratio, 0.5))  # Avoid extreme adjustments
        
        return 1.0


class UncertaintyDecomposer:
    """Decomposer for different sources of uncertainty."""
    
    def decompose(self, predictions: np.ndarray, lower_bounds: np.ndarray,
                 upper_bounds: np.ndarray, features: np.ndarray) -> Dict:
        """
        Decompose uncertainty into different sources.
        
        Returns:
            Dictionary with uncertainty decomposition
        """
        
        total_uncertainty = upper_bounds - lower_bounds
        
        # Estimate different uncertainty components
        aleatoric_uncertainty = self._estimate_aleatoric(predictions, total_uncertainty)
        epistemic_uncertainty = self._estimate_epistemic(features, total_uncertainty)
        distributional_uncertainty = total_uncertainty - aleatoric_uncertainty - epistemic_uncertainty
        
        return {
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'distributional_uncertainty': np.maximum(distributional_uncertainty, 0),
            'uncertainty_ratio': {
                'aleatoric': np.mean(aleatoric_uncertainty / total_uncertainty),
                'epistemic': np.mean(epistemic_uncertainty / total_uncertainty),
                'distributional': np.mean(np.maximum(distributional_uncertainty, 0) / total_uncertainty)
            }
        }
    
    def _estimate_aleatoric(self, predictions: np.ndarray, 
                          total_uncertainty: np.ndarray) -> np.ndarray:
        """Estimate aleatoric (data) uncertainty."""
        
        # Simple heuristic: assume aleatoric uncertainty is proportional to prediction magnitude
        base_aleatoric = 0.1 * np.abs(predictions)  # 10% of prediction magnitude
        
        # Cap at 50% of total uncertainty
        return np.minimum(base_aleatoric, 0.5 * total_uncertainty)
    
    def _estimate_epistemic(self, features: np.ndarray, 
                          total_uncertainty: np.ndarray) -> np.ndarray:
        """Estimate epistemic (model) uncertainty."""
        
        # Simple heuristic: assume epistemic uncertainty decreases with more "typical" inputs
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Calculate distance from feature means (proxy for typicality)
        feature_means = np.mean(features, axis=0)
        distances = np.linalg.norm(features - feature_means, axis=1)
        
        # Normalize distances to [0, 1]
        if np.max(distances) > 0:
            normalized_distances = distances / np.max(distances)
        else:
            normalized_distances = np.zeros_like(distances)
        
        # Epistemic uncertainty proportional to distance from typical inputs
        base_epistemic = 0.2 * normalized_distances * np.mean(total_uncertainty)
        
        # Cap at 40% of total uncertainty
        return np.minimum(base_epistemic, 0.4 * total_uncertainty)
