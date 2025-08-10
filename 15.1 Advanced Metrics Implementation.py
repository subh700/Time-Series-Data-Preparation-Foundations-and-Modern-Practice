import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from abc import ABC, abstractmethod

class TimeSeriesEvaluationFramework:
    """
    Comprehensive framework for time series forecasting evaluation
    supporting multiple metrics, statistical testing, and business impact assessment.
    """
    
    def __init__(self, metrics_config: Optional[Dict[str, Dict]] = None):
        """
        Initialize evaluation framework with configurable metrics.
        
        Args:
            metrics_config: Configuration for metrics and their parameters
        """
        
        self.metrics_config = metrics_config or self._default_metrics_config()
        self.metric_calculators = self._initialize_metrics()
        self.evaluation_history = []
        self.statistical_tests = StatisticalTestSuite()
        self.business_evaluator = BusinessImpactEvaluator()
        
    def _default_metrics_config(self) -> Dict[str, Dict]:
        """Default configuration for evaluation metrics."""
        return {
            'scale_dependent': {
                'MAE': {'robust_outliers': True, 'weight_recent': False},
                'MSE': {'robust_outliers': False, 'weight_recent': False},
                'RMSE': {'robust_outliers': False, 'weight_recent': False},
                'WAPE': {'aggregate_first': True}
            },
            'scale_independent': {
                'MAPE': {'handle_zeros': 'epsilon', 'epsilon': 1e-10},
                'sMAPE': {'version': 'symmetric'},
                'MASE': {'seasonal_period': 1, 'seasonal_test': True},
                'RMSSE': {'seasonal_period': 1}
            },
            'probabilistic': {
                'QuantileLoss': {'quantiles': [0.1, 0.5, 0.9]},
                'CRPS': {'method': 'exact'},
                'LogScore': {'handle_negatives': True}
            },
            'distribution_based': {
                'KL_Divergence': {'bins': 50},
                'Wasserstein': {'order': 1},
                'Energy_Distance': {}
            }
        }
    
    def _initialize_metrics(self) -> Dict[str, Callable]:
        """Initialize metric calculator functions."""
        return {
            # Scale-dependent metrics
            'MAE': self._calculate_mae,
            'MSE': self._calculate_mse,
            'RMSE': self._calculate_rmse,
            'WAPE': self._calculate_wape,
            
            # Scale-independent metrics
            'MAPE': self._calculate_mape,
            'sMAPE': self._calculate_smape,
            'MASE': self._calculate_mase,
            'RMSSE': self._calculate_rmsse,
            
            # Probabilistic metrics
            'QuantileLoss': self._calculate_quantile_loss,
            'CRPS': self._calculate_crps,
            'LogScore': self._calculate_log_score,
            
            # Distribution-based metrics
            'KL_Divergence': self._calculate_kl_divergence,
            'Wasserstein': self._calculate_wasserstein,
            'Energy_Distance': self._calculate_energy_distance
        }
    
    def evaluate_forecasts(self, actual: np.ndarray, predicted: np.ndarray,
                          historical_data: Optional[np.ndarray] = None,
                          prediction_intervals: Optional[Dict[str, np.ndarray]] = None,
                          business_context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of forecasts using multiple metrics.
        
        Args:
            actual: Actual observed values
            predicted: Point predictions
            historical_data: Historical data for scaled metrics
            prediction_intervals: Probabilistic prediction intervals
            business_context: Business context for impact evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        results = {}
        
        # Validate inputs
        actual, predicted = self._validate_inputs(actual, predicted)
        
        # Calculate scale-dependent metrics
        scale_dependent = self._calculate_scale_dependent_metrics(actual, predicted)
        results.update(scale_dependent)
        
        # Calculate scale-independent metrics
        if historical_data is not None:
            scale_independent = self._calculate_scale_independent_metrics(
                actual, predicted, historical_data
            )
            results.update(scale_independent)
        
        # Calculate probabilistic metrics
        if prediction_intervals is not None:
            probabilistic = self._calculate_probabilistic_metrics(
                actual, predicted, prediction_intervals
            )
            results.update(probabilistic)
        
        # Calculate business impact metrics
        if business_context is not None:
            business_metrics = self.business_evaluator.evaluate_business_impact(
                actual, predicted, business_context
            )
            results.update(business_metrics)
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': pd.Timestamp.now(),
            'metrics': results.copy(),
            'sample_size': len(actual)
        })
        
        return results
    
    def _calculate_scale_dependent_metrics(self, actual: np.ndarray, 
                                         predicted: np.ndarray) -> Dict[str, float]:
        """Calculate scale-dependent metrics."""
        
        metrics = {}
        
        for metric_name in self.metrics_config['scale_dependent']:
            try:
                value = self.metric_calculators[metric_name](actual, predicted)
                metrics[metric_name] = value
            except Exception as e:
                warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
                metrics[metric_name] = np.nan
        
        return metrics
    
    def _calculate_scale_independent_metrics(self, actual: np.ndarray, 
                                           predicted: np.ndarray,
                                           historical_data: np.ndarray) -> Dict[str, float]:
        """Calculate scale-independent metrics."""
        
        metrics = {}
        
        for metric_name in self.metrics_config['scale_independent']:
            try:
                if metric_name in ['MASE', 'RMSSE']:
                    value = self.metric_calculators[metric_name](
                        actual, predicted, historical_data
                    )
                else:
                    value = self.metric_calculators[metric_name](actual, predicted)
                metrics[metric_name] = value
            except Exception as e:
                warnings.warn(f"Failed to calculate {metric_name}: {str(e)}")
                metrics[metric_name] = np.nan
        
        return metrics
    
    def _calculate_probabilistic_metrics(self, actual: np.ndarray, 
                                       predicted: np.ndarray,
                                       prediction_intervals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate probabilistic forecasting metrics."""
        
        metrics = {}
        
        # Interval coverage
        for interval_name, bounds in prediction_intervals.items():
            if bounds.shape[1] == 2:  # Lower and upper bounds
                coverage = self._calculate_interval_coverage(
                    actual, bounds[:, 0], bounds[:, 1]
                )
                metrics[f'Coverage_{interval_name}'] = coverage
                
                # Interval width
                width = np.mean(bounds[:, 1] - bounds[:, 0])
                metrics[f'Width_{interval_name}'] = width
        
        # Quantile loss if available
        if 'quantiles' in prediction_intervals:
            quantile_loss = self._calculate_quantile_loss(
                actual, predicted, prediction_intervals['quantiles']
            )
            metrics['QuantileLoss'] = quantile_loss
        
        return metrics
    
    # Metric calculation methods
    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        config = self.metrics_config['scale_dependent']['MAE']
        
        errors = np.abs(actual - predicted)
        
        if config.get('robust_outliers', False):
            # Use median instead of mean for robustness
            return float(np.median(errors))
        
        if config.get('weight_recent', False):
            # Weight recent observations more heavily
            weights = np.linspace(0.5, 1.0, len(errors))
            return float(np.average(errors, weights=weights))
        
        return float(np.mean(errors))
    
    def _calculate_mse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return float(np.mean((actual - predicted) ** 2))
    
    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(self._calculate_mse(actual, predicted)))
    
    def _calculate_wape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Weighted Absolute Percentage Error."""
        config = self.metrics_config['scale_dependent']['WAPE']
        
        if config.get('aggregate_first', True):
            # Aggregate then calculate percentage
            return float(np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)))
        else:
            # Calculate percentage then aggregate
            with np.errstate(divide='ignore', invalid='ignore'):
                ape = np.abs((actual - predicted) / actual)
            return float(np.nanmean(ape))
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error with zero handling."""
        config = self.metrics_config['scale_independent']['MAPE']
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if config['handle_zeros'] == 'epsilon':
                # Add small epsilon to avoid division by zero
                denominator = np.where(actual == 0, config['epsilon'], actual)
                ape = np.abs((actual - predicted) / denominator)
            elif config['handle_zeros'] == 'skip':
                # Skip zero values
                non_zero_mask = actual != 0
                if not np.any(non_zero_mask):
                    return np.nan
                ape = np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / 
                           actual[non_zero_mask])
            else:
                ape = np.abs((actual - predicted) / actual)
        
        return float(np.nanmean(ape) * 100)
    
    def _calculate_smape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        
        numerator = np.abs(actual - predicted)
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            smape = numerator / denominator
        
        return float(np.nanmean(smape) * 100)
    
    def _calculate_mase(self, actual: np.ndarray, predicted: np.ndarray,
                       historical_data: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        config = self.metrics_config['scale_independent']['MASE']
        seasonal_period = config['seasonal_period']
        
        # Calculate MAE of predictions
        mae_pred = np.mean(np.abs(actual - predicted))
        
        # Calculate MAE of seasonal naive forecast on historical data
        if len(historical_data) > seasonal_period:
            naive_errors = np.abs(
                historical_data[seasonal_period:] - historical_data[:-seasonal_period]
            )
            mae_naive = np.mean(naive_errors)
            
            if mae_naive == 0:
                return np.inf if mae_pred > 0 else np.nan
            
            return float(mae_pred / mae_naive)
        else:
            warnings.warn("Insufficient historical data for MASE calculation")
            return np.nan
    
    def _calculate_rmsse(self, actual: np.ndarray, predicted: np.ndarray,
                        historical_data: np.ndarray) -> float:
        """Calculate Root Mean Squared Scaled Error."""
        config = self.metrics_config['scale_independent']['RMSSE']
        seasonal_period = config['seasonal_period']
        
        # Calculate RMSE of predictions
        rmse_pred = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Calculate RMSE of seasonal naive forecast on historical data
        if len(historical_data) > seasonal_period:
            naive_errors = (
                historical_data[seasonal_period:] - historical_data[:-seasonal_period]
            ) ** 2
            rmse_naive = np.sqrt(np.mean(naive_errors))
            
            if rmse_naive == 0:
                return np.inf if rmse_pred > 0 else np.nan
            
            return float(rmse_pred / rmse_naive)
        else:
            warnings.warn("Insufficient historical data for RMSSE calculation")
            return np.nan
    
    def _calculate_interval_coverage(self, actual: np.ndarray, 
                                   lower: np.ndarray, upper: np.ndarray) -> float:
        """Calculate prediction interval coverage."""
        
        coverage = np.mean((actual >= lower) & (actual <= upper))
        return float(coverage)
    
    def _calculate_quantile_loss(self, actual: np.ndarray, predicted: np.ndarray,
                               quantile_predictions: Optional[Dict] = None) -> float:
        """Calculate quantile loss for probabilistic forecasts."""
        
        if quantile_predictions is None:
            # Assume median prediction
            quantiles = [0.5]
            predictions = {'0.5': predicted}
        else:
            quantiles = list(quantile_predictions.keys())
            predictions = quantile_predictions
        
        total_loss = 0.0
        
        for q_str in quantiles:
            q = float(q_str)
            q_pred = predictions[q_str]
            
            errors = actual - q_pred
            loss = np.maximum(q * errors, (q - 1) * errors)
            total_loss += np.mean(loss)
        
        return float(total_loss / len(quantiles))
    
    def _calculate_crps(self, actual: np.ndarray, predicted: np.ndarray,
                       prediction_intervals: Optional[Dict] = None) -> float:
        """Calculate Continuous Ranked Probability Score."""
        
        # Simplified CRPS calculation - in practice, this would need
        # full distributional forecasts
        if prediction_intervals is None:
            # Use point forecast assumption
            return float(np.mean(np.abs(actual - predicted)))
        
        # More sophisticated CRPS calculation would go here
        return np.nan
    
    def _validate_inputs(self, actual: np.ndarray, 
                        predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess inputs."""
        
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        
        if actual.shape != predicted.shape:
            raise ValueError(f"Shape mismatch: actual {actual.shape}, predicted {predicted.shape}")
        
        if len(actual) == 0:
            raise ValueError("Empty arrays provided")
        
        return actual, predicted


class StatisticalTestSuite:
    """Suite of statistical tests for time series model comparison."""
    
    def __init__(self):
        self.test_results_history = []
    
    def diebold_mariano_test(self, errors1: np.ndarray, errors2: np.ndarray,
                           h: int = 1, alternative: str = 'two-sided') -> Dict[str, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.
        
        Args:
            errors1: Forecast errors from model 1
            errors2: Forecast errors from model 2
            h: Forecast horizon
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test statistic and p-value
        """
        
        if len(errors1) != len(errors2):
            raise ValueError("Error arrays must have same length")
        
        # Calculate loss differential
        d = errors1**2 - errors2**2
        
        # Mean of loss differential
        d_mean = np.mean(d)
        
        # Variance of loss differential (with HAC correction for h > 1)
        if h == 1:
            d_var = np.var(d, ddof=1)
        else:
            # Newey-West HAC estimator
            d_var = self._newey_west_variance(d, h-1)
        
        # Test statistic
        if d_var == 0:
            dm_stat = np.inf if d_mean != 0 else 0
        else:
            dm_stat = d_mean / np.sqrt(d_var / len(d))
        
        # P-value calculation
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        elif alternative == 'less':
            p_value = stats.norm.cdf(dm_stat)
        else:  # 'greater'
            p_value = 1 - stats.norm.cdf(dm_stat)
        
        result = {
            'statistic': float(dm_stat),
            'p_value': float(p_value),
            'alternative': alternative,
            'method': 'Diebold-Mariano Test'
        }
        
        self.test_results_history.append(result)
        
        return result
    
    def _newey_west_variance(self, series: np.ndarray, lags: int) -> float:
        """Calculate Newey-West HAC variance estimator."""
        
        n = len(series)
        series_centered = series - np.mean(series)
        
        # Zero lag
        variance = np.sum(series_centered**2) / n
        
        # Add lag terms
        for lag in range(1, lags + 1):
            gamma = np.sum(series_centered[lag:] * series_centered[:-lag]) / n
            weight = 1 - lag / (lags + 1)  # Bartlett kernel
            variance += 2 * weight * gamma
        
        return variance
    
    def wilcoxon_signed_rank_test(self, errors1: np.ndarray, 
                                errors2: np.ndarray) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for non-parametric comparison.
        """
        
        differences = errors1**2 - errors2**2
        
        # Remove zeros
        differences = differences[differences != 0]
        
        if len(differences) == 0:
            return {'statistic': 0, 'p_value': 1.0, 'method': 'Wilcoxon Signed-Rank'}
        
        statistic, p_value = stats.wilcoxon(differences)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'method': 'Wilcoxon Signed-Rank Test'
        }
    
    def superior_predictive_ability_test(self, benchmark_errors: np.ndarray,
                                       *alternative_errors: np.ndarray,
                                       bootstrap_samples: int = 1000) -> Dict[str, float]:
        """
        Superior Predictive Ability (SPA) test using bootstrap.
        """
        
        n_models = len(alternative_errors)
        n_obs = len(benchmark_errors)
        
        # Calculate loss differentials
        loss_diffs = []
        for alt_errors in alternative_errors:
            diff = benchmark_errors**2 - alt_errors**2
            loss_diffs.append(diff)
        
        loss_diffs = np.array(loss_diffs)
        
        # Original test statistic
        mean_diffs = np.mean(loss_diffs, axis=1)
        max_stat = np.max(mean_diffs)
        
        # Bootstrap procedure
        bootstrap_stats = []
        
        for _ in range(bootstrap_samples):
            # Resample indices
            indices = np.random.choice(n_obs, n_obs, replace=True)
            bootstrap_diffs = loss_diffs[:, indices]
            
            # Recenter around original means
            bootstrap_diffs_centered = bootstrap_diffs - np.mean(bootstrap_diffs, axis=1, keepdims=True) + mean_diffs.reshape(-1, 1)
            
            bootstrap_means = np.mean(bootstrap_diffs_centered, axis=1)
            bootstrap_max = np.max(bootstrap_means)
            bootstrap_stats.append(bootstrap_max)
        
        # P-value
        p_value = np.mean(np.array(bootstrap_stats) >= max_stat)
        
        return {
            'statistic': float(max_stat),
            'p_value': float(p_value),
            'method': 'Superior Predictive Ability Test',
            'bootstrap_samples': bootstrap_samples
        }


class BusinessImpactEvaluator:
    """Evaluator for business impact and ROI of forecasting models."""
    
    def __init__(self):
        self.impact_models = {
            'inventory_cost': self._calculate_inventory_impact,
            'service_level': self._calculate_service_level_impact,
            'revenue_impact': self._calculate_revenue_impact,
            'operational_efficiency': self._calculate_operational_impact
        }
    
    def evaluate_business_impact(self, actual: np.ndarray, predicted: np.ndarray,
                               business_context: Dict) -> Dict[str, float]:
        """
        Evaluate business impact of forecasting performance.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            business_context: Business parameters and constraints
            
        Returns:
            Dictionary of business impact metrics
        """
        
        impact_metrics = {}
        
        # Calculate basic forecast errors
        errors = predicted - actual
        abs_errors = np.abs(errors)
        
        # Inventory impact
        if 'inventory_parameters' in business_context:
            inventory_impact = self._calculate_inventory_impact(
                errors, business_context['inventory_parameters']
            )
            impact_metrics.update(inventory_impact)
        
        # Service level impact
        if 'service_parameters' in business_context:
            service_impact = self._calculate_service_level_impact(
                errors, business_context['service_parameters']
            )
            impact_metrics.update(service_impact)
        
        # Revenue impact
        if 'revenue_parameters' in business_context:
            revenue_impact = self._calculate_revenue_impact(
                errors, actual, business_context['revenue_parameters']
            )
            impact_metrics.update(revenue_impact)
        
        # ROI calculation
        if 'cost_parameters' in business_context:
            roi_metrics = self._calculate_roi_metrics(
                impact_metrics, business_context['cost_parameters']
            )
            impact_metrics.update(roi_metrics)
        
        return impact_metrics
    
    def _calculate_inventory_impact(self, errors: np.ndarray, 
                                  params: Dict) -> Dict[str, float]:
        """Calculate inventory-related business impact."""
        
        holding_cost_per_unit = params.get('holding_cost_per_unit', 1.0)
        stockout_cost_per_unit = params.get('stockout_cost_per_unit', 10.0)
        
        # Overforecasting leads to excess inventory
        overforecast_cost = np.sum(np.maximum(errors, 0)) * holding_cost_per_unit
        
        # Underforecasting leads to stockouts
        underforecast_cost = np.sum(np.maximum(-errors, 0)) * stockout_cost_per_unit
        
        total_inventory_cost = overforecast_cost + underforecast_cost
        
        return {
            'inventory_holding_cost': float(overforecast_cost),
            'stockout_cost': float(underforecast_cost),
            'total_inventory_cost': float(total_inventory_cost)
        }
    
    def _calculate_service_level_impact(self, errors: np.ndarray,
                                      params: Dict) -> Dict[str, float]:
        """Calculate service level impact."""
        
        target_service_level = params.get('target_service_level', 0.95)
        
        # Service level is percentage of periods without stockouts
        # (assuming stockout when actual > predicted)
        stockouts = np.sum(errors < 0)
        achieved_service_level = 1 - (stockouts / len(errors))
        
        service_level_gap = target_service_level - achieved_service_level
        
        return {
            'achieved_service_level': float(achieved_service_level),
            'service_level_gap': float(service_level_gap),
            'stockout_frequency': float(stockouts / len(errors))
        }
    
    def _calculate_revenue_impact(self, errors: np.ndarray, actual: np.ndarray,
                                params: Dict) -> Dict[str, float]:
        """Calculate revenue impact of forecast errors."""
        
        unit_price = params.get('unit_price', 1.0)
        demand_elasticity = params.get('demand_elasticity', -0.5)
        
        # Revenue loss from stockouts (lost sales)
        stockout_units = np.maximum(-errors, 0)
        lost_revenue = np.sum(stockout_units) * unit_price
        
        # Revenue from actual sales
        total_revenue = np.sum(actual) * unit_price
        
        # Revenue loss percentage
        revenue_loss_pct = lost_revenue / total_revenue if total_revenue > 0 else 0
        
        return {
            'lost_revenue': float(lost_revenue),
            'total_revenue': float(total_revenue),
            'revenue_loss_percentage': float(revenue_loss_pct * 100)
        }
    
    def _calculate_operational_impact(self, errors: np.ndarray,
                                    params: Dict) -> Dict[str, float]:
        """Calculate operational efficiency impact."""
        
        # Placeholder for operational impact calculation
        forecast_accuracy = 1 - (np.mean(np.abs(errors)) / np.mean(np.abs(params.get('baseline_demand', errors))))
        
        return {
            'forecast_accuracy': float(max(0, forecast_accuracy)),
            'operational_efficiency_score': float(max(0, forecast_accuracy * 100))
        }
    
    def _calculate_roi_metrics(self, impact_metrics: Dict[str, float],
                             cost_params: Dict) -> Dict[str, float]:
        """Calculate ROI metrics for the forecasting system."""
        
        # System costs
        development_cost = cost_params.get('development_cost', 0)
        operational_cost_per_period = cost_params.get('operational_cost_per_period', 0)
        
        # Benefits (cost savings)
        total_benefits = 0
        if 'total_inventory_cost' in impact_metrics:
            baseline_inventory_cost = cost_params.get('baseline_inventory_cost', 0)
            inventory_savings = baseline_inventory_cost - impact_metrics['total_inventory_cost']
            total_benefits += inventory_savings
        
        if 'lost_revenue' in impact_metrics:
            baseline_lost_revenue = cost_params.get('baseline_lost_revenue', 0)
            revenue_savings = baseline_lost_revenue - impact_metrics['lost_revenue']
            total_benefits += revenue_savings
        
        # ROI calculation
        total_costs = development_cost + operational_cost_per_period
        
        if total_costs > 0:
            roi = (total_benefits - total_costs) / total_costs * 100
        else:
            roi = float('inf') if total_benefits > 0 else 0
        
        return {
            'total_benefits': float(total_benefits),
            'total_costs': float(total_costs),
            'roi_percentage': float(roi),
            'payback_periods': float(total_costs / max(total_benefits, 1e-10))
        }
