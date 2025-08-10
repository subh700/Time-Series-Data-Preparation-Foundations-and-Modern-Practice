import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt

class ForecastingMetrics:
    """Comprehensive guide to time series forecasting evaluation metrics."""
    
    def __init__(self):
        self.metrics_catalog = self._create_metrics_catalog()
        
    def _create_metrics_catalog(self) -> Dict[str, Any]:
        """Create comprehensive catalog of forecasting metrics."""
        
        return {
            "scale_dependent": {
                "category": "Scale-Dependent Metrics",
                "description": "Metrics that depend on the scale of the data",
                "when_to_use": "When all time series have similar scales or you want to weight large values more",
                "metrics": {
                    "MAE": {
                        "name": "Mean Absolute Error",
                        "formula": "mean(|y_true - y_pred|)",
                        "interpretation": "Average absolute difference between predictions and actual values",
                        "advantages": ["Easy to interpret", "Robust to outliers", "Linear scoring"],
                        "disadvantages": ["Scale dependent", "Not differentiable at zero"],
                        "best_for": "When you want equal weight for all errors",
                        "python_code": "sklearn.metrics.mean_absolute_error(y_true, y_pred)"
                    },
                    
                    "MSE": {
                        "name": "Mean Squared Error", 
                        "formula": "mean((y_true - y_pred)²)",
                        "interpretation": "Average squared difference between predictions and actual values",
                        "advantages": ["Penalizes large errors", "Differentiable", "Mathematical properties"],
                        "disadvantages": ["Scale dependent", "Sensitive to outliers", "Units are squared"],
                        "best_for": "When large errors are particularly undesirable",
                        "python_code": "sklearn.metrics.mean_squared_error(y_true, y_pred)"
                    },
                    
                    "RMSE": {
                        "name": "Root Mean Squared Error",
                        "formula": "sqrt(mean((y_true - y_pred)²))",
                        "interpretation": "Square root of MSE, in original units",
                        "advantages": ["Same units as target", "Penalizes large errors", "Widely used"],
                        "disadvantages": ["Scale dependent", "Sensitive to outliers"],
                        "best_for": "When you want MSE benefits but in original units",
                        "python_code": "np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))"
                    }
                }
            },
            
            "percentage_based": {
                "category": "Percentage-Based Metrics",
                "description": "Metrics expressed as percentages, scale-independent", 
                "when_to_use": "When comparing across different scales or want intuitive percentages",
                "metrics": {
                    "MAPE": {
                        "name": "Mean Absolute Percentage Error",
                        "formula": "mean(|100 * (y_true - y_pred) / y_true|)",
                        "interpretation": "Average absolute percentage error",
                        "advantages": ["Scale independent", "Easy to interpret", "Widely understood"],
                        "disadvantages": ["Undefined for zero values", "Asymmetric", "Biased toward underforecasts"],
                        "best_for": "Business reporting when values are always positive",
                        "python_code": "sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred)"
                    },
                    
                    "SMAPE": {
                        "name": "Symmetric Mean Absolute Percentage Error",
                        "formula": "mean(200 * |y_true - y_pred| / (|y_true| + |y_pred|))",
                        "interpretation": "Symmetric version of MAPE",
                        "advantages": ["More symmetric than MAPE", "Bounded between 0-200%"],
                        "disadvantages": ["Still problematic with zeros", "Less intuitive than MAPE"],
                        "best_for": "When you need symmetry but still want percentage interpretation",
                        "python_code": "2 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))"
                    }
                }
            },
            
            "scaled_metrics": {
                "category": "Scaled Metrics",
                "description": "Metrics scaled by a naive forecast benchmark",
                "when_to_use": "When comparing models across different time series or scales",
                "metrics": {
                    "MASE": {
                        "name": "Mean Absolute Scaled Error",
                        "formula": "MAE / MAE_naive",
                        "interpretation": "Error relative to naive forecast (seasonal or simple)",
                        "advantages": ["Scale independent", "Interpretable benchmark", "No division by zero issues"],
                        "disadvantages": ["Requires training data", "Benchmark dependent"],
                        "best_for": "Comparing models across different time series",
                        "python_code": "Custom implementation required"
                    },
                    
                    "RMSSE": {
                        "name": "Root Mean Squared Scaled Error", 
                        "formula": "sqrt(MSE / MSE_naive)",
                        "interpretation": "RMSE relative to naive forecast",
                        "advantages": ["Scale independent", "Penalizes large errors", "Benchmark comparison"],
                        "disadvantages": ["Requires training data", "Complex calculation"],
                        "best_for": "When you want scaled metric with MSE properties",
                        "python_code": "Custom implementation required"
                    }
                }
            },
            
            "probabilistic_metrics": {
                "category": "Probabilistic Metrics",
                "description": "Metrics for probabilistic/interval forecasts",
                "when_to_use": "When working with probabilistic forecasts or uncertainty quantification",
                "metrics": {
                    "CRPS": {
                        "name": "Continuous Ranked Probability Score",
                        "formula": "Integral of (F(y) - I(y >= observation))² dy",
                        "interpretation": "Measures accuracy of entire predictive distribution",
                        "advantages": ["Proper scoring rule", "Evaluates full distribution", "Reduces to MAE for point forecasts"],
                        "disadvantages": ["Complex to compute", "Requires probabilistic forecasts"],
                        "best_for": "Evaluating probabilistic forecasting models",
                        "python_code": "properscoring.crps_gaussian(observations, forecasts, std)"
                    },
                    
                    "Quantile_Loss": {
                        "name": "Quantile Loss",
                        "formula": "mean(max(τ(y_true - y_pred), (τ-1)(y_true - y_pred)))",
                        "interpretation": "Asymmetric loss for quantile forecasts",
                        "advantages": ["Proper scoring rule", "Evaluates specific quantiles", "Asymmetric penalty"],
                        "disadvantages": ["Only evaluates single quantile", "Requires quantile forecasts"],
                        "best_for": "Evaluating specific quantile predictions",
                        "python_code": "Custom implementation based on quantile level"
                    }
                }
            }
        }
    
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_train: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive set of forecasting metrics."""
        
        metrics = {}
        
        # Scale-dependent metrics
        metrics['MAE'] = np.mean(np.abs(y_true - y_pred))
        metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        
        # Percentage-based metrics (handle zeros)
        mask = y_true != 0
        if mask.any():
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = np.inf
            
        # SMAPE
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if mask.any():
            metrics['SMAPE'] = np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            metrics['SMAPE'] = 0
        
        # Scaled metrics (if training data provided)
        if y_train is not None:
            naive_errors = np.abs(np.diff(y_train))
            if len(naive_errors) > 0:
                naive_mae = np.mean(naive_errors)
                metrics['MASE'] = metrics['MAE'] / naive_mae if naive_mae != 0 else np.inf
        
        return metrics
    
    def metric_selection_guide(self, data_characteristics: Dict[str, Any]) -> Dict[str, str]:
        """Provide metric selection guidance based on data characteristics."""
        
        recommendations = {
            "primary_metric": None,
            "secondary_metrics": [],
            "avoid_metrics": [],
            "reasoning": ""
        }
        
        # Decision tree for metric selection
        has_zeros = data_characteristics.get('has_zeros', False)
        is_sparse = data_characteristics.get('is_sparse', False)
        scale_varies = data_characteristics.get('scale_varies', False)
        need_interpretability = data_characteristics.get('need_interpretability', True)
        care_about_large_errors = data_characteristics.get('care_about_large_errors', False)
        
        if has_zeros or is_sparse:
            if scale_varies:
                recommendations['primary_metric'] = 'MASE'
                recommendations['secondary_metrics'] = ['MAE', 'RMSE']
                recommendations['avoid_metrics'] = ['MAPE', 'SMAPE']
                recommendations['reasoning'] = "MASE avoids division by zero and handles scale differences"
            else:
                recommendations['primary_metric'] = 'MAE'
                recommendations['secondary_metrics'] = ['RMSE', 'MASE']
                recommendations['avoid_metrics'] = ['MAPE', 'SMAPE']
                recommendations['reasoning'] = "MAE is robust to zeros and outliers"
        
        elif scale_varies:
            if need_interpretability:
                recommendations['primary_metric'] = 'MAPE'
                recommendations['secondary_metrics'] = ['MASE', 'SMAPE']
                recommendations['avoid_metrics'] = []
                recommendations['reasoning'] = "MAPE provides interpretable percentages across scales"
            else:
                recommendations['primary_metric'] = 'MASE'
                recommendations['secondary_metrics'] = ['MAPE', 'RMSSE']
                recommendations['avoid_metrics'] = []
                recommendations['reasoning'] = "MASE provides robust cross-scale comparison"
        
        elif care_about_large_errors:
            recommendations['primary_metric'] = 'RMSE'
            recommendations['secondary_metrics'] = ['MSE', 'MAE']
            recommendations['avoid_metrics'] = []
            recommendations['reasoning'] = "RMSE penalizes large errors more heavily"
        
        else:
            recommendations['primary_metric'] = 'MAE'
            recommendations['secondary_metrics'] = ['RMSE', 'MAPE']
            recommendations['avoid_metrics'] = []
            recommendations['reasoning'] = "MAE provides balanced, interpretable error measurement"
        
        return recommendations
    
    def create_metrics_comparison_chart(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                                      y_train: np.ndarray = None) -> pd.DataFrame:
        """Create comparison chart of different models using multiple metrics."""
        
        results = []
        
        for model_name, y_pred in predictions.items():
            metrics = self.calculate_all_metrics(y_true, y_pred, y_train)
            metrics['Model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Reorder columns to put Model first
        cols = ['Model'] + [col for col in df.columns if col != 'Model']
        df = df[cols]
        
        return df.round(4)

# Demonstrate metrics system
metrics_system = ForecastingMetrics()

print("📊 TIME SERIES FORECASTING EVALUATION METRICS GUIDE")
print("=" * 60)

# Show metrics categories
print("\n📋 METRICS CATEGORIES:")
for category_key, category in metrics_system.metrics_catalog.items():
    print(f"\n• {category['category']}")
    print(f"  Description: {category['description']}")
    print(f"  When to use: {category['when_to_use']}")
    print(f"  Available metrics: {len(category['metrics'])}")

# Show detailed metric information
print(f"\n🔍 DETAILED METRIC INFORMATION:")
mae_info = metrics_system.metrics_catalog['scale_dependent']['metrics']['MAE']
print(f"\nMAE (Mean Absolute Error):")
print(f"  Formula: {mae_info['formula']}")
print(f"  Interpretation: {mae_info['interpretation']}")
print(f"  Best for: {mae_info['best_for']}")
print(f"  Advantages: {', '.join(mae_info['advantages'][:2])}")

# Demonstrate metric selection guide
sample_characteristics = {
    'has_zeros': True,
    'is_sparse': False,
    'scale_varies': True,
    'need_interpretability': True,
    'care_about_large_errors': False
}

recommendation = metrics_system.metric_selection_guide(sample_characteristics)
print(f"\n🎯 METRIC SELECTION RECOMMENDATION:")
print(f"Data characteristics: {sample_characteristics}")
print(f"Primary metric: {recommendation['primary_metric']}")
print(f"Secondary metrics: {', '.join(recommendation['secondary_metrics'])}")
print(f"Reasoning: {recommendation['reasoning']}")

# Create sample comparison
print(f"\n📈 SAMPLE METRICS COMPARISON:")
np.random.seed(42)
y_true = np.random.randn(100) * 10 + 50
y_train = np.random.randn(200) * 10 + 50

predictions = {
    'Naive': y_true + np.random.randn(100) * 2,
    'ARIMA': y_true + np.random.randn(100) * 1.5,
    'Prophet': y_true + np.random.randn(100) * 1.8,
    'LSTM': y_true + np.random.randn(100) * 1.2
}

comparison_df = metrics_system.create_metrics_comparison_chart(y_true, predictions, y_train)
print(comparison_df.to_string(index=False))

print(f"\n💡 INTERPRETATION GUIDE:")
print("• Lower values are better for all metrics shown")
print("• MASE < 1.0 means better than naive forecast")
print("• MAPE values are in percentage points")
print("• Compare metrics within same category for best insights")
