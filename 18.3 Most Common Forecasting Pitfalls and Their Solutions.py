import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Statistical testing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

@dataclass
class DebuggingConfig:
    """Configuration for time series debugging toolkit."""
    
    # Test parameters
    stationarity_pvalue_threshold: float = 0.05
    seasonality_detection_threshold: float = 0.3
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Validation parameters
    lookhead_check_enabled: bool = True
    overfitting_check_enabled: bool = True
    data_leakage_check_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    save_debug_plots: bool = True
    debug_output_dir: str = "debug_output/"


class TimeSeriesDebugger:
    """
    Comprehensive debugging toolkit for time series forecasting projects.
    Based on TimeGym principles and common forecasting pitfalls.
    """
    
    def __init__(self, config: DebuggingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.debug_results = {}
        self.test_results = {}
        
        # Create output directory
        import os
        os.makedirs(config.debug_output_dir, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for debugging."""
        
        logger = logging.getLogger("TimeSeriesDebugger")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_comprehensive_debug(self, 
                               data: pd.DataFrame,
                               target_column: str,
                               time_column: str,
                               model=None,
                               predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run comprehensive debugging suite on time series data and model.
        
        Args:
            data: Time series dataset
            target_column: Name of target variable
            time_column: Name of time column
            model: Trained model (optional)
            predictions: Model predictions (optional)
            
        Returns:
            Comprehensive debugging report
        """
        
        self.logger.info("Starting comprehensive time series debugging")
        
        debug_report = {
            'data_quality_issues': self._check_data_quality(data, target_column, time_column),
            'stationarity_issues': self._check_stationarity_issues(data, target_column),
            'seasonality_issues': self._check_seasonality_issues(data, target_column, time_column),
            'modeling_pitfalls': self._check_modeling_pitfalls(data, target_column, time_column),
            'validation_issues': self._check_validation_issues(data, target_column, time_column),
            'prediction_issues': self._check_prediction_issues(predictions, data, target_column) if predictions is not None else {},
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        debug_report['recommendations'] = self._generate_recommendations(debug_report)
        
        # Save debug report
        self._save_debug_report(debug_report)
        
        # Create visualizations
        if self.config.save_debug_plots:
            self._create_debug_visualizations(data, target_column, time_column, debug_report)
        
        self.logger.info("Debugging completed successfully")
        return debug_report
    
    def _check_data_quality(self, 
                           data: pd.DataFrame, 
                           target_column: str, 
                           time_column: str) -> Dict[str, Any]:
        """Check for data quality issues."""
        
        self.logger.info("Checking data quality issues")
        
        issues = {
            'missing_values': {},
            'outliers': {},
            'data_types': {},
            'temporal_issues': {},
            'duplicates': {},
            'severity': 'low'
        }
        
        # Missing values check
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        critical_missing = missing_percentages[missing_percentages > 10]
        if len(critical_missing) > 0:
            issues['missing_values']['critical_columns'] = critical_missing.to_dict()
            issues['severity'] = 'high'
        
        target_missing = missing_counts[target_column]
        if target_missing > 0:
            issues['missing_values']['target_missing'] = target_missing
            issues['missing_values']['target_missing_pct'] = missing_percentages[target_column]
            if missing_percentages[target_column] > 5:
                issues['severity'] = 'high'
        
        # Outlier detection
        target_series = data[target_column].dropna()
        
        if self.config.outlier_detection_method == "iqr":
            Q1 = target_series.quantile(0.25)
            Q3 = target_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = target_series[(target_series < lower_bound) | (target_series > upper_bound)]
            outlier_percentage = (len(outliers) / len(target_series)) * 100
            
            issues['outliers'] = {
                'method': 'IQR',
                'count': len(outliers),
                'percentage': outlier_percentage,
                'indices': outliers.index.tolist()[:10],  # First 10 for brevity
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
            if outlier_percentage > 5:
                issues['severity'] = 'medium'
        
        # Data type checks
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            issues['data_types']['time_column_not_datetime'] = True
            issues['severity'] = 'medium'
        
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            issues['data_types']['target_not_numeric'] = True
            issues['severity'] = 'high'
        
        # Temporal issues
        if pd.api.types.is_datetime64_any_dtype(data[time_column]):
            time_diffs = data[time_column].diff().dropna()
            
            # Check for irregular intervals
            if len(time_diffs.unique()) > 1:
                issues['temporal_issues']['irregular_intervals'] = True
                issues['temporal_issues']['interval_variations'] = len(time_diffs.unique())
        
        # Duplicate time stamps
        duplicate_times = data[time_column].duplicated().sum()
        if duplicate_times > 0:
            issues['duplicates']['duplicate_timestamps'] = duplicate_times
            issues['severity'] = 'medium'
        
        return issues
    
    def _check_stationarity_issues(self, 
                                  data: pd.DataFrame, 
                                  target_column: str) -> Dict[str, Any]:
        """Check for stationarity-related issues."""
        
        self.logger.info("Checking stationarity issues")
        
        issues = {
            'non_stationary': False,
            'tests': {},
            'trend_detected': False,
            'variance_issues': False,
            'recommendations': [],
            'severity': 'low'
        }
        
        target_series = data[target_column].dropna()
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(target_series)
            issues['tests']['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.config.stationarity_pvalue_threshold
            }
            
            if not issues['tests']['adf']['is_stationary']:
                issues['non_stationary'] = True
                issues['severity'] = 'high'
                issues['recommendations'].append("Apply differencing to make series stationary")
        
        except Exception as e:
            self.logger.warning(f"ADF test failed: {str(e)}")
        
        # KPSS test
        try:
            kpss_result = kpss(target_series, regression='c')
            issues['tests']['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > self.config.stationarity_pvalue_threshold
            }
            
            if not issues['tests']['kpss']['is_stationary']:
                issues['non_stationary'] = True
                issues['severity'] = 'high'
        
        except Exception as e:
            self.logger.warning(f"KPSS test failed: {str(e)}")
        
        # Trend detection
        x = np.arange(len(target_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, target_series.values)
        
        if abs(r_value) > 0.3 and p_value < 0.05:
            issues['trend_detected'] = True
            issues['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
            issues['trend_strength'] = abs(r_value)
            issues['severity'] = 'medium'
            issues['recommendations'].append("Consider detrending the series")
        
        # Variance stability check
        # Split series into chunks and check variance consistency
        chunk_size = len(target_series) // 4
        if chunk_size > 10:
            chunks = [target_series.iloc[i:i+chunk_size] for i in range(0, len(target_series), chunk_size)]
            variances = [chunk.var() for chunk in chunks if len(chunk) > 5]
            
            if len(variances) > 1:
                var_ratio = max(variances) / min(variances)
                if var_ratio > 4:  # Significant variance change
                    issues['variance_issues'] = True
                    issues['variance_ratio'] = var_ratio
                    issues['severity'] = 'medium'
                    issues['recommendations'].append("Consider log transformation to stabilize variance")
        
        return issues
    
    def _check_seasonality_issues(self, 
                                 data: pd.DataFrame, 
                                 target_column: str, 
                                 time_column: str) -> Dict[str, Any]:
        """Check for seasonality-related issues."""
        
        self.logger.info("Checking seasonality issues")
        
        issues = {
            'seasonality_ignored': False,
            'seasonal_patterns': {},
            'decomposition_possible': False,
            'recommendations': [],
            'severity': 'low'
        }
        
        # Set time index for analysis
        data_copy = data.copy()
        if pd.api.types.is_datetime64_any_dtype(data_copy[time_column]):
            data_copy = data_copy.set_index(time_column)
            target_series = data_copy[target_column].dropna()
            
            # Check if series is long enough for seasonal decomposition
            if len(target_series) >= 24:  # Minimum for seasonal analysis
                issues['decomposition_possible'] = True
                
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Try different seasonal periods
                    periods_to_test = []
                    
                    # Infer frequency from data
                    freq = pd.infer_freq(target_series.index)
                    if freq:
                        if 'D' in freq:
                            periods_to_test = [7, 30, 365]  # Weekly, monthly, yearly
                        elif 'H' in freq:
                            periods_to_test = [24, 168, 8760]  # Daily, weekly, yearly
                        elif 'M' in freq:
                            periods_to_test = [12]  # Yearly
                    else:
                        periods_to_test = [12]  # Default
                    
                    for period in periods_to_test:
                        if len(target_series) >= 2 * period:
                            try:
                                decomp = seasonal_decompose(target_series, period=period, model='additive')
                                
                                # Calculate seasonal strength
                                seasonal_var = np.var(decomp.seasonal.dropna())
                                resid_var = np.var(decomp.resid.dropna())
                                
                                if seasonal_var > 0 and resid_var > 0:
                                    seasonal_strength = seasonal_var / (seasonal_var + resid_var)
                                    
                                    issues['seasonal_patterns'][f'period_{period}'] = {
                                        'strength': seasonal_strength,
                                        'significant': seasonal_strength > self.config.seasonality_detection_threshold
                                    }
                                    
                                    if seasonal_strength > self.config.seasonality_detection_threshold:
                                        issues['seasonality_ignored'] = True
                                        issues['severity'] = 'high'
                                        issues['recommendations'].append(
                                            f"Strong seasonality detected with period {period}. Use seasonal models (SARIMA, seasonal decomposition)"
                                        )
                            
                            except Exception as e:
                                self.logger.warning(f"Seasonal decomposition failed for period {period}: {str(e)}")
                
                except ImportError:
                    self.logger.warning("statsmodels not available for seasonal decomposition")
        
        return issues
    
    def _check_modeling_pitfalls(self, 
                                data: pd.DataFrame, 
                                target_column: str, 
                                time_column: str) -> Dict[str, Any]:
        """Check for common modeling pitfalls."""
        
        self.logger.info("Checking modeling pitfalls")
        
        issues = {
            'look_ahead_bias_risk': False,
            'data_leakage_risk': False,
            'insufficient_data': False,
            'feature_engineering_issues': {},
            'recommendations': [],
            'severity': 'low'
        }
        
        # Check data size
        if len(data) < 100:
            issues['insufficient_data'] = True
            issues['data_size'] = len(data)
            issues['severity'] = 'high'
            issues['recommendations'].append("Consider collecting more data for reliable modeling")
        
        # Check for potential look-ahead bias in features
        if self.config.lookhead_check_enabled:
            feature_columns = [col for col in data.columns 
                             if col not in [target_column, time_column]]
            
            suspicious_features = []
            for col in feature_columns:
                # Check if feature has unrealistic correlation with target
                if pd.api.types.is_numeric_dtype(data[col]):
                    correlation = data[col].corr(data[target_column])
                    if abs(correlation) > 0.95:  # Suspiciously high correlation
                        suspicious_features.append({
                            'feature': col,
                            'correlation': correlation
                        })
            
            if suspicious_features:
                issues['look_ahead_bias_risk'] = True
                issues['suspicious_features'] = suspicious_features
                issues['severity'] = 'high'
                issues['recommendations'].append("Check for look-ahead bias in highly correlated features")
        
        # Check temporal ordering
        if pd.api.types.is_datetime64_any_dtype(data[time_column]):
            if not data[time_column].is_monotonic_increasing:
                issues['temporal_ordering'] = False
                issues['severity'] = 'medium'
                issues['recommendations'].append("Ensure data is sorted by time column")
        
        return issues
    
    def _check_validation_issues(self, 
                                data: pd.DataFrame, 
                                target_column: str, 
                                time_column: str) -> Dict[str, Any]:
        """Check for validation-related issues."""
        
        self.logger.info("Checking validation issues")
        
        issues = {
            'temporal_split_required': True,
            'cross_validation_issues': {},
            'test_set_contamination_risk': False,
            'recommendations': [],
            'severity': 'low'
        }
        
        # Time series require temporal splits
        issues['temporal_split_required'] = True
        issues['recommendations'].append("Use temporal train-test splits, not random splits")
        
        # Check for sufficient data for proper validation
        min_test_size = max(30, len(data) * 0.1)  # At least 30 observations or 10%
        if len(data) < min_test_size * 3:  # Need at least 3x test size for train
            issues['insufficient_for_validation'] = True
            issues['severity'] = 'medium'
            issues['recommendations'].append("Consider gathering more data for robust validation")
        
        return issues
    
    def _check_prediction_issues(self, 
                               predictions: np.ndarray,
                               data: pd.DataFrame,
                               target_column: str) -> Dict[str, Any]:
        """Check for prediction-related issues."""
        
        self.logger.info("Checking prediction issues")
        
        issues = {
            'prediction_quality': {},
            'residual_issues': {},
            'recommendations': [],
            'severity': 'low'
        }
        
        # Assuming predictions are for the last part of the data
        actual_values = data[target_column].dropna().iloc[-len(predictions):].values
        
        if len(actual_values) == len(predictions):
            residuals = actual_values - predictions
            
            # Check residual properties
            issues['residual_issues'] = {
                'mean_residual': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals),
                'residual_kurtosis': stats.kurtosis(residuals)
            }
            
            # Check if residuals are normally distributed
            _, p_value = stats.normaltest(residuals)
            if p_value < 0.05:
                issues['residual_issues']['non_normal_residuals'] = True
                issues['recommendations'].append("Residuals are not normally distributed - check model assumptions")
            
            # Check for autocorrelation in residuals
            try:
                lb_stat, lb_p_value = acorr_ljungbox(residuals, lags=[10], return_df=True).iloc[0]
                if lb_p_value < 0.05:
                    issues['residual_issues']['autocorrelated_residuals'] = True
                    issues['severity'] = 'medium'
                    issues['recommendations'].append("Residuals show autocorrelation - model may be underfitting")
            except:
                pass
        
        return issues
    
    def _generate_recommendations(self, debug_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on debug findings."""
        
        recommendations = []
        
        # Collect all recommendations from individual checks
        for section, results in debug_report.items():
            if isinstance(results, dict) and 'recommendations' in results:
                recommendations.extend(results['recommendations'])
        
        # Add general recommendations based on severity
        high_severity_issues = [
            section for section, results in debug_report.items()
            if isinstance(results, dict) and results.get('severity') == 'high'
        ]
        
        if high_severity_issues:
            recommendations.insert(0, f"🚨 Critical issues found in: {', '.join(high_severity_issues)}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _save_debug_report(self, debug_report: Dict[str, Any]):
        """Save debugging report to file."""
        
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        serializable_report = convert_numpy(debug_report)
        
        report_path = f"{self.config.debug_output_dir}debug_report.json"
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        self.logger.info(f"Debug report saved to {report_path}")
    
    def _create_debug_visualizations(self, 
                                   data: pd.DataFrame,
                                   target_column: str,
                                   time_column: str,
                                   debug_report: Dict[str, Any]):
        """Create comprehensive debug visualizations."""
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Time Series Plot', 'Distribution',
                'Autocorrelation', 'Seasonal Decomposition',
                'Outliers', 'Residuals (if available)'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        target_series = data[target_column].dropna()
        
        # 1. Time series plot
        fig.add_trace(
            go.Scatter(x=data[time_column], y=data[target_column], 
                      mode='lines', name='Time Series'),
            row=1, col=1
        )
        
        # 2. Distribution plot
        fig.add_trace(
            go.Histogram(x=target_series, nbinsx=30, name='Distribution'),
            row=1, col=2
        )
        
        # 3. Autocorrelation
        if len(target_series) > 10:
            lags = range(1, min(21, len(target_series)//2))
            autocorrs = [target_series.autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(
                go.Bar(x=list(lags), y=autocorrs, name='Autocorrelation'),
                row=2, col=1
            )
        
        # 4. Outliers
        if 'outliers' in debug_report['data_quality_issues']:
            outlier_info = debug_report['data_quality_issues']['outliers']
            if 'bounds' in outlier_info:
                bounds = outlier_info['bounds']
                fig.add_hline(y=bounds['upper'], line_dash="dash", 
                             line_color="red", row=3, col=1)
                fig.add_hline(y=bounds['lower'], line_dash="dash", 
                             line_color="red", row=3, col=1)
        
        fig.add_trace(
            go.Scatter(x=data[time_column], y=data[target_column], 
                      mode='markers', name='Data Points'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Time Series Debugging Dashboard",
            showlegend=True
        )
        
        # Save plot
        plot_path = f"{self.config.debug_output_dir}debug_dashboard.html"
        fig.write_html(plot_path)
        self.logger.info(f"Debug visualizations saved to {plot_path}")


class TimeSeriesTestSuite:
    """
    Automated test suite for time series forecasting pipelines.
    Based on TimeGym principles for test-driven development.
    """
    
    def __init__(self):
        self.test_results = {}
        
    def test_data_integrity(self, data: pd.DataFrame, target_column: str) -> Dict[str, bool]:
        """Test basic data integrity."""
        
        tests = {
            'no_missing_target': data[target_column].isnull().sum() == 0,
            'numeric_target': pd.api.types.is_numeric_dtype(data[target_column]),
            'sufficient_data': len(data) >= 30,
            'no_infinite_values': not np.isinf(data[target_column]).any(),
            'no_constant_values': data[target_column].nunique() > 1
        }
        
        return tests
    
    def test_temporal_consistency(self, data: pd.DataFrame, time_column: str) -> Dict[str, bool]:
        """Test temporal consistency of the data."""
        
        tests = {
            'datetime_index': pd.api.types.is_datetime64_any_dtype(data[time_column]),
            'monotonic_time': data[time_column].is_monotonic_increasing,
            'no_duplicate_times': not data[time_column].duplicated().any(),
            'regular_frequency': len(data[time_column].diff().dropna().unique()) <= 2
        }
        
        return tests
    
    def test_model_assumptions(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, bool]:
        """Test if model satisfies basic assumptions."""
        
        tests = {}
        
        try:
            # Test if model can fit and predict
            model.fit(X, y)
            predictions = model.predict(X)
            
            tests['can_fit'] = True
            tests['can_predict'] = True
            tests['prediction_shape_correct'] = len(predictions) == len(y)
            tests['no_nan_predictions'] = not np.isnan(predictions).any()
            
        except Exception as e:
            tests['can_fit'] = False
            tests['error'] = str(e)
        
        return tests
    
    def run_full_test_suite(self, 
                           data: pd.DataFrame,
                           target_column: str,
                           time_column: str,
                           model=None) -> Dict[str, Dict[str, bool]]:
        """Run complete test suite."""
        
        results = {
            'data_integrity': self.test_data_integrity(data, target_column),
            'temporal_consistency': self.test_temporal_consistency(data, time_column)
        }
        
        if model is not None:
            # Create simple features for model testing
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[target_column].values
            results['model_assumptions'] = self.test_model_assumptions(model, X, y)
        
        return results


# Example usage and demonstration
def demonstrate_debugging_toolkit():
    """Demonstrate the time series debugging toolkit."""
    
    print("🔍 Time Series Debugging Toolkit Demonstration")
    print("=" * 60)
    
    # Generate problematic synthetic data
    np.random.seed(42)
    
    # Create data with various issues
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    # Base trend with seasonality
    trend = np.linspace(100, 150, 200)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(200) / 30)
    noise = np.random.normal(0, 5, 200)
    
    # Add some issues
    values = trend + seasonal + noise
    
    # Introduce missing values
    missing_indices = np.random.choice(200, 10, replace=False)
    values[missing_indices] = np.nan
    
    # Introduce outliers
    outlier_indices = np.random.choice(200, 5, replace=False)
    values[outlier_indices] = values[outlier_indices] * 3
    
    # Create suspicious feature (look-ahead bias)
    suspicious_feature = values * 0.98 + np.random.normal(0, 1, 200)
    
    # Create DataFrame
    problematic_data = pd.DataFrame({
        'date': dates,
        'target': values,
        'suspicious_feature': suspicious_feature,
        'normal_feature': np.random.normal(50, 10, 200)
    })
    
    # Initialize debugger
    config = DebuggingConfig()
    debugger = TimeSeriesDebugger(config)
    
    # Run comprehensive debugging
    print("\n🔍 Running comprehensive debugging analysis...")
    debug_report = debugger.run_comprehensive_debug(
        data=problematic_data,
        target_column='target',
        time_column='date'
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("🚨 DEBUGGING RESULTS")
    print("=" * 60)
    
    for section, results in debug_report.items():
        if section == 'recommendations':
            continue
            
        if isinstance(results, dict) and 'severity' in results:
            severity_emoji = {
                'low': '✅',
                'medium': '⚠️',
                'high': '🚨'
            }
            
            print(f"\n{severity_emoji.get(results['severity'], '❓')} {section.upper().replace('_', ' ')}")
            print(f"   Severity: {results['severity']}")
            
            # Display key findings
            if section == 'data_quality_issues':
                if 'missing_values' in results and results['missing_values']:
                    if 'target_missing' in results['missing_values']:
                        print(f"   Missing target values: {results['missing_values']['target_missing']}")
                
                if 'outliers' in results and results['outliers']:
                    print(f"   Outliers detected: {results['outliers']['count']} ({results['outliers']['percentage']:.1f}%)")
            
            elif section == 'stationarity_issues':
                if results.get('non_stationary'):
                    print(f"   Non-stationary series detected")
                if results.get('trend_detected'):
                    print(f"   Trend detected: {results.get('trend_direction', 'unknown')}")
            
            elif section == 'seasonality_issues':
                if results.get('seasonality_ignored'):
                    print(f"   Seasonality patterns found")
            
            elif section == 'modeling_pitfalls':
                if results.get('look_ahead_bias_risk'):
                    print(f"   Look-ahead bias risk detected")
                if results.get('insufficient_data'):
                    print(f"   Insufficient data: {results.get('data_size', 'unknown')} samples")
    
    # Display recommendations
    if debug_report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(debug_report['recommendations'])})")
        print("-" * 40)
        for i, rec in enumerate(debug_report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Run test suite
    print(f"\n🧪 Running automated test suite...")
    test_suite = TimeSeriesTestSuite()
    test_results = test_suite.run_full_test_suite(
        data=problematic_data,
        target_column='target',
        time_column='date'
    )
    
    print("\n" + "=" * 60)
    print("🧪 TEST RESULTS")
    print("=" * 60)
    
    for test_category, tests in test_results.items():
        print(f"\n{test_category.upper().replace('_', ' ')}:")
        for test_name, passed in tests.items():
            if test_name == 'error':
                continue
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    print(f"\n🏁 Debugging and testing completed!")
    print(f"📄 Detailed report saved to: {config.debug_output_dir}")
    print(f"📊 Debug visualizations saved to: {config.debug_output_dir}debug_dashboard.html")
    
    return debug_report, test_results


if __name__ == "__main__":
    # Run debugging demonstration
    debug_report, test_results = demonstrate_debugging_toolkit()
    
    print("\n" + "="*60)
    print("Key takeaways:")
    print("1. Always validate data quality before modeling")
    print("2. Check for stationarity and seasonality")
    print("3. Avoid look-ahead bias and data leakage")
    print("4. Use proper temporal validation")
    print("5. Monitor prediction quality with residual analysis")
