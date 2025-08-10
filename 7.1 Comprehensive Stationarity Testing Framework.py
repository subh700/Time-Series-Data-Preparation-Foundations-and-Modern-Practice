def comprehensive_stationarity_assessment(data, max_lags=24, seasonal_period=None):
    """
    Multi-faceted stationarity assessment using statistical tests and visual methods.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import het_white
    from scipy import stats
    
    results = {
        'statistical_tests': {},
        'visual_indicators': {},
        'recommendations': []
    }
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(data.dropna(), maxlag=max_lags, regression='ct')
    results['statistical_tests']['adf'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'conclusion': 'stationary' if adf_result[1] < 0.05 else 'non_stationary'
    }
    
    # KPSS test (null hypothesis: stationary)
    kpss_result = kpss(data.dropna(), regression='ct', nlags='auto')
    results['statistical_tests']['kpss'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'conclusion': 'stationary' if kpss_result[1] > 0.05 else 'non_stationary'
    }
    
    # Variance stability test (rolling window approach)
    window_size = min(len(data) // 4, 100)
    rolling_var = data.rolling(window=window_size).var().dropna()
    
    if len(rolling_var) > 10:
        # Test for constant variance using Levene's test
        split_point = len(rolling_var) // 2
        var_test = stats.levene(
            rolling_var.iloc[:split_point], 
            rolling_var.iloc[split_point:]
        )
        results['statistical_tests']['variance_stability'] = {
            'statistic': var_test.statistic,
            'p_value': var_test.pvalue,
            'conclusion': 'stable_variance' if var_test.pvalue > 0.05 else 'heteroscedastic'
        }
    
    # Mean stability assessment
    mean_changes = detect_mean_changes(data, min_segment_length=window_size)
    results['statistical_tests']['mean_stability'] = {
        'change_points': mean_changes,
        'conclusion': 'stable_mean' if len(mean_changes) == 0 else 'changing_mean'
    }
    
    # Seasonal unit root test (if seasonal period specified)
    if seasonal_period:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            seasonal_component = seasonal_decompose(
                data.dropna(), 
                period=seasonal_period, 
                model='additive'
            ).seasonal
            
            # Test stationarity of seasonal component
            seasonal_adf = adfuller(seasonal_component.dropna())
            results['statistical_tests']['seasonal_stationarity'] = {
                'p_value': seasonal_adf[1],
                'conclusion': 'seasonal_stationary' if seasonal_adf[1] < 0.05 else 'seasonal_non_stationary'
            }
        except:
            results['statistical_tests']['seasonal_stationarity'] = {
                'conclusion': 'test_failed'
            }
    
    # Generate recommendations
    adf_stationary = results['statistical_tests']['adf']['conclusion'] == 'stationary'
    kpss_stationary = results['statistical_tests']['kpss']['conclusion'] == 'stationary'
    
    if adf_stationary and kpss_stationary:
        results['overall_conclusion'] = 'stationary'
        results['recommendations'].append('Data appears stationary - no transformation needed')
    elif not adf_stationary and not kpss_stationary:
        results['overall_conclusion'] = 'non_stationary'
        results['recommendations'].append('Data is non-stationary - consider differencing or detrending')
    else:
        results['overall_conclusion'] = 'mixed_signals'
        results['recommendations'].append('Mixed test results - investigate trend vs. stochastic non-stationarity')
    
    # Specific recommendations based on test results
    if 'variance_stability' in results['statistical_tests']:
        if results['statistical_tests']['variance_stability']['conclusion'] == 'heteroscedastic':
            results['recommendations'].append('Consider variance-stabilizing transformation (log, Box-Cox)')
    
    if 'mean_stability' in results['statistical_tests']:
        if results['statistical_tests']['mean_stability']['conclusion'] == 'changing_mean':
            results['recommendations'].append('Detected mean shifts - consider structural break modeling')
    
    return results

def detect_mean_changes(data, min_segment_length=30, significance_level=0.05):
    """Detect structural breaks in mean using recursive segmentation."""
    from scipy import stats
    import numpy as np
    
    def test_break(series, break_point):
        """Test for significant mean difference at break point."""
        before = series[:break_point]
        after = series[break_point:]
        
        if len(before) < min_segment_length or len(after) < min_segment_length:
            return False, 1.0
        
        t_stat, p_val = stats.ttest_ind(before, after)
        return p_val < significance_level, p_val
    
    def find_breaks_recursive(series, start_idx=0):
        """Recursively find break points."""
        if len(series) < 2 * min_segment_length:
            return []
        
        breaks = []
        best_break = None
        best_p_val = 1.0
        
        # Test potential break points
        for i in range(min_segment_length, len(series) - min_segment_length):
            is_break, p_val = test_break(series, i)
            if is_break and p_val < best_p_val:
                best_break = i
                best_p_val = p_val
        
        if best_break is not None:
            breaks.append(start_idx + best_break)
            # Recursively search segments
            left_breaks = find_breaks_recursive(series[:best_break], start_idx)
            right_breaks = find_breaks_recursive(series[best_break:], start_idx + best_break)
            breaks.extend(left_breaks)
            breaks.extend(right_breaks)
        
        return sorted(breaks)
    
    return find_breaks_recursive(data.values)
