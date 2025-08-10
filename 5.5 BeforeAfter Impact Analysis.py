def assess_treatment_effectiveness(original_data, treated_data, outlier_indices):
    """
    Comprehensive assessment of outlier treatment effectiveness.
    """
    from scipy import stats
    
    # Statistical property preservation
    orig_stats = {
        'mean': original_data.mean(),
        'std': original_data.std(),
        'skewness': stats.skew(original_data),
        'kurtosis': stats.kurtosis(original_data)
    }
    
    treated_stats = {
        'mean': treated_data.mean(),
        'std': treated_data.std(),
        'skewness': stats.skew(treated_data),
        'kurtosis': stats.kurtosis(treated_data)
    }
    
    # Temporal structure preservation
    orig_acf = acf(original_data, nlags=24, missing='drop')
    treated_acf = acf(treated_data, nlags=24, missing='drop')
    acf_similarity = np.corrcoef(orig_acf[1:], treated_acf[1:])[0, 1]
    
    # Forecast accuracy impact (if ground truth available)
    forecast_impact = assess_forecast_impact(original_data, treated_data)
    
    return {
        'statistical_changes': {
            'mean_change': abs(treated_stats['mean'] - orig_stats['mean']) / orig_stats['std'],
            'std_change': abs(treated_stats['std'] - orig_stats['std']) / orig_stats['std'],
            'distribution_changes': {
                'skewness_change': treated_stats['skewness'] - orig_stats['skewness'],
                'kurtosis_change': treated_stats['kurtosis'] - orig_stats['kurtosis']
            }
        },
        'temporal_preservation': {
            'autocorrelation_similarity': acf_similarity
        },
        'forecast_impact': forecast_impact,
        'treatment_summary': {
            'outliers_treated': len(outlier_indices),
            'treatment_rate': len(outlier_indices) / len(original_data)
        }
    }
