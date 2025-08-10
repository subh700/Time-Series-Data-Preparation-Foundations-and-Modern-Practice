def evaluate_temporal_coherence(original, imputed):
    """Assess how well imputation preserves temporal structure."""
    
    # Autocorrelation preservation
    orig_acf = acf(original.dropna(), nlags=24)
    imp_acf = acf(imputed, nlags=24)
    acf_similarity = np.corrcoef(orig_acf[1:], imp_acf[1:])[0,1]
    
    # Trend preservation
    orig_trend = np.polyfit(range(len(original.dropna())), original.dropna(), deg=1)[0]
    imp_trend = np.polyfit(range(len(imputed)), imputed, deg=1)[0]
    trend_similarity = 1 - abs(orig_trend - imp_trend) / abs(orig_trend)
    
    # Seasonal pattern preservation
    if len(imputed) >= 24:  # Ensure sufficient data for seasonal analysis
        orig_seasonal = seasonal_decompose(original.dropna(), period=24).seasonal
        imp_seasonal = seasonal_decompose(imputed, period=24).seasonal
        seasonal_similarity = np.corrcoef(orig_seasonal, imp_seasonal)[0,1]
    else:
        seasonal_similarity = np.nan
        
    return {
        'autocorrelation_preservation': acf_similarity,
        'trend_preservation': trend_similarity,
        'seasonal_preservation': seasonal_similarity
    }
