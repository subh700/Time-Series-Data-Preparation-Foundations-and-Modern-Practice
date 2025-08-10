def evaluate_transformation_quality(original_data, transformed_data, transformation_method):
    """
    Comprehensive evaluation of transformation quality across multiple dimensions.
    """
    
    evaluation_results = {
        'stationarity_improvement': {},
        'distributional_properties': {},
        'temporal_structure_preservation': {},
        'interpretability_impact': {},
        'overall_score': 0
    }
    
    # 1. Stationarity Assessment
    orig_stationarity = comprehensive_stationarity_assessment(original_data)
    trans_stationarity = comprehensive_stationarity_assessment(transformed_data)
    
    evaluation_results['stationarity_improvement'] = {
        'original_adf_pvalue': orig_stationarity['statistical_tests']['adf']['p_value'],
        'transformed_adf_pvalue': trans_stationarity['statistical_tests']['adf']['p_value'],
        'stationarity_achieved': trans_stationarity['overall_conclusion'] == 'stationary',
        'improvement_score': calculate_stationarity_improvement_score(
            orig_stationarity, trans_stationarity
        )
    }
    
    # 2. Distributional Properties
    from scipy import stats
    
    orig_skew = stats.skew(original_data.dropna())
    trans_skew = stats.skew(transformed_data.dropna())
    orig_kurtosis = stats.kurtosis(original_data.dropna())
    trans_kurtosis = stats.kurtosis(transformed_data.dropna())
    
    # Normality tests
    orig_normality = stats.jarque_bera(original_data.dropna())
    trans_normality = stats.jarque_bera(transformed_data.dropna())
    
    evaluation_results['distributional_properties'] = {
        'skewness_change': abs(trans_skew) - abs(orig_skew),  # Negative is better
        'kurtosis_change': abs(trans_kurtosis - 3) - abs(orig_kurtosis - 3),  # Closer to 3 is better
        'normality_improvement': orig_normality[1] - trans_normality[1],  # Higher p-value is better
        'distribution_score': calculate_distributional_score(
            orig_skew, trans_skew, orig_kurtosis, trans_kurtosis,
            orig_normality[1], trans_normality[1]
        )
    }
    
    # 3. Temporal Structure Preservation
    from statsmodels.tsa.stattools import acf
    
    # Calculate autocorrelation functions
    orig_acf = acf(original_data.dropna(), nlags=min(24, len(original_data)//4), missing='drop')
    trans_acf = acf(transformed_data.dropna(), nlags=min(24, len(transformed_data)//4), missing='drop')
    
    # Correlation between ACFs (measure of structure preservation)
    min_len = min(len(orig_acf), len(trans_acf))
    acf_correlation = np.corrcoef(orig_acf[:min_len], trans_acf[:min_len])[0, 1]
    
    # Spectral density preservation (frequency domain analysis)
    orig_psd = np.abs(np.fft.fft(original_data.dropna().values))**2
    trans_psd = np.abs(np.fft.fft(transformed_data.dropna().values))**2
    
    # Normalize PSDs for comparison
    orig_psd_norm = orig_psd / np.sum(orig_psd)
    trans_psd_norm = trans_psd / np.sum(trans_psd)
    
    spectral_similarity = calculate_spectral_similarity(orig_psd_norm, trans_psd_norm)
    
    evaluation_results['temporal_structure_preservation'] = {
        'acf_correlation': acf_correlation if not np.isnan(acf_correlation) else 0,
        'spectral_similarity': spectral_similarity,
        'structure_preservation_score': (acf_correlation + spectral_similarity) / 2 if not np.isnan(acf_correlation) else spectral_similarity / 2
    }
    
    # 4. Interpretability Impact
    interpretability_score = assess_interpretability_impact(transformation_method, original_data, transformed_data)
    evaluation_results['interpretability_impact'] = interpretability_score
    
    # 5. Calculate Overall Score
    weights = {
        'stationarity': 0.4,
        'distribution': 0.2,
        'structure_preservation': 0.3,
        'interpretability': 0.1
    }
    
    overall_score = (
        weights['stationarity'] * evaluation_results['stationarity_improvement']['improvement_score'] +
        weights['distribution'] * evaluation_results['distributional_properties']['distribution_score'] +
        weights['structure_preservation'] * evaluation_results['temporal_structure_preservation']['structure_preservation_score'] +
        weights['interpretability'] * evaluation_results['interpretability_impact']['score']
    )
    
    evaluation_results['overall_score'] = overall_score
    
    return evaluation_results

def calculate_stationarity_improvement_score(orig_results, trans_results):
    """Calculate improvement in stationarity (0-1 scale)."""
    
    orig_adf_p = orig_results['statistical_tests']['adf']['p_value']
    trans_adf_p = trans_results['statistical_tests']['adf']['p_value']
    
    # Base score from achieving stationarity
    if trans_results['overall_conclusion'] == 'stationary':
        base_score = 0.7
    else:
        base_score = 0.3
    
    # Bonus for p-value improvement
    p_value_improvement = max(0, orig_adf_p - trans_adf_p) / max(orig_adf_p, 0.01)
    p_value_bonus = min(0.3, p_value_improvement * 0.3)
    
    return base_score + p_value_bonus

def calculate_distributional_score(orig_skew, trans_skew, orig_kurt, trans_kurt, orig_norm_p, trans_norm_p):
    """Calculate distributional improvement score (0-1 scale)."""
    
    # Skewness improvement (closer to 0 is better)
    skew_improvement = max(0, abs(orig_skew) - abs(trans_skew)) / max(abs(orig_skew), 1)
    skew_score = min(0.33, skew_improvement * 0.33)
    
    # Kurtosis improvement (closer to 3 is better for normal distribution)
    kurt_improvement = max(0, abs(orig_kurt - 3) - abs(trans_kurt - 3)) / max(abs(orig_kurt - 3), 1)
    kurt_score = min(0.33, kurt_improvement * 0.33)
    
    # Normality improvement
    norm_improvement = max(0, trans_norm_p - orig_norm_p)
    norm_score = min(0.34, norm_improvement * 0.34)
    
    return skew_score + kurt_score + norm_score

def calculate_spectral_similarity(psd1, psd2):
    """Calculate spectral similarity between two power spectral densities."""
    
    # Use Jensen-Shannon divergence for similarity
    from scipy.spatial.distance import jensenshannon
    
    # Ensure same length
    min_len = min(len(psd1), len(psd2))
    psd1_trimmed = psd1[:min_len]
    psd2_trimmed = psd2[:min_len]
    
    # Calculate JS divergence (0 = identical, 1 = completely different)
    js_divergence = jensenshannon(psd1_trimmed, psd2_trimmed)
    
    # Convert to similarity score (1 = identical, 0 = completely different)
    similarity = 1 - js_divergence
    
    return similarity if not np.isnan(similarity) else 0

def assess_interpretability_impact(transformation_method, original_data, transformed_data):
    """Assess impact on interpretability based on transformation type."""
    
    interpretability_scores = {
        'log': {'score': 0.8, 'reason': 'Logarithmic transformation maintains intuitive interpretation'},
        'sqrt': {'score': 0.7, 'reason': 'Square root transformation somewhat interpretable'},
        'box_cox': {'score': 0.6, 'reason': 'Box-Cox transformation less intuitive but systematic'},
        'differencing': {'score': 0.9, 'reason': 'Differencing represents change, highly interpretable'},
        'fractional_differencing': {'score': 0.5, 'reason': 'Fractional differencing difficult to interpret'},
        'emd': {'score': 0.4, 'reason': 'EMD components not directly interpretable'},
        'wavelet': {'score': 0.3, 'reason': 'Wavelet coefficients require specialized knowledge'},
        'power': {'score': 0.5, 'reason': 'Power transformations moderately interpretable'}
    }
    
    # Determine transformation type
    if 'log' in transformation_method.lower():
        method_key = 'log'
    elif 'box' in transformation_method.lower() and 'cox' in transformation_method.lower():
        method_key = 'box_cox'
    elif 'diff' in transformation_method.lower() and 'fractional' in transformation_method.lower():
        method_key = 'fractional_differencing'
    elif 'diff' in transformation_method.lower():
        method_key = 'differencing'
    elif 'emd' in transformation_method.lower():
        method_key = 'emd'
    elif 'wavelet' in transformation_method.lower():
        method_key = 'wavelet'
    elif 'sqrt' in transformation_method.lower():
        method_key = 'sqrt'
    else:
        method_key = 'power'
    
    base_score = interpretability_scores.get(method_key, {'score': 0.5, 'reason': 'Unknown transformation type'})
    
    # Adjust score based on magnitude of transformation
    original_range = original_data.max() - original_data.min()
    transformed_range = transformed_data.max() - transformed_data.min()
    
    if original_range > 0:
        range_ratio = transformed_range / original_range
        
        # Penalize extreme range changes
        if range_ratio > 10 or range_ratio < 0.1:
            base_score['score'] *= 0.8
        elif range_ratio > 5 or range_ratio < 0.2:
            base_score['score'] *= 0.9
    
    return base_score
