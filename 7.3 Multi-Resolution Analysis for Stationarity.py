def wavelet_stationarity_transformation(data, wavelet='db4', levels=None, reconstruction_strategy='adaptive'):
    """
    Use discrete wavelet transform for stationarity transformation.
    Particularly effective for data with time-varying characteristics.
    """
    import pywt
    
    # Determine decomposition levels
    if levels is None:
        levels = min(6, int(np.log2(len(data))))
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data.values, wavelet, level=levels)
    
    # Analyze stationarity of each level
    level_analysis = {}
    
    # Approximation coefficients (low frequency)
    approx_coeffs = coeffs[0]
    if len(approx_coeffs) > 10:
        adf_result = adfuller(approx_coeffs, autolag='AIC')
        level_analysis['approximation'] = {
            'coefficients': approx_coeffs,
            'adf_p_value': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'frequency_content': 'low'
        }
    
    # Detail coefficients (high frequency)
    for i, detail_coeffs in enumerate(coeffs[1:], 1):
        if len(detail_coeffs) > 10:
            adf_result = adfuller(detail_coeffs, autolag='AIC')
            level_analysis[f'detail_{i}'] = {
                'coefficients': detail_coeffs,
                'adf_p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'frequency_content': f'band_{i}'
            }
    
    # Apply reconstruction strategy
    if reconstruction_strategy == 'adaptive':
        # Reconstruct using stationary components and differenced non-stationary ones
        modified_coeffs = coeffs.copy()
        
        # Handle approximation coefficients
        if 'approximation' in level_analysis:
            if not level_analysis['approximation']['is_stationary']:
                # Apply differencing to approximation coefficients
                approx_diff = np.diff(coeffs[0])
                # Pad to maintain length
                modified_coeffs[0] = np.concatenate([[coeffs[0][0]], approx_diff])
        
        # Detail coefficients are typically already stationary
        # (high-frequency components are often stationary by nature)
        
        # Reconstruct signal
        reconstructed = pywt.waverec(modified_coeffs, wavelet)
        
        # Trim to original length if necessary
        if len(reconstructed) > len(data):
            reconstructed = reconstructed[:len(data)]
        elif len(reconstructed) < len(data):
            # Pad with last value if needed
            reconstructed = np.concatenate([
                reconstructed, 
                np.full(len(data) - len(reconstructed), reconstructed[-1])
            ])
        
        transformed_series = pd.Series(reconstructed, index=data.index)
        
    elif reconstruction_strategy == 'high_frequency_only':
        # Use only detail coefficients (high-frequency components)
        detail_only_coeffs = [np.zeros_like(coeffs[0])] + coeffs[1:]
        reconstructed = pywt.waverec(detail_only_coeffs, wavelet)
        
        # Trim/pad to match original length
        if len(reconstructed) != len(data):
            if len(reconstructed) > len(data):
                reconstructed = reconstructed[:len(data)]
            else:
                reconstructed = np.concatenate([
                    reconstructed, 
                    np.zeros(len(data) - len(reconstructed))
                ])
        
        transformed_series = pd.Series(reconstructed, index=data.index)
        
    elif reconstruction_strategy == 'denoised':
        # Apply soft thresholding for denoising
        threshold = np.std(data) * np.sqrt(2 * np.log(len(data)))
        
        modified_coeffs = coeffs.copy()
        for i in range(1, len(modified_coeffs)):  # Skip approximation coefficients
            modified_coeffs[i] = pywt.threshold(
                modified_coeffs[i], 
                threshold / (2**i),  # Scale threshold by level
                mode='soft'
            )
        
        reconstructed = pywt.waverec(modified_coeffs, wavelet)
        
        # Trim/pad to match original length
        if len(reconstructed) != len(data):
            if len(reconstructed) > len(data):
                reconstructed = reconstructed[:len(data)]
            else:
                reconstructed = np.concatenate([
                    reconstructed, 
                    np.full(len(data) - len(reconstructed), reconstructed[-1])
                ])
        
        transformed_series = pd.Series(reconstructed, index=data.index)
    
    return {
        'transformed_series': transformed_series,
        'wavelet_coefficients': coeffs,
        'level_analysis': level_analysis,
        'reconstruction_strategy': reconstruction_strategy,
        'wavelet_used': wavelet,
        'decomposition_levels': levels
    }
