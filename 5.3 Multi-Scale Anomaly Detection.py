def multiscale_sensor_outliers(data, scales=[1, 5, 15, 60], method='robust_zscore'):
    """
    Multi-scale outlier detection for industrial sensor data.
    Detects anomalies at different temporal resolutions.
    """
    outlier_scores = np.zeros(len(data))
    
    for scale in scales:
        # Downsample data to different scales
        if scale == 1:
            scaled_data = data
        else:
            # Rolling window downsampling
            scaled_data = data.rolling(window=scale, center=True).mean()
        
        # Detect outliers at this scale
        if method == 'robust_zscore':
            # Use median and MAD for robustness
            median = scaled_data.median()
            mad = np.median(np.abs(scaled_data - median))
            if mad == 0:
                continue
            modified_z_scores = 0.6745 * (scaled_data - median) / mad
            scale_outliers = np.abs(modified_z_scores) > 3.5
        
        # Combine scores across scales
        outlier_scores += scale_outliers.astype(float) / scale
    
    # Normalize combined scores
    outlier_scores = outlier_scores / len(scales)
    
    return {
        'combined_scores': outlier_scores,
        'outlier_mask': outlier_scores > 0.5,  # Threshold for final decision
        'scales_used': scales
    }
