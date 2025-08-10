def stl_outlier_detection(data, seasonal_period=24, alpha=0.05, max_outliers=None):
    """
    STL decomposition followed by Generalized Extreme Studentized Deviate test.
    Based on Twitter's anomaly detection approach.
    """
    from statsmodels.tsa.seasonal import STL
    from scipy import stats
    
    # Perform STL decomposition
    stl = STL(data, seasonal=seasonal_period)
    decomposition = stl.fit()
    
    # Extract residuals for outlier detection
    residuals = decomposition.resid
    
    # Apply Generalized ESD test on residuals
    outliers = generalized_esd_test(residuals, alpha=alpha, max_outliers=max_outliers)
    
    return {
        'outlier_indices': outliers,
        'decomposition': decomposition,
        'outlier_scores': np.abs(residuals) / np.std(residuals)
    }

def generalized_esd_test(data, alpha=0.05, max_outliers=None):
    """Generalized Extreme Studentized Deviate test for outliers."""
    if max_outliers is None:
        max_outliers = int(0.1 * len(data))  # Max 10% outliers
    
    outliers = []
    data_copy = data.copy()
    
    for i in range(max_outliers):
        # Calculate test statistic
        mean_val = np.mean(data_copy)
        std_val = np.std(data_copy, ddof=1)
        
        if std_val == 0:
            break
            
        test_stats = np.abs(data_copy - mean_val) / std_val
        max_idx = np.argmax(test_stats)
        max_stat = test_stats[max_idx]
        
        # Critical value calculation
        n = len(data_copy)
        t_val = stats.t.ppf(1 - alpha / (2 * (n - i)), n - i - 2)
        critical_val = ((n - i - 1) * t_val) / np.sqrt((n - i - 2 + t_val**2) * (n - i))
        
        if max_stat > critical_val:
            outliers.append(max_idx)
            data_copy = np.delete(data_copy, max_idx)
        else:
            break
    
    return outliers
