def adaptive_winsorization(data, outlier_mask, percentile_range=(5, 95)):
    """
    Adaptive winsorization that adjusts capping values based on local context.
    """
    treated_data = data.copy()
    
    for idx in np.where(outlier_mask)[0]:
        # Define local window for percentile calculation
        window_start = max(0, idx - 50)
        window_end = min(len(data), idx + 50)
        local_window = data.iloc[window_start:window_end]
        
        # Calculate adaptive percentiles
        lower_bound = np.percentile(local_window, percentile_range[0])
        upper_bound = np.percentile(local_window, percentile_range[1])
        
        # Apply winsorization
        if data.iloc[idx] < lower_bound:
            treated_data.iloc[idx] = lower_bound
        elif data.iloc[idx] > upper_bound:
            treated_data.iloc[idx] = upper_bound
    
    return treated_data
