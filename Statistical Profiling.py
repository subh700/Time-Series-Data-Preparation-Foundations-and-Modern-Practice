def profile_time_series(data, timestamp_col, value_col):
    """Comprehensive time series profiling."""
    profile = {
        'temporal_span': data[timestamp_col].max() - data[timestamp_col].min(),
        'observation_count': len(data),
        'missing_values': data[value_col].isnull().sum(),
        'duplicate_timestamps': data[timestamp_col].duplicated().sum(),
        'value_range': (data[value_col].min(), data[value_col].max()),
        'temporal_resolution': infer_frequency(data[timestamp_col]),
        'stationarity_test': adf_test(data[value_col].dropna()),
        'seasonality_strength': seasonal_strength(data, value_col),
        'outlier_count': detect_outliers(data[value_col]).sum()
    }
    return profile
