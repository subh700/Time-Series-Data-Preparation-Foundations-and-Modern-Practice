def isolation_forest_ts_detection(data, contamination=0.05, window_features=True):
    """
    Isolation Forest adapted for time series with temporal features.
    """
    from sklearn.ensemble import IsolationForest
    
    # Create temporal features
    features = create_temporal_features(data, include_lags=True, window_stats=window_features)
    
    # Initialize and fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        behaviour='new'  # Use new behavior for consistency
    )
    
    # Detect outliers (-1 for outliers, 1 for normal)
    outlier_labels = iso_forest.fit_predict(features)
    
    # Calculate anomaly scores
    anomaly_scores = iso_forest.score_samples(features)
    
    return {
        'outlier_mask': outlier_labels == -1,
        'anomaly_scores': anomaly_scores,
        'features_used': features.columns.tolist()
    }

def create_temporal_features(data, include_lags=True, window_stats=True):
    """Create comprehensive feature set for ML-based outlier detection."""
    features = pd.DataFrame(index=data.index)
    
    # Basic temporal features
    features['hour'] = data.index.hour
    features['day_of_week'] = data.index.dayofweek
    features['month'] = data.index.month
    features['value'] = data.values
    
    if include_lags:
        # Lag features
        for lag in [1, 2, 3, 7, 24]:
            features[f'lag_{lag}'] = data.shift(lag)
    
    if window_stats:
        # Rolling window statistics
        for window in [7, 24, 168]:  # Week, day, hour patterns
            features[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            features[f'rolling_std_{window}'] = data.rolling(window=window).std()
            features[f'rolling_min_{window}'] = data.rolling(window=window).min()
            features[f'rolling_max_{window}'] = data.rolling(window=window).max()
    
    # Drop rows with NaN values (from lags and rolling stats)
    features = features.dropna()
    
    return features
