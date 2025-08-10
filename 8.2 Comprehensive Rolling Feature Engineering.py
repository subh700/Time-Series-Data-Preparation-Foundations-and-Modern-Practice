class MultiScaleRollingFeatureEngineer:
    """
    Create rolling window features at multiple time scales with
    adaptive window sizing based on data characteristics.
    """
    
    def __init__(self, base_windows=None, include_expanding=True, robust_statistics=True):
        self.base_windows = base_windows or [3, 7, 14, 30, 90]
        self.include_expanding = include_expanding
        self.robust_statistics = robust_statistics
        
    def create_rolling_features(self, data, target_cols, timestamp_col=None, 
                              seasonal_period=None):
        """
        Create comprehensive rolling window features with multiple scales.
        """
        feature_df = data.copy()
        
        # Adaptive window sizing based on data frequency
        if timestamp_col and seasonal_period:
            adaptive_windows = self._get_adaptive_windows(
                data[timestamp_col], seasonal_period
            )
        else:
            adaptive_windows = self.base_windows
        
        for col in target_cols:
            if col not in data.columns:
                continue
                
            # Standard rolling statistics
            for window in adaptive_windows:
                if window >= len(data):
                    continue
                    
                # Basic statistics
                feature_df[f'{col}_rolling_mean_{window}'] = (
                    data[col].rolling(window=window, min_periods=1).mean()
                )
                feature_df[f'{col}_rolling_std_{window}'] = (
                    data[col].rolling(window=window, min_periods=1).std()
                )
                feature_df[f'{col}_rolling_min_{window}'] = (
                    data[col].rolling(window=window, min_periods=1).min()
                )
                feature_df[f'{col}_rolling_max_{window}'] = (
                    data[col].rolling(window=window, min_periods=1).max()
                )
                
                # Advanced statistics
                feature_df[f'{col}_rolling_skew_{window}'] = (
                    data[col].rolling(window=window, min_periods=3).skew()
                )
                feature_df[f'{col}_rolling_kurt_{window}'] = (
                    data[col].rolling(window=window, min_periods=4).kurt()
                )
                
                # Percentiles
                for q in [0.1, 0.25, 0.75, 0.9]:
                    feature_df[f'{col}_rolling_q{int(q*100)}_{window}'] = (
                        data[col].rolling(window=window, min_periods=1).quantile(q)
                    )
                
                # Range and IQR
                feature_df[f'{col}_rolling_range_{window}'] = (
                    feature_df[f'{col}_rolling_max_{window}'] - 
                    feature_df[f'{col}_rolling_min_{window}']
                )
                feature_df[f'{col}_rolling_iqr_{window}'] = (
                    feature_df[f'{col}_rolling_q75_{window}'] - 
                    feature_df[f'{col}_rolling_q25_{window}']
                )
                
                # Robust statistics (if enabled)
                if self.robust_statistics:
                    feature_df[f'{col}_rolling_median_{window}'] = (
                        data[col].rolling(window=window, min_periods=1).median()
                    )
                    
                    # Median Absolute Deviation (MAD)
                    rolling_median = feature_df[f'{col}_rolling_median_{window}']
                    feature_df[f'{col}_rolling_mad_{window}'] = (
                        data[col].rolling(window=window, min_periods=1)
                        .apply(lambda x: np.median(np.abs(x - np.median(x))))
                    )
                
                # Momentum features
                feature_df[f'{col}_rolling_momentum_{window}'] = (
                    data[col] - feature_df[f'{col}_rolling_mean_{window}']
                )
                
                feature_df[f'{col}_rolling_zscore_{window}'] = (
                    feature_df[f'{col}_rolling_momentum_{window}'] / 
                    feature_df[f'{col}_rolling_std_{window}']
                )
                
                # Rate of change
                feature_df[f'{col}_rolling_roc_{window}'] = (
                    (data[col] - data[col].shift(window)) / data[col].shift(window)
                )
        
        # Expanding window features (if enabled)
        if self.include_expanding:
            feature_df = self._add_expanding_features(feature_df, target_cols)
        
        return feature_df
    
    def _get_adaptive_windows(self, timestamps, seasonal_period):
        """Adapt window sizes based on data frequency and seasonality."""
        # Infer data frequency
        freq_seconds = pd.infer_freq(timestamps)
        if freq_seconds:
            freq_timedelta = pd.Timedelta(freq_seconds).seconds
        else:
            # Fallback: estimate from differences
            time_diffs = timestamps.diff().dropna()
            freq_timedelta = time_diffs.median().seconds
        
        # Calculate windows in terms of observations
        adaptive_windows = []
        
        # Short-term: 1-2 periods
        if seasonal_period:
            adaptive_windows.extend([
                max(1, seasonal_period // 4),
                max(2, seasonal_period // 2),
                seasonal_period,
                seasonal_period * 2
            ])
        
        # Standard time-based windows
        if freq_timedelta <= 3600:  # Hourly or higher frequency
            adaptive_windows.extend([6, 12, 24, 48, 168])  # Hours to week
        elif freq_timedelta <= 86400:  # Daily
            adaptive_windows.extend([3, 7, 14, 30, 90])    # Days to quarter
        else:  # Weekly or lower frequency
            adaptive_windows.extend([2, 4, 8, 12, 26])     # Periods
        
        # Remove duplicates and sort
        return sorted(list(set(adaptive_windows)))
    
    def _add_expanding_features(self, feature_df, target_cols):
        """Add expanding window features (cumulative statistics)."""
        
        for col in target_cols:
            if col not in feature_df.columns:
                continue
            
            # Expanding mean (running average)
            feature_df[f'{col}_expanding_mean'] = (
                feature_df[col].expanding(min_periods=1).mean()
            )
            
            # Expanding std (running standard deviation)
            feature_df[f'{col}_expanding_std'] = (
                feature_df[col].expanding(min_periods=2).std()
            )
            
            # Expanding min/max (running extremes)
            feature_df[f'{col}_expanding_min'] = (
                feature_df[col].expanding(min_periods=1).min()
            )
            feature_df[f'{col}_expanding_max'] = (
                feature_df[col].expanding(min_periods=1).max()
            )
            
            # Expanding count (number of observations seen so far)
            feature_df[f'{col}_expanding_count'] = (
                feature_df[col].expanding(min_periods=1).count()
            )
            
            # Expanding sum (cumulative sum)
            feature_df[f'{col}_expanding_sum'] = (
                feature_df[col].expanding(min_periods=1).sum()
            )
        
        return feature_df
