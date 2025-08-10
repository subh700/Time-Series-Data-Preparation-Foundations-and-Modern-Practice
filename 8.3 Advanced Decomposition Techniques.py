class SeasonalDecompositionFeatureEngineer:
    """
    Extract features from time series decomposition using multiple methods.
    """
    
    def __init__(self, decomposition_methods=None, seasonal_periods=None):
        self.decomposition_methods = decomposition_methods or ['stl', 'x13', 'mstl']
        self.seasonal_periods = seasonal_periods or [24, 168, 8760]  # Hour, week, year
        
    def create_decomposition_features(self, data, target_col, timestamp_col=None):
        """
        Create comprehensive decomposition-based features using multiple methods.
        """
        feature_df = data.copy()
        decomposition_results = {}
        
        # STL Decomposition
        if 'stl' in self.decomposition_methods:
            stl_features, stl_results = self._stl_decomposition_features(
                data[target_col], self.seasonal_periods[0]
            )
            feature_df = pd.concat([feature_df, stl_features], axis=1)
            decomposition_results['stl'] = stl_results
        
        # X-13 ARIMA-SEATS (if available)
        if 'x13' in self.decomposition_methods:
            try:
                x13_features, x13_results = self._x13_decomposition_features(
                    data[target_col], timestamp_col
                )
                feature_df = pd.concat([feature_df, x13_features], axis=1)
                decomposition_results['x13'] = x13_results
            except Exception as e:
                print(f"X-13 decomposition failed: {e}")
        
        # Multiple Seasonal-Trend decomposition using Loess (MSTL)
        if 'mstl' in self.decomposition_methods and len(self.seasonal_periods) > 1:
            mstl_features, mstl_results = self._mstl_decomposition_features(
                data[target_col], self.seasonal_periods
            )
            feature_df = pd.concat([feature_df, mstl_features], axis=1)
            decomposition_results['mstl'] = mstl_results
        
        # Custom decomposition features
        feature_df = self._add_decomposition_derived_features(
            feature_df, target_col, decomposition_results
        )
        
        return feature_df, decomposition_results
    
    def _stl_decomposition_features(self, series, seasonal_period):
        """STL decomposition with feature extraction."""
        from statsmodels.tsa.seasonal import STL
        
        # Perform STL decomposition
        stl = STL(series.dropna(), seasonal=seasonal_period, robust=True)
        decomposition = stl.fit()
        
        # Create feature DataFrame
        features = pd.DataFrame(index=series.index)
        
        # Basic components
        features['stl_trend'] = decomposition.trend.reindex(series.index)
        features['stl_seasonal'] = decomposition.seasonal.reindex(series.index)
        features['stl_resid'] = decomposition.resid.reindex(series.index)
        
        # Derived features
        features['stl_detrended'] = series - features['stl_trend']
        features['stl_deseasonalized'] = series - features['stl_seasonal']
        features['stl_strength_of_trend'] = self._calculate_strength_of_trend(
            features['stl_trend'], series
        )
        features['stl_strength_of_seasonality'] = self._calculate_strength_of_seasonality(
            features['stl_seasonal'], series
        )
        
        # Seasonal pattern characteristics
        seasonal_component = features['stl_seasonal'].dropna()
        if len(seasonal_component) >= seasonal_period:
            # Seasonal amplitude
            features['stl_seasonal_amplitude'] = (
                seasonal_component.rolling(window=seasonal_period).max() - 
                seasonal_component.rolling(window=seasonal_period).min()
            )
            
            # Seasonal stability (rolling correlation with average seasonal pattern)
            avg_seasonal_pattern = (
                seasonal_component.groupby(seasonal_component.index % seasonal_period).mean()
            )
            features['stl_seasonal_stability'] = (
                seasonal_component.rolling(window=seasonal_period)
                .apply(lambda x: np.corrcoef(x, avg_seasonal_pattern)[0,1] 
                       if len(x) == seasonal_period else np.nan)
            )
        
        return features, decomposition
    
    def _mstl_decomposition_features(self, series, seasonal_periods):
        """Multiple Seasonal-Trend decomposition using Loess (MSTL)."""
        try:
            from statsmodels.tsa.seasonal import MSTL
            
            # Perform MSTL decomposition
            mstl = MSTL(series.dropna(), periods=seasonal_periods, robust=True)
            decomposition = mstl.fit()
            
            features = pd.DataFrame(index=series.index)
            
            # Trend component
            features['mstl_trend'] = decomposition.trend.reindex(series.index)
            
            # Multiple seasonal components
            for i, period in enumerate(seasonal_periods):
                seasonal_col = f'seasonal_{period}'
                if hasattr(decomposition, seasonal_col):
                    features[f'mstl_seasonal_{period}'] = (
                        getattr(decomposition, seasonal_col).reindex(series.index)
                    )
                    
                    # Seasonal strength for each component
                    features[f'mstl_seasonal_strength_{period}'] = (
                        self._calculate_strength_of_seasonality(
                            features[f'mstl_seasonal_{period}'], series
                        )
                    )
            
            # Residual
            features['mstl_resid'] = decomposition.resid.reindex(series.index)
            
            # Combined seasonal effect
            seasonal_cols = [col for col in features.columns if 'mstl_seasonal_' in col]
            if seasonal_cols:
                features['mstl_total_seasonal'] = features[seasonal_cols].sum(axis=1)
            
            return features, decomposition
            
        except ImportError:
            print("MSTL not available. Please update statsmodels.")
            return pd.DataFrame(index=series.index), None
    
    def _calculate_strength_of_trend(self, trend_component, original_series):
        """Calculate strength of trend component."""
        detrended = original_series - trend_component
        if detrended.var() == 0:
            return 1.0
        return max(0, 1 - (detrended.var() / original_series.var()))
    
    def _calculate_strength_of_seasonality(self, seasonal_component, original_series):
        """Calculate strength of seasonal component."""
        deseasoned = original_series - seasonal_component
        if deseasoned.var() == 0:
            return 1.0
        return max(0, 1 - (deseasoned.var() / original_series.var()))
    
    def _add_decomposition_derived_features(self, feature_df, target_col, decomposition_results):
        """Add derived features from decomposition results."""
        
        # Trend direction and acceleration
        if 'stl_trend' in feature_df.columns:
            feature_df['stl_trend_direction'] = np.sign(
                feature_df['stl_trend'].diff()
            )
            feature_df['stl_trend_acceleration'] = (
                feature_df['stl_trend'].diff().diff()
            )
            
            # Trend turning points
            trend_direction_changes = (
                feature_df['stl_trend_direction'].diff() != 0
            )
            feature_df['stl_trend_turning_point'] = trend_direction_changes.astype(int)
        
        # Residual analysis
        if 'stl_resid' in feature_df.columns:
            # Residual volatility
            feature_df['stl_resid_volatility'] = (
                feature_df['stl_resid'].rolling(window=30).std()
            )
            
            # Residual outliers
            resid_mean = feature_df['stl_resid'].mean()
            resid_std = feature_df['stl_resid'].std()
            feature_df['stl_resid_outlier'] = (
                np.abs(feature_df['stl_resid'] - resid_mean) > 2 * resid_std
            ).astype(int)
        
        # Ratio features
        for comp in ['trend', 'seasonal', 'resid']:
            if f'stl_{comp}' in feature_df.columns:
                feature_df[f'stl_{comp}_ratio'] = (
                    feature_df[f'stl_{comp}'] / feature_df[target_col]
                )
        
        return feature_df
