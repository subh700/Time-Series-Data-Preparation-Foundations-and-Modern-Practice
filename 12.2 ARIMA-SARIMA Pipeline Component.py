class ARIMAPreprocessor(TimeSeriesPipelineComponent):
    """
    Specialized preprocessing for ARIMA/SARIMA models.
    """
    
    def __init__(self, name: str = "arima_preprocessor", config: Dict[str, Any] = None):
        default_config = {
            'target_column': 'value',
            'timestamp_column': 'timestamp',
            'frequency': None,  # Auto-detect
            'seasonal_periods': [12, 24, 168],  # Monthly, daily, weekly
            'max_differencing_order': 2,
            'log_transform': 'auto',  # True, False, or 'auto'
            'outlier_treatment': 'winsorize',  # 'remove', 'winsorize', 'none'
            'missing_value_strategy': 'interpolate'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # Fitted parameters
        self.frequency = None
        self.seasonal_periods_detected = []
        self.differencing_order = 0
        self.seasonal_differencing_orders = {}
        self.log_transform_applied = False
        self.outlier_bounds = {}
        self.stationarity_tests = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ARIMAPreprocessor':
        """Fit ARIMA preprocessor to data."""
        
        self.validate_data(data)
        
        target_col = self.config['target_column']
        timestamp_col = self.config['timestamp_column']
        
        # Ensure proper datetime index
        if timestamp_col in data.columns:
            data = data.set_index(timestamp_col)
        
        series = data[target_col].copy()
        
        # 1. Handle missing values
        series = self._handle_missing_values(series)
        
        # 2. Detect and handle outliers
        self.outlier_bounds = self._detect_outliers(series)
        series = self._treat_outliers(series)
        
        # 3. Determine frequency
        self.frequency = self._detect_frequency(series.index)
        
        # 4. Test for seasonality
        self.seasonal_periods_detected = self._detect_seasonality(series)
        
        # 5. Determine log transformation
        self.log_transform_applied = self._should_apply_log_transform(series)
        if self.log_transform_applied:
            series = np.log(series)
        
        # 6. Determine differencing requirements
        self._determine_differencing_requirements(series)
        
        # Store metadata
        self.metadata.update({
            'original_series_length': len(data),
            'frequency': self.frequency,
            'seasonal_periods': self.seasonal_periods_detected,
            'differencing_order': self.differencing_order,
            'seasonal_differencing_orders': self.seasonal_differencing_orders,
            'log_transform_applied': self.log_transform_applied,
            'outlier_bounds': self.outlier_bounds,
            'stationarity_tests': self.stationarity_tests
        })
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data for ARIMA modeling."""
        
        if not self.is_fitted:
            raise RuntimeError("ARIMAPreprocessor must be fitted before transformation")
        
        result_data = data.copy()
        target_col = self.config['target_column']
        timestamp_col = self.config['timestamp_column']
        
        # Ensure proper datetime index
        if timestamp_col in result_data.columns:
            result_data = result_data.set_index(timestamp_col)
        
        series = result_data[target_col].copy()
        
        # Apply transformations in the same order as fitting
        series = self._handle_missing_values(series)
        series = self._treat_outliers(series)
        
        if self.log_transform_applied:
            series = np.log(series)
        
        # Apply differencing
        differenced_series = self._apply_differencing(series)
        
        # Create output DataFrame
        output_data = pd.DataFrame(index=differenced_series.index)
        output_data['differenced_value'] = differenced_series
        output_data['original_value'] = series.reindex(differenced_series.index)
        
        # Add seasonal indicators if seasonal periods detected
        for period in self.seasonal_periods_detected:
            output_data[f'seasonal_{period}'] = (
                output_data.index.to_series().dt.dayofyear % period
            )
        
        return output_data
    
    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """Handle missing values according to strategy."""
        
        strategy = self.config['missing_value_strategy']
        
        if strategy == 'interpolate':
            return series.interpolate(method='time')
        elif strategy == 'forward_fill':
            return series.fillna(method='ffill')
        elif strategy == 'backward_fill':
            return series.fillna(method='bfill')
        elif strategy == 'drop':
            return series.dropna()
        else:
            return series
    
    def _detect_outliers(self, series: pd.Series) -> Dict[str, float]:
        """Detect outliers using IQR method."""
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': ((series < lower_bound) | (series > upper_bound)).sum()
        }
    
    def _treat_outliers(self, series: pd.Series) -> pd.Series:
        """Treat outliers according to specified method."""
        
        treatment = self.config['outlier_treatment']
        
        if treatment == 'none':
            return series
        
        lower_bound = self.outlier_bounds['lower_bound']
        upper_bound = self.outlier_bounds['upper_bound']
        
        if treatment == 'remove':
            mask = (series >= lower_bound) & (series <= upper_bound)
            return series[mask]
        elif treatment == 'winsorize':
            return series.clip(lower=lower_bound, upper=upper_bound)
        
        return series
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect time series frequency."""
        
        if self.config['frequency']:
            return self.config['frequency']
        
        # Attempt to infer frequency
        inferred_freq = pd.infer_freq(index)
        
        if inferred_freq:
            return inferred_freq
        
        # Fallback: calculate most common difference
        diffs = index.to_series().diff().dropna()
        mode_diff = diffs.mode()
        
        if len(mode_diff) > 0:
            # Convert timedelta to frequency string (simplified)
            seconds = mode_diff.iloc[0].total_seconds()
            
            if seconds == 3600:  # 1 hour
                return 'H'
            elif seconds == 86400:  # 1 day
                return 'D'
            elif seconds == 604800:  # 1 week
                return 'W'
            else:
                return f'{int(seconds)}S'
        
        return 'infer'
    
    def _detect_seasonality(self, series: pd.Series) -> List[int]:
        """Detect seasonal periods in the series."""
        
        from scipy import signal
        
        detected_periods = []
        
        for period in self.config['seasonal_periods']:
            if len(series) < 2 * period:
                continue
            
            # Use autocorrelation to test for seasonality
            autocorr = series.autocorr(lag=period)
            
            if autocorr > 0.3:  # Threshold for significant autocorrelation
                detected_periods.append(period)
        
        return detected_periods
    
    def _should_apply_log_transform(self, series: pd.Series) -> bool:
        """Determine if log transformation should be applied."""
        
        log_config = self.config['log_transform']
        
        if log_config == True:
            return True
        elif log_config == False:
            return False
        else:  # 'auto'
            # Apply log transform if variance increases with level
            if series.min() <= 0:
                return False  # Cannot log transform non-positive values
            
            # Split series into high and low value periods
            median_value = series.median()
            high_periods = series[series > median_value]
            low_periods = series[series <= median_value]
            
            if len(high_periods) > 10 and len(low_periods) > 10:
                high_variance = high_periods.var()
                low_variance = low_periods.var()
                
                # Apply log if high-value periods have much higher variance
                return high_variance > 2 * low_variance
            
            return False
    
    def _determine_differencing_requirements(self, series: pd.Series) -> None:
        """Determine required differencing orders for stationarity."""
        
        from statsmodels.tsa.stattools import adfuller
        
        # Test original series
        current_series = series.copy()
        differencing_order = 0
        max_order = self.config['max_differencing_order']
        
        # Regular differencing
        while differencing_order < max_order:
            # Augmented Dickey-Fuller test
            adf_stat, adf_p_value = adfuller(current_series.dropna())[:2]
            
            self.stationarity_tests[f'adf_d{differencing_order}'] = {
                'statistic': adf_stat,
                'p_value': adf_p_value,
                'is_stationary': adf_p_value < 0.05
            }
            
            if adf_p_value < 0.05:  # Series is stationary
                break
            
            # Apply differencing
            current_series = current_series.diff().dropna()
            differencing_order += 1
        
        self.differencing_order = differencing_order
        
        # Seasonal differencing for each detected seasonal period
        for period in self.seasonal_periods_detected:
            seasonal_diff_order = 0
            seasonal_series = series.copy()
            
            # Apply regular differencing first
            for _ in range(self.differencing_order):
                seasonal_series = seasonal_series.diff().dropna()
            
            # Test for seasonal stationarity
            if len(seasonal_series) > 2 * period:
                seasonal_adf_stat, seasonal_adf_p = adfuller(seasonal_series)[:2]
                
                if seasonal_adf_p >= 0.05:  # Not seasonally stationary
                    seasonal_series = seasonal_series.diff(periods=period).dropna()
                    seasonal_diff_order = 1
                    
                    # Test again
                    if len(seasonal_series) > period:
                        final_adf_stat, final_adf_p = adfuller(seasonal_series)[:2]
                        
                        self.stationarity_tests[f'seasonal_adf_{period}'] = {
                            'statistic': final_adf_stat,
                            'p_value': final_adf_p,
                            'is_stationary': final_adf_p < 0.05
                        }
            
            self.seasonal_differencing_orders[period] = seasonal_diff_order
    
    def _apply_differencing(self, series: pd.Series) -> pd.Series:
        """Apply determined differencing to series."""
        
        result = series.copy()
        
        # Apply regular differencing
        for _ in range(self.differencing_order):
            result = result.diff().dropna()
        
        # Apply seasonal differencing
        for period, order in self.seasonal_differencing_orders.items():
            for _ in range(order):
                result = result.diff(periods=period).dropna()
        
        return result
    
    def _get_state(self) -> Dict[str, Any]:
        """Get component state for serialization."""
        return {
            'frequency': self.frequency,
            'seasonal_periods_detected': self.seasonal_periods_detected,
            'differencing_order': self.differencing_order,
            'seasonal_differencing_orders': self.seasonal_differencing_orders,
            'log_transform_applied': self.log_transform_applied,
            'outlier_bounds': self.outlier_bounds,
            'stationarity_tests': self.stationarity_tests
        }
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set component state from deserialization."""
        self.frequency = state['frequency']
        self.seasonal_periods_detected = state['seasonal_periods_detected']
        self.differencing_order = state['differencing_order']
        self.seasonal_differencing_orders = state['seasonal_differencing_orders']
        self.log_transform_applied = state['log_transform_applied']
        self.outlier_bounds = state['outlier_bounds']
        self.stationarity_tests = state['stationarity_tests']
