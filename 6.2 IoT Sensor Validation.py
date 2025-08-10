class IoTSensorValidator:
    """Validation for IoT sensor time series data."""
    
    def __init__(self, sensor_type='temperature', unit='celsius'):
        self.sensor_type = sensor_type
        self.unit = unit
        self.physical_limits = self._set_physical_limits()
        self.quality_thresholds = self._set_quality_thresholds()
    
    def _set_physical_limits(self):
        """Set physically possible limits based on sensor type."""
        limits = {
            'temperature': {
                'celsius': (-273.15, 200),    # Absolute zero to very high temp
                'fahrenheit': (-459.67, 392), # Absolute zero to very high temp
            },
            'humidity': (0, 100),             # Percentage
            'pressure': (0, 2000),            # Reasonable atmospheric pressure range (hPa)
            'flow_rate': (0, None),           # Cannot be negative
            'ph': (0, 14),                    # pH scale
            'voltage': (-1000, 1000),         # Reasonable voltage range
        }
        
        return limits.get(self.sensor_type, (-float('inf'), float('inf')))
    
    def _set_quality_thresholds(self):
        """Set data quality thresholds for anomaly detection."""
        return {
            'max_rate_of_change': self._get_max_rate_of_change(),
            'stuck_value_threshold': 10,  # Same value for 10+ consecutive readings
            'noise_level_threshold': 3,   # Standard deviations for noise detection
        }
    
    def _get_max_rate_of_change(self):
        """Get maximum reasonable rate of change per time unit."""
        rates = {
            'temperature': 5.0,    # 5 degrees per minute maximum
            'humidity': 10.0,      # 10% per minute maximum
            'pressure': 50.0,      # 50 hPa per minute maximum
            'flow_rate': 100.0,    # Depends on application
        }
        return rates.get(self.sensor_type, float('inf'))
    
    def validate_sensor_data(self, data, value_col='value', timestamp_col='timestamp'):
        """Comprehensive IoT sensor data validation."""
        violations = {
            'out_of_range': [],
            'excessive_rate_change': [],
            'stuck_values': [],
            'noise_anomalies': [],
            'sensor_drift': [],
            'calibration_issues': []
        }
        
        # Physical range validation
        if isinstance(self.physical_limits, tuple):
            min_val, max_val = self.physical_limits
            if min_val is not None:
                violations['out_of_range'].extend(
                    data[data[value_col] < min_val].index.tolist()
                )
            if max_val is not None:
                violations['out_of_range'].extend(
                    data[data[value_col] > max_val].index.tolist()
                )
        
        # Rate of change validation
        if self.quality_thresholds['max_rate_of_change'] < float('inf'):
            time_diff = data[timestamp_col].diff().dt.total_seconds() / 60  # Convert to minutes
            value_diff = data[value_col].diff()
            rate_of_change = np.abs(value_diff / time_diff)
            
            excessive_rate_mask = rate_of_change > self.quality_thresholds['max_rate_of_change']
            violations['excessive_rate_change'] = data[excessive_rate_mask].index.tolist()
        
        # Stuck value detection
        stuck_sequences = self._detect_stuck_values(
            data[value_col], 
            threshold=self.quality_thresholds['stuck_value_threshold']
        )
        violations['stuck_values'] = stuck_sequences
        
        # Sensor drift detection
        drift_points = self._detect_sensor_drift(data[value_col])
        violations['sensor_drift'] = drift_points
        
        return violations
    
    def _detect_stuck_values(self, series, threshold=10):
        """Detect sequences of identical values (stuck sensor)."""
        stuck_sequences = []
        current_sequence = []
        
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1]:
                if not current_sequence:
                    current_sequence = [series.index[i-1]]
                current_sequence.append(series.index[i])
            else:
                if len(current_sequence) >= threshold:
                    stuck_sequences.extend(current_sequence)
                current_sequence = []
        
        # Check final sequence
        if len(current_sequence) >= threshold:
            stuck_sequences.extend(current_sequence)
        
        return stuck_sequences
    
    def _detect_sensor_drift(self, series, window_size=100):
        """Detect gradual sensor drift using statistical process control."""
        drift_points = []
        
        if len(series) < window_size * 2:
            return drift_points
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        # Detect significant shifts in mean
        for i in range(window_size, len(rolling_mean) - window_size):
            before_mean = rolling_mean.iloc[i - window_size:i].mean()
            after_mean = rolling_mean.iloc[i:i + window_size].mean()
            
            # Statistical significance test for mean shift
            pooled_std = np.sqrt(
                (rolling_std.iloc[i - window_size:i].var() + 
                 rolling_std.iloc[i:i + window_size].var()) / 2
            )
            
            if pooled_std > 0:
                t_stat = abs(after_mean - before_mean) / (pooled_std * np.sqrt(2/window_size))
                if t_stat > 2.0:  # Roughly 95% confidence
                    drift_points.append(series.index[i])
        
        return drift_points
