class TimelinessAssessor:
    """Assess timeliness of time series data."""
    
    def assess(self, data, timestamp_col, value_cols):
        """Assess data timeliness across multiple metrics."""
        
        results = {
            'arrival_regularity': self._assess_arrival_regularity(data, timestamp_col),
            'processing_delay': self._assess_processing_delay(data, timestamp_col),
            'data_freshness': self._assess_data_freshness(data, timestamp_col),
            'temporal_drift': self._assess_temporal_drift(data, timestamp_col)
        }
        
        # Weighted average of timeliness aspects
        overall_score = (
            results['arrival_regularity']['score'] * 0.3 +
            results['processing_delay']['score'] * 0.3 +
            results['data_freshness']['score'] * 0.2 +
            results['temporal_drift']['score'] * 0.2
        )
        
        return overall_score, results
    
    def _assess_arrival_regularity(self, data, timestamp_col):
        """Assess regularity of data arrival intervals."""
        
        timestamps = pd.to_datetime(data[timestamp_col]).sort_values()
        
        if len(timestamps) < 2:
            return {'score': 0.0, 'reason': 'Insufficient data'}
        
        # Calculate inter-arrival times
        inter_arrival_times = timestamps.diff().dropna()
        
        # Convert to seconds for analysis
        inter_arrival_seconds = inter_arrival_times.dt.total_seconds()
        
        # Calculate regularity metrics
        mean_interval = inter_arrival_seconds.mean()
        std_interval = inter_arrival_seconds.std()
        cv_interval = std_interval / mean_interval if mean_interval > 0 else float('inf')
        
        # Regularity score (lower CV is better)
        # CV of 0 = perfect regularity (score 1.0)
        # CV of 1 = high variability (score ~0.5)
        regularity_score = max(0, 1 - cv_interval)
        
        return {
            'score': regularity_score,
            'mean_interval_seconds': mean_interval,
            'std_interval_seconds': std_interval,
            'coefficient_of_variation': cv_interval,
            'min_interval_seconds': inter_arrival_seconds.min(),
            'max_interval_seconds': inter_arrival_seconds.max()
        }
    
    def _assess_processing_delay(self, data, timestamp_col):
        """Assess processing delay (current time vs. latest data timestamp)."""
        
        timestamps = pd.to_datetime(data[timestamp_col])
        latest_data_time = timestamps.max()
        current_time = pd.Timestamp.now()
        
        # Calculate delay
        delay = current_time - latest_data_time
        delay_hours = delay.total_seconds() / 3600
        
        # Score based on delay (domain-specific thresholds)
        if delay_hours <= 1:          # < 1 hour
            score = 1.0
        elif delay_hours <= 6:       # 1-6 hours
            score = 0.8
        elif delay_hours <= 24:      # 6-24 hours
            score = 0.6
        elif delay_hours <= 168:     # 1-7 days
            score = 0.4
        else:                        # > 7 days
            score = 0.2
        
        return {
            'score': score,
            'latest_data_timestamp': latest_data_time,
            'current_timestamp': current_time,
            'delay_hours': delay_hours,
            'delay_category': self._categorize_delay(delay_hours)
        }
    
    def _assess_data_freshness(self, data, timestamp_col):
        """Assess overall data freshness distribution."""
        
        timestamps = pd.to_datetime(data[timestamp_col])
        current_time = pd.Timestamp.now()
        
        # Calculate age of all data points
        data_ages = current_time - timestamps
        age_hours = data_ages.dt.total_seconds() / 3600
        
        # Freshness distribution
        fresh_count = (age_hours <= 24).sum()      # < 1 day
        recent_count = (age_hours <= 168).sum()    # < 1 week
        old_count = (age_hours > 168).sum()        # > 1 week
        
        total_count = len(age_hours)
        
        # Freshness score
        if total_count > 0:
            freshness_score = (
                (fresh_count / total_count) * 1.0 +
                ((recent_count - fresh_count) / total_count) * 0.7 +
                (old_count / total_count) * 0.3
            )
        else:
            freshness_score = 0.0
        
        return {
            'score': freshness_score,
            'fresh_data_percentage': fresh_count / total_count if total_count > 0 else 0,
            'recent_data_percentage': recent_count / total_count if total_count > 0 else 0,
            'old_data_percentage': old_count / total_count if total_count > 0 else 0,
            'median_age_hours': age_hours.median()
        }
    
    def _assess_temporal_drift(self, data, timestamp_col):
        """Assess temporal drift in data arrival patterns."""
        
        timestamps = pd.to_datetime(data[timestamp_col]).sort_values()
        
        if len(timestamps) < 10:
            return {'score': 1.0, 'reason': 'Insufficient data for drift analysis'}
        
        # Split data into chunks to analyze trend
        chunk_size = max(5, len(timestamps) // 5)
        chunks = [timestamps[i:i+chunk_size] for i in range(0, len(timestamps), chunk_size)]
        
        if len(chunks) < 2:
            return {'score': 1.0, 'reason': 'Insufficient chunks for drift analysis'}
        
        # Calculate mean interval for each chunk
        chunk_intervals = []
        for chunk in chunks:
            if len(chunk) > 1:
                intervals = chunk.diff().dropna().dt.total_seconds()
                chunk_intervals.append(intervals.mean())
        
        if len(chunk_intervals) < 2:
            return {'score': 1.0, 'reason': 'Insufficient intervals for drift analysis'}
        
        # Check for trend in intervals (increasing/decreasing over time)
        from scipy.stats import linregress
        
        x = np.arange(len(chunk_intervals))
        slope, intercept, r_value, p_value, std_err = linregress(x, chunk_intervals)
        
        # Drift score (no trend is good, strong trend is bad)
        abs_r_value = abs(r_value)
        drift_score = max(0, 1 - abs_r_value)
        
        return {
            'score': drift_score,
            'trend_slope': slope,
            'trend_correlation': r_value,
            'trend_p_value': p_value,
            'drift_detected': abs_r_value > 0.3 and p_value < 0.05
        }
    
    def _categorize_delay(self, delay_hours):
        """Categorize delay into human-readable categories."""
        if delay_hours <= 1:
            return 'Very Fresh'
        elif delay_hours <= 6:
            return 'Fresh'
        elif delay_hours <= 24:
            return 'Recent'
        elif delay_hours <= 168:
            return 'Stale'
        else:
            return 'Very Stale'
