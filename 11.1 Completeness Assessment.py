class CompletenessAssessor:
    """Assess completeness of time series data."""
    
    def assess(self, data, timestamp_col, value_cols):
        """
        Assess completeness across multiple aspects.
        """
        results = {
            'temporal_completeness': self._assess_temporal_completeness(data, timestamp_col),
            'value_completeness': self._assess_value_completeness(data, value_cols),
            'contextual_completeness': self._assess_contextual_completeness(data, timestamp_col)
        }
        
        # Weighted average of completeness aspects
        overall_score = (
            results['temporal_completeness']['score'] * 0.4 +
            results['value_completeness']['score'] * 0.4 +
            results['contextual_completeness']['score'] * 0.2
        )
        
        return overall_score, results
    
    def _assess_temporal_completeness(self, data, timestamp_col):
        """Assess temporal completeness - missing timestamps."""
        
        timestamps = pd.to_datetime(data[timestamp_col])
        timestamps = timestamps.sort_values()
        
        # Infer expected frequency
        freq = pd.infer_freq(timestamps)
        
        if freq is None:
            # Try to infer from most common difference
            diffs = timestamps.diff().dropna()
            mode_diff = diffs.mode()
            if len(mode_diff) > 0:
                freq = mode_diff.iloc[0]
            else:
                return {'score': 0.5, 'reason': 'Cannot infer frequency'}
        
        # Generate expected timestamp range
        expected_range = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq=freq
        )
        
        # Calculate missing timestamps
        missing_timestamps = expected_range.difference(timestamps)
        completeness_ratio = 1 - (len(missing_timestamps) / len(expected_range))
        
        return {
            'score': max(0, completeness_ratio),
            'expected_count': len(expected_range),
            'actual_count': len(timestamps),
            'missing_count': len(missing_timestamps),
            'inferred_frequency': str(freq),
            'missing_timestamps': missing_timestamps.tolist()[:10]  # Sample
        }
    
    def _assess_value_completeness(self, data, value_cols):
        """Assess value completeness - missing values in data columns."""
        
        completeness_by_column = {}
        total_cells = 0
        missing_cells = 0
        
        for col in value_cols:
            column_total = len(data)
            column_missing = data[col].isnull().sum()
            column_completeness = 1 - (column_missing / column_total) if column_total > 0 else 0
            
            completeness_by_column[col] = {
                'completeness_rate': column_completeness,
                'missing_count': column_missing,
                'total_count': column_total
            }
            
            total_cells += column_total
            missing_cells += column_missing
        
        overall_completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        return {
            'score': overall_completeness,
            'by_column': completeness_by_column,
            'total_missing_rate': missing_cells / total_cells if total_cells > 0 else 0
        }
    
    def _assess_contextual_completeness(self, data, timestamp_col):
        """Assess contextual completeness - business day/hour coverage."""
        
        timestamps = pd.to_datetime(data[timestamp_col])
        
        # Business hours coverage (9 AM - 5 PM weekdays)
        business_hours = timestamps[(timestamps.dt.hour >= 9) & 
                                   (timestamps.dt.hour <= 17) &
                                   (timestamps.dt.weekday < 5)]
        
        business_hour_coverage = len(business_hours) / len(timestamps) if len(timestamps) > 0 else 0
        
        # Weekend coverage
        weekend_hours = timestamps[timestamps.dt.weekday >= 5]
        weekend_coverage = len(weekend_hours) / len(timestamps) if len(timestamps) > 0 else 0
        
        # Holiday coverage (approximate)
        holidays = self._approximate_holidays(timestamps)
        holiday_coverage = len(holidays) / len(timestamps) if len(timestamps) > 0 else 0
        
        # Contextual score based on expected coverage patterns
        # This is domain-specific and may need adjustment
        contextual_score = (business_hour_coverage * 0.6 + 
                          weekend_coverage * 0.2 + 
                          holiday_coverage * 0.2)
        
        return {
            'score': min(1.0, contextual_score),  # Cap at 1.0
            'business_hour_coverage': business_hour_coverage,
            'weekend_coverage': weekend_coverage,
            'holiday_coverage': holiday_coverage
        }
    
    def _approximate_holidays(self, timestamps):
        """Approximate holiday detection (simplified)."""
        # This is a simplified approach - real implementation would use
        # proper holiday libraries like 'holidays' package
        
        # Common holidays (approximate dates)
        years = timestamps.dt.year.unique()
        approximate_holidays = []
        
        for year in years:
            holidays_for_year = [
                pd.Timestamp(f'{year}-01-01'),  # New Year
                pd.Timestamp(f'{year}-07-04'),  # Independence Day (US)
                pd.Timestamp(f'{year}-12-25'),  # Christmas
            ]
            approximate_holidays.extend(holidays_for_year)
        
        # Find timestamps that fall on these holidays
        holiday_timestamps = [ts for ts in timestamps 
                            if any(ts.date() == h.date() for h in approximate_holidays)]
        
        return holiday_timestamps
