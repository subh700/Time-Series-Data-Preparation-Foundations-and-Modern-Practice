class ConsistencyAssessor:
    """Assess consistency of time series data."""
    
    def assess(self, data, timestamp_col, value_cols):
        """Assess data consistency across multiple dimensions."""
        
        results = {
            'format_consistency': self._assess_format_consistency(data, timestamp_col, value_cols),
            'temporal_consistency': self._assess_temporal_consistency(data, timestamp_col),
            'value_consistency': self._assess_value_consistency(data, value_cols),
            'cross_series_consistency': self._assess_cross_series_consistency(data, value_cols)
        }
        
        # Weighted average of consistency aspects
        overall_score = (
            results['format_consistency']['score'] * 0.25 +
            results['temporal_consistency']['score'] * 0.25 +
            results['value_consistency']['score'] * 0.25 +
            results['cross_series_consistency']['score'] * 0.25
        )
        
        return overall_score, results
    
    def _assess_format_consistency(self, data, timestamp_col, value_cols):
        """Assess consistency of data formats."""
        
        format_issues = {}
        
        # Timestamp format consistency
        timestamp_series = data[timestamp_col]
        
        # Check for mixed timestamp formats
        try:
            parsed_timestamps = pd.to_datetime(timestamp_series, infer_datetime_format=True)
            timestamp_parsing_success = (~parsed_timestamps.isnull()).sum() / len(timestamp_series)
        except:
            timestamp_parsing_success = 0.0
        
        format_issues['timestamp_format'] = {
            'parsing_success_rate': timestamp_parsing_success,
            'unparseable_count': len(timestamp_series) - int(timestamp_parsing_success * len(timestamp_series))
        }
        
        # Value format consistency
        value_format_scores = []
        
        for col in value_cols:
            # Check numeric consistency
            numeric_conversion_rate = pd.to_numeric(data[col], errors='coerce').notna().sum() / len(data[col])
            
            # Check for mixed types
            type_counts = data[col].apply(type).value_counts()
            type_consistency = type_counts.iloc[0] / len(data[col]) if len(type_counts) > 0 else 0
            
            format_issues[f'{col}_format'] = {
                'numeric_conversion_rate': numeric_conversion_rate,
                'type_consistency': type_consistency,
                'detected_types': type_counts.to_dict()
            }
            
            value_format_scores.append((numeric_conversion_rate + type_consistency) / 2)
        
        overall_format_score = (
            timestamp_parsing_success * 0.5 +
            np.mean(value_format_scores) * 0.5 if value_format_scores else timestamp_parsing_success
        )
        
        return {
            'score': overall_format_score,
            'details': format_issues
        }
    
    def _assess_temporal_consistency(self, data, timestamp_col):
        """Assess temporal consistency - ordering, duplicates, gaps."""
        
        timestamps = pd.to_datetime(data[timestamp_col])
        
        # Check chronological ordering
        is_sorted = timestamps.is_monotonic_increasing
        
        # Check for duplicate timestamps
        duplicate_timestamps = timestamps.duplicated().sum()
        
        # Check for reasonable timestamp ranges
        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()
        current_time = pd.Timestamp.now()
        
        reasonable_range = (
            min_timestamp >= pd.Timestamp('1970-01-01') and
            max_timestamp <= current_time + pd.Timedelta(days=1)  # Allow slight future dates
        )
        
        # Calculate consistency score
        ordering_score = 1.0 if is_sorted else 0.5
        duplicate_score = 1.0 - (duplicate_timestamps / len(timestamps))
        range_score = 1.0 if reasonable_range else 0.5
        
        temporal_consistency_score = (ordering_score + duplicate_score + range_score) / 3
        
        return {
            'score': temporal_consistency_score,
            'is_chronologically_ordered': is_sorted,
            'duplicate_timestamp_count': duplicate_timestamps,
            'timestamp_range_reasonable': reasonable_range,
            'min_timestamp': min_timestamp,
            'max_timestamp': max_timestamp
        }
    
    def _assess_value_consistency(self, data, value_cols):
        """Assess consistency of value distributions over time."""
        
        consistency_scores = []
        details = {}
        
        for col in value_cols:
            series = pd.to_numeric(data[col], errors='coerce')
            
            if series.notna().sum() < 10:
                details[col] = {'score': 0.5, 'reason': 'Insufficient numeric data'}
                continue
            
            # Split data into temporal chunks
            chunk_size = max(10, len(series) // 5)
            chunks = []
            
            for i in range(0, len(series), chunk_size):
                chunk = series.iloc[i:i+chunk_size].dropna()
                if len(chunk) >= 5:  # Minimum for statistical analysis
                    chunks.append(chunk)
            
            if len(chunks) < 2:
                details[col] = {'score': 0.5, 'reason': 'Insufficient chunks for consistency analysis'}
                continue
            
            # Compare distribution characteristics across chunks
            means = [chunk.mean() for chunk in chunks]
            stds = [chunk.std() for chunk in chunks]
            
            # Consistency measured by stability of mean and std
            mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
            std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else 0
            
            # Lower coefficient of variation indicates higher consistency
            consistency_score = max(0, 1 - (mean_cv + std_cv) / 2)
            
            consistency_scores.append(consistency_score)
            details[col] = {
                'score': consistency_score,
                'mean_coefficient_of_variation': mean_cv,
                'std_coefficient_of_variation': std_cv,
                'chunk_count': len(chunks)
            }
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return {
            'score': overall_consistency,
            'by_column': details
        }
    
    def _assess_cross_series_consistency(self, data, value_cols):
        """Assess consistency across related time series."""
        
        if len(value_cols) < 2:
            return {'score': 1.0, 'reason': 'Single series - no cross-series analysis needed'}
        
        consistency_results = {}
        
        # Convert all columns to numeric
        numeric_data = {}
        for col in value_cols:
            numeric_series = pd.to_numeric(data[col], errors='coerce')
            if numeric_series.notna().sum() >= 10:
                numeric_data[col] = numeric_series
        
        if len(numeric_data) < 2:
            return {'score': 0.5, 'reason': 'Insufficient numeric columns for cross-series analysis'}
        
        # Calculate pairwise correlations
        correlation_matrix = pd.DataFrame(numeric_data).corr()
        
        # Stability of correlations over time
        chunk_size = max(50, len(data) // 5)
        correlation_stability_scores = []
        
        for i in range(len(numeric_data)):
            for j in range(i+1, len(numeric_data)):
                col1, col2 = list(numeric_data.keys())[i], list(numeric_data.keys())[j]
                
                chunk_correlations = []
                for start_idx in range(0, len(data) - chunk_size, chunk_size):
                    chunk_data = pd.DataFrame({
                        col1: numeric_data[col1].iloc[start_idx:start_idx+chunk_size],
                        col2: numeric_data[col2].iloc[start_idx:start_idx+chunk_size]
                    }).dropna()
                    
                    if len(chunk_data) >= 10:
                        chunk_corr = chunk_data[col1].corr(chunk_data[col2])
                        if not pd.isna(chunk_corr):
                            chunk_correlations.append(chunk_corr)
                
                if len(chunk_correlations) >= 2:
                    stability_score = 1 - np.std(chunk_correlations)  # Lower std = higher stability
                    correlation_stability_scores.append(max(0, stability_score))
        
        cross_series_score = np.mean(correlation_stability_scores) if correlation_stability_scores else 0.5
        
        return {
            'score': cross_series_score,
            'correlation_matrix': correlation_matrix.to_dict(),
            'correlation_stability_scores': correlation_stability_scores
        }
