def validate_temporal_ordering(data, timestamp_col='timestamp'):
    """
    Comprehensive validation of temporal ordering and consistency.
    """
    issues = {
        'duplicate_timestamps': [],
        'out_of_order': [],
        'future_timestamps': [],
        'impossible_timestamps': [],
        'frequency_violations': []
    }
    
    timestamps = data[timestamp_col]
    
    # Check for duplicates
    duplicates = timestamps[timestamps.duplicated()]
    issues['duplicate_timestamps'] = duplicates.tolist()
    
    # Check chronological ordering
    for i in range(1, len(timestamps)):
        if timestamps.iloc[i] <= timestamps.iloc[i-1]:
            issues['out_of_order'].append({
                'index': i,
                'timestamp': timestamps.iloc[i],
                'previous': timestamps.iloc[i-1]
            })
    
    # Check for future timestamps
    current_time = pd.Timestamp.now()
    future_mask = timestamps > current_time
    issues['future_timestamps'] = timestamps[future_mask].tolist()
    
    # Check for impossible timestamps (before year 1900 or too far in future)
    min_valid = pd.Timestamp('1900-01-01')
    max_valid = pd.Timestamp('2100-01-01')
    impossible_mask = (timestamps < min_valid) | (timestamps > max_valid)
    issues['impossible_timestamps'] = timestamps[impossible_mask].tolist()
    
    # Infer expected frequency and check violations
    inferred_freq = pd.infer_freq(timestamps.sort_values())
    if inferred_freq:
        expected_range = pd.date_range(
            start=timestamps.min(), 
            end=timestamps.max(), 
            freq=inferred_freq
        )
        missing_timestamps = expected_range.difference(timestamps)
        issues['frequency_violations'] = missing_timestamps.tolist()
    
    return issues

def repair_timestamp_issues(data, timestamp_col='timestamp', strategy='conservative'):
    """
    Repair common timestamp issues based on specified strategy.
    """
    repaired_data = data.copy()
    repair_log = []
    
    if strategy == 'conservative':
        # Remove duplicates, keeping first occurrence
        before_count = len(repaired_data)
        repaired_data = repaired_data.drop_duplicates(subset=[timestamp_col], keep='first')
        after_count = len(repaired_data)
        
        if before_count != after_count:
            repair_log.append(f"Removed {before_count - after_count} duplicate timestamps")
        
        # Sort by timestamp to fix ordering
        repaired_data = repaired_data.sort_values(timestamp_col)
        repair_log.append("Sorted data chronologically")
        
        # Remove impossible timestamps
        min_valid = pd.Timestamp('1900-01-01')
        max_valid = pd.Timestamp('2100-01-01')
        valid_mask = (
            (repaired_data[timestamp_col] >= min_valid) & 
            (repaired_data[timestamp_col] <= max_valid)
        )
        
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            repaired_data = repaired_data[valid_mask]
            repair_log.append(f"Removed {invalid_count} invalid timestamps")
    
    return repaired_data, repair_log
