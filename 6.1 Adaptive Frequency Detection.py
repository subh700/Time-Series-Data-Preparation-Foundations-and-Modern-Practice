def detect_and_regularize_frequency(data, timestamp_col='timestamp', value_col='value'):
    """
    Detect the most appropriate frequency and regularize the time series.
    """
    timestamps = data[timestamp_col]
    
    # Calculate time differences
    time_diffs = timestamps.diff().dropna()
    
    # Find the most common time difference (mode)
    mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else None
    
    # Calculate frequency statistics
    freq_stats = {
        'mean_diff': time_diffs.mean(),
        'median_diff': time_diffs.median(),
        'mode_diff': mode_diff,
        'std_diff': time_diffs.std(),
        'regularity_score': calculate_regularity_score(time_diffs)
    }
    
    # Determine best frequency for regularization
    if freq_stats['regularity_score'] > 0.8:  # Highly regular
        target_freq = freq_stats['mode_diff']
    else:  # Irregular - use median as compromise
        target_freq = freq_stats['median_diff']
    
    # Create regular time index
    regular_index = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=target_freq
    )
    
    # Reindex data to regular frequency
    data_indexed = data.set_index(timestamp_col)
    regularized_data = data_indexed.reindex(regular_index, method='nearest', tolerance=target_freq/2)
    
    return {
        'regularized_data': regularized_data,
        'frequency_stats': freq_stats,
        'target_frequency': target_freq,
        'regularity_improvement': calculate_regularity_improvement(timestamps, regular_index)
    }

def calculate_regularity_score(time_diffs):
    """Calculate how regular the time intervals are (0=irregular, 1=perfectly regular)."""
    if len(time_diffs) == 0:
        return 0
    
    mode_diff = time_diffs.mode().iloc[0] if not time_diffs.mode().empty else time_diffs.median()
    deviations = np.abs(time_diffs - mode_diff)
    
    # Normalize deviations by mode difference
    normalized_deviations = deviations / mode_diff
    
    # Calculate regularity as inverse of average relative deviation
    avg_deviation = normalized_deviations.mean()
    regularity = max(0, 1 - avg_deviation)
    
    return regularity
