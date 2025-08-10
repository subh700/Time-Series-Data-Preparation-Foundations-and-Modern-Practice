def assess_outlier_business_impact(data, outlier_indices, context_window=24):
    """
    Assess business impact of detected outliers to guide treatment decisions.
    """
    impact_assessment = {}
    
    for idx in outlier_indices:
        # Extract context around outlier
        start_idx = max(0, idx - context_window)
        end_idx = min(len(data), idx + context_window + 1)
        context = data.iloc[start_idx:end_idx]
        
        # Calculate impact metrics
        outlier_value = data.iloc[idx]
        local_mean = context.drop(context.index[idx - start_idx]).mean()
        local_std = context.drop(context.index[idx - start_idx]).std()
        
        # Magnitude of deviation
        deviation_magnitude = abs(outlier_value - local_mean) / local_std if local_std > 0 else 0
        
        # Impact on trend calculation
        trend_with = np.polyfit(range(len(context)), context.values, deg=1)[0]
        trend_without = np.polyfit(
            range(len(context) - 1), 
            context.drop(context.index[idx - start_idx]).values, 
            deg=1
        )[0]
        trend_impact = abs(trend_with - trend_without)
        
        impact_assessment[idx] = {
            'deviation_magnitude': deviation_magnitude,
            'trend_impact': trend_impact,
            'local_context_std': local_std,
            'treatment_recommendation': recommend_treatment(deviation_magnitude, trend_impact)
        }
    
    return impact_assessment

def recommend_treatment(deviation_magnitude, trend_impact):
    """Recommend treatment based on outlier characteristics."""
    if deviation_magnitude > 5 and trend_impact > 0.1:
        return 'investigate_and_potentially_remove'
    elif deviation_magnitude > 3:
        return 'winsorize_or_cap'
    else:
        return 'keep_with_flag'
