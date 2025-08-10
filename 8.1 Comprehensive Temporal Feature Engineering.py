def create_comprehensive_temporal_features(data, timestamp_col='timestamp', 
                                         include_interactions=True, 
                                         business_calendar=None):
    """
    Create extensive temporal features capturing multiple time scales and patterns.
    """
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    feature_df = data.copy()
    
    # Basic temporal components
    feature_df['year'] = feature_df[timestamp_col].dt.year
    feature_df['quarter'] = feature_df[timestamp_col].dt.quarter
    feature_df['month'] = feature_df[timestamp_col].dt.month
    feature_df['week_of_year'] = feature_df[timestamp_col].dt.isocalendar().week
    feature_df['day_of_year'] = feature_df[timestamp_col].dt.dayofyear
    feature_df['day_of_month'] = feature_df[timestamp_col].dt.day
    feature_df['day_of_week'] = feature_df[timestamp_col].dt.dayofweek
    feature_df['hour'] = feature_df[timestamp_col].dt.hour
    feature_df['minute'] = feature_df[timestamp_col].dt.minute
    
    # Cyclical encoding for periodic features
    cyclical_features = {
        'month': 12,
        'day_of_month': 31,
        'day_of_week': 7,
        'hour': 24,
        'minute': 60,
        'day_of_year': 365
    }
    
    for feature, period in cyclical_features.items():
        if feature in feature_df.columns:
            feature_df[f'{feature}_sin'] = np.sin(2 * np.pi * feature_df[feature] / period)
            feature_df[f'{feature}_cos'] = np.cos(2 * np.pi * feature_df[feature] / period)
    
    # Weekend/weekday indicators
    feature_df['is_weekend'] = (feature_df['day_of_week'] >= 5).astype(int)
    feature_df['is_weekday'] = (feature_df['day_of_week'] < 5).astype(int)
    
    # Month characteristics
    feature_df['is_month_start'] = feature_df[timestamp_col].dt.is_month_start.astype(int)
    feature_df['is_month_end'] = feature_df[timestamp_col].dt.is_month_end.astype(int)
    feature_df['is_quarter_start'] = feature_df[timestamp_col].dt.is_quarter_start.astype(int)
    feature_df['is_quarter_end'] = feature_df[timestamp_col].dt.is_quarter_end.astype(int)
    
    # Business calendar features
    if business_calendar is not None:
        feature_df = add_business_calendar_features(feature_df, timestamp_col, business_calendar)
    
    # Time since epoch (for trend modeling)
    epoch = pd.Timestamp('1970-01-01')
    feature_df['days_since_epoch'] = (feature_df[timestamp_col] - epoch).dt.days
    feature_df['hours_since_epoch'] = (feature_df[timestamp_col] - epoch).dt.total_seconds() / 3600
    
    # Interaction features
    if include_interactions:
        # Hour-day of week interactions (capture weekly patterns within days)
        feature_df['hour_dow_interaction'] = feature_df['hour'] * feature_df['day_of_week']
        
        # Month-year interactions (capture yearly evolution of seasonal patterns)
        feature_df['month_year_interaction'] = feature_df['month'] * (feature_df['year'] - feature_df['year'].min())
        
        # Weekend-hour interactions
        feature_df['weekend_hour_interaction'] = feature_df['is_weekend'] * feature_df['hour']
    
    return feature_df

def add_business_calendar_features(data, timestamp_col, business_calendar):
    """
    Add business-specific calendar features.
    
    business_calendar should be a dict with:
    - 'holidays': list of holiday dates
    - 'business_hours': tuple of (start_hour, end_hour)
    - 'business_days': list of weekdays (0=Monday, 6=Sunday)
    """
    
    # Holiday indicators
    if 'holidays' in business_calendar:
        holiday_dates = pd.to_datetime(business_calendar['holidays'])
        data['is_holiday'] = data[timestamp_col].dt.date.isin(holiday_dates.date).astype(int)
        
        # Days before/after holidays
        data['days_to_holiday'] = np.inf
        data['days_from_holiday'] = np.inf
        
        for _, row in data.iterrows():
            current_date = row[timestamp_col].date()
            
            # Find nearest holiday
            future_holidays = holiday_dates[holiday_dates.date > current_date]
            past_holidays = holiday_dates[holiday_dates.date < current_date]
            
            if len(future_holidays) > 0:
                days_to = (future_holidays.min().date() - current_date).days
                data.loc[row.name, 'days_to_holiday'] = days_to
            
            if len(past_holidays) > 0:
                days_from = (current_date - past_holidays.max().date()).days
                data.loc[row.name, 'days_from_holiday'] = days_from
        
        # Replace inf with reasonable max values
        data['days_to_holiday'] = data['days_to_holiday'].replace(np.inf, 365)
        data['days_from_holiday'] = data['days_from_holiday'].replace(np.inf, 365)
    
    # Business hours indicator
    if 'business_hours' in business_calendar:
        start_hour, end_hour = business_calendar['business_hours']
        data['is_business_hours'] = (
            (data[timestamp_col].dt.hour >= start_hour) & 
            (data[timestamp_col].dt.hour < end_hour)
        ).astype(int)
    
    # Business days indicator
    if 'business_days' in business_calendar:
        business_days_set = set(business_calendar['business_days'])
        data['is_business_day'] = data[timestamp_col].dt.dayofweek.isin(business_days_set).astype(int)
    
    return data
