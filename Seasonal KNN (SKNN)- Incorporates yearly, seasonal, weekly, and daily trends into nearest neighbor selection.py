def seasonal_knn_imputation(data, k=5, seasonal_features=True):
    """KNN imputation with seasonal feature engineering."""
    
    # Create seasonal features
    if seasonal_features:
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        
        # Cyclical encoding for periodic features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    # Apply KNN imputation with seasonal features
    feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month', 'quarter']
    imputer = KNNImputer(n_neighbors=k, weights='distance')
    
    # Impute using seasonal context
    data_imputed = imputer.fit_transform(data[feature_cols + ['value']])
    
    return data_imputed[:, -1]  # Return imputed values
