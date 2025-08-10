from statsmodels.tsa.statespace.sarimax import SARIMAX

def state_space_imputation(series, order=(1,1,1)):
    """Use SARIMAX model to impute missing values."""
    # Fit model on available data
    model = SARIMAX(series, order=order, enforce_stationarity=False)
    results = model.fit(disp=False)
    
    # Impute missing values using model predictions
    imputed_series = series.copy()
    missing_mask = series.isnull()
    
    for idx in series[missing_mask].index:
        # Use model to predict at missing timestamp
        prediction = results.get_prediction(start=idx, end=idx)
        imputed_series.loc[idx] = prediction.predicted_mean.iloc[0]
    
    return imputed_series
