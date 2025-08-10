def garch_volatility_outliers(returns, threshold_multiplier=3.0):
    """
    Detect volatility outliers using GARCH model residuals.
    Particularly useful for financial time series.
    """
    from arch import arch_model
    
    # Fit GARCH(1,1) model
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_results = garch_model.fit(show_warning=False)
    
    # Extract standardized residuals
    standardized_residuals = garch_results.std_resid
    
    # Detect outliers based on standardized residuals
    threshold = threshold_multiplier
    outliers = np.abs(standardized_residuals) > threshold
    
    return {
        'outlier_mask': outliers,
        'standardized_residuals': standardized_residuals,
        'conditional_volatility': garch_results.conditional_volatility,
        'model_params': garch_results.params
    }
