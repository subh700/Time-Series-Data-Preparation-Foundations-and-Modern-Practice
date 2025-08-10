def optimize_rolling_window_size(data, target_col, max_window=50, 
                                method='information_criterion'):
    """
    Automatically determine optimal rolling window sizes using various criteria.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    results = {}
    
    for window in range(3, min(max_window, len(data))):
        # Create rolling mean feature
        rolling_mean = data[target_col].rolling(window=window).mean()
        
        # Create lagged version for prediction
        X = rolling_mean.shift(1).dropna()
        y = data[target_col].loc[X.index]
        
        if len(X) < 10:
            continue
        
        if method == 'information_criterion':
            # Fit simple linear model
            model = LinearRegression()
            model.fit(X.values.reshape(-1, 1), y.values)
            predictions = model.predict(X.values.reshape(-1, 1))
            
            # Calculate AIC
            mse = mean_squared_error(y, predictions)
            n = len(y)
            aic = n * np.log(mse) + 2 * 2  # 2 parameters (slope, intercept)
            
            results[window] = {
                'aic': aic,
                'mse': mse,
                'observations': n
            }
        
        elif method == 'autocorrelation_decay':
            # Measure how well the rolling mean preserves autocorrelation
            from statsmodels.tsa.stattools import acf
            
            original_acf = acf(data[target_col].dropna(), nlags=10, missing='drop')
            rolling_acf = acf(rolling_mean.dropna(), nlags=10, missing='drop')
            
            # Calculate correlation between ACF patterns
            acf_correlation = np.corrcoef(original_acf[1:], rolling_acf[1:])[0, 1]
            
            results[window] = {
                'acf_preservation': acf_correlation if not np.isnan(acf_correlation) else 0,
                'observations': len(rolling_mean.dropna())
            }
    
    # Select optimal window
    if method == 'information_criterion':
        optimal_window = min(results.keys(), key=lambda k: results[k]['aic'])
        selection_criterion = 'minimum_aic'
    elif method == 'autocorrelation_decay':
        optimal_window = max(results.keys(), key=lambda k: results[k]['acf_preservation'])
        selection_criterion = 'maximum_acf_preservation'
    
    return {
        'optimal_window': optimal_window,
        'selection_criterion': selection_criterion,
        'all_results': results,
        'window_performance': results[optimal_window]
    }
