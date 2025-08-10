def fractional_differencing(data, d=0.5, threshold=1e-5):
    """
    Apply fractional differencing to achieve stationarity while preserving memory.
    
    The fractional difference operator is defined as:
    (1-L)^d X_t where L is the lag operator and d can be non-integer.
    """
    
    def get_weights(d, size):
        """Calculate fractional differencing weights."""
        weights = np.zeros(size)
        weights[0] = 1
        
        for k in range(1, size):
            weights[k] = -weights[k-1] * (d - k + 1) / k
            
            # Truncate weights below threshold
            if abs(weights[k]) < threshold:
                return weights[:k]
        
        return weights
    
    # Calculate weights
    weights = get_weights(d, len(data))
    
    # Apply fractional differencing
    fractionally_differenced = np.zeros(len(data))
    
    for i in range(len(weights), len(data)):
        fractionally_differenced[i] = np.dot(weights, data[i-len(weights)+1:i+1][::-1])
    
    # Return as pandas Series with appropriate index
    result = pd.Series(
        fractionally_differenced[len(weights):], 
        index=data.index[len(weights):]
    )
    
    return result, weights

def adaptive_differencing(data, max_differences=3, seasonal_period=None):
    """
    Automatically determine optimal differencing order using information criteria.
    """
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    
    results = {}
    current_data = data.copy()
    
    # Test regular differencing
    for d in range(max_differences + 1):
        try:
            # Apply d-order differencing
            if d == 0:
                test_data = current_data
            else:
                test_data = current_data.diff(periods=1).dropna()
                current_data = test_data
            
            # Stationarity test
            adf_result = adfuller(test_data.dropna(), autolag='AIC')
            
            # Fit ARIMA model to get information criteria
            try:
                model = ARIMA(test_data.dropna(), order=(1, 0, 1))
                fitted = model.fit()
                aic = fitted.aic
                bic = fitted.bic
            except:
                aic = float('inf')
                bic = float('inf')
            
            results[f'regular_d_{d}'] = {
                'data': test_data,
                'adf_statistic': adf_result[0],
                'adf_p_value': adf_result[1],
                'aic': aic,
                'bic': bic,
                'is_stationary': adf_result[1] < 0.05
            }
            
        except Exception as e:
            results[f'regular_d_{d}'] = {
                'error': str(e)
            }
    
    # Test seasonal differencing if period specified
    if seasonal_period:
        current_data = data.copy()
        
        for D in range(2):  # Usually 0 or 1 seasonal difference is sufficient
            try:
                if D == 0:
                    test_data = current_data
                else:
                    test_data = current_data.diff(periods=seasonal_period).dropna()
                
                # Apply regular differencing on seasonally differenced data
                for d in range(2):
                    if d > 0:
                        final_data = test_data.diff(periods=1).dropna()
                    else:
                        final_data = test_data
                    
                    if len(final_data) < 10:
                        continue
                    
                    adf_result = adfuller(final_data.dropna(), autolag='AIC')
                    
                    try:
                        model = ARIMA(final_data.dropna(), order=(1, 0, 1))
                        fitted = model.fit()
                        aic = fitted.aic
                        bic = fitted.bic
                    except:
                        aic = float('inf')
                        bic = float('inf')
                    
                    results[f'seasonal_D_{D}_d_{d}'] = {
                        'data': final_data,
                        'adf_statistic': adf_result[0],
                        'adf_p_value': adf_result[1],
                        'aic': aic,
                        'bic': bic,
                        'is_stationary': adf_result[1] < 0.05
                    }
                    
            except Exception as e:
                results[f'seasonal_D_{D}_d_0'] = {
                    'error': str(e)
                }
    
    # Find optimal differencing based on criteria
    best_config = find_optimal_differencing(results)
    
    return results, best_config

def find_optimal_differencing(results):
    """Find optimal differencing configuration based on multiple criteria."""
    
    # Filter valid results
    valid_results = {k: v for k, v in results.items() 
                    if 'error' not in v and 'is_stationary' in v}
    
    if not valid_results:
        return None
    
    # Scoring function combining stationarity, AIC, and parsimony
    def score_configuration(config_name, config_data):
        score = 0
        
        # Stationarity bonus
        if config_data['is_stationary']:
            score += 100
        
        # Lower AIC is better
        if config_data['aic'] != float('inf'):
            score -= config_data['aic'] / 100  # Normalize AIC contribution
        
        # Parsimony bonus (prefer fewer differences)
        d_count = config_name.count('_d_') + config_name.count('_D_')
        score -= d_count * 10  # Penalty for complexity
        
        # Strong stationarity bonus (lower p-value)
        if config_data['adf_p_value'] < 0.01:
            score += 50
        elif config_data['adf_p_value'] < 0.05:
            score += 25
        
        return score
    
    # Score all configurations
    scored_configs = {}
    for name, data in valid_results.items():
        scored_configs[name] = {
            'score': score_configuration(name, data),
            'config_data': data
        }
    
    # Find best configuration
    best_name = max(scored_configs.keys(), key=lambda k: scored_configs[k]['score'])
    
    return {
        'best_configuration': best_name,
        'score': scored_configs[best_name]['score'],
        'transformed_data': scored_configs[best_name]['config_data']['data'],
        'all_scores': {k: v['score'] for k, v in scored_configs.items()}
    }
