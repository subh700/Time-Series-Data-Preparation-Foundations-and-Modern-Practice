def optimal_box_cox_transformation(data, lambda_range=(-2, 2), method='mle'):
    """
    Find optimal Box-Cox transformation parameter using multiple criteria.
    """
    from scipy import stats
    from scipy.optimize import minimize_scalar
    from statsmodels.tsa.stattools import adfuller
    
    # Remove zeros and negative values for Box-Cox
    positive_data = data[data > 0]
    
    if len(positive_data) < len(data) * 0.8:
        print(f"Warning: {len(data) - len(positive_data)} non-positive values removed")
    
    results = {}
    
    # Method 1: Maximum Likelihood Estimation
    if method in ['mle', 'all']:
        # Using scipy's built-in Box-Cox
        transformed_mle, lambda_mle = stats.boxcox(positive_data)
        
        results['mle'] = {
            'lambda': lambda_mle,
            'transformed_data': transformed_mle,
            'log_likelihood': _boxcox_log_likelihood(positive_data, lambda_mle)
        }
    
    # Method 2: Stationarity-based optimization
    if method in ['stationarity', 'all']:
        def stationarity_objective(lam):
            if lam == 0:
                transformed = np.log(positive_data)
            else:
                transformed = (positive_data**lam - 1) / lam
            
            # Return negative p-value from ADF test (to maximize stationarity)
            adf_result = adfuller(transformed, autolag='AIC')
            return -adf_result[1]  # Negative p-value for minimization
        
        opt_result = minimize_scalar(
            stationarity_objective, 
            bounds=lambda_range, 
            method='bounded'
        )
        
        lambda_stat = opt_result.x
        if lambda_stat == 0:
            transformed_stat = np.log(positive_data)
        else:
            transformed_stat = (positive_data**lambda_stat - 1) / lambda_stat
        
        results['stationarity'] = {
            'lambda': lambda_stat,
            'transformed_data': transformed_stat,
            'adf_p_value': -opt_result.fun
        }
    
    # Method 3: Variance stabilization
    if method in ['variance', 'all']:
        def variance_objective(lam):
            if lam == 0:
                transformed = np.log(positive_data)
            else:
                transformed = (positive_data**lam - 1) / lam
            
            # Calculate coefficient of variation of rolling variance
            rolling_var = pd.Series(transformed).rolling(window=20).var().dropna()
            if len(rolling_var) < 10:
                return float('inf')
            
            cv = rolling_var.std() / rolling_var.mean()
            return cv
        
        opt_result = minimize_scalar(
            variance_objective,
            bounds=lambda_range,
            method='bounded'
        )
        
        lambda_var = opt_result.x
        if lambda_var == 0:
            transformed_var = np.log(positive_data)
        else:
            transformed_var = (positive_data**lambda_var - 1) / lambda_var
        
        results['variance'] = {
            'lambda': lambda_var,
            'transformed_data': transformed_var,
            'variance_stability': opt_result.fun
        }
    
    # Return best result based on method
    if method == 'all':
        return results
    else:
        return results[method]

def _boxcox_log_likelihood(data, lam):
    """Calculate log-likelihood for Box-Cox transformation."""
    n = len(data)
    
    if lam == 0:
        transformed = np.log(data)
    else:
        transformed = (data**lam - 1) / lam
    
    # Log-likelihood calculation
    log_data_sum = np.sum(np.log(data))
    residual_sum_squares = np.sum((transformed - np.mean(transformed))**2)
    
    log_likelihood = (lam - 1) * log_data_sum - (n/2) * np.log(residual_sum_squares/n)
    
    return log_likelihood
