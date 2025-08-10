def time_series_imputation_cv(data, imputation_method, gap_sizes=[1, 3, 7, 24]):
    """Cross-validation specifically designed for time series imputation."""
    
    results = {}
    
    for gap_size in gap_sizes:
        scores = []
        
        # Create artificial gaps for testing
        for start_idx in range(gap_size, len(data) - gap_size, gap_size * 2):
            # Create gap
            test_data = data.copy()
            true_values = test_data.iloc[start_idx:start_idx + gap_size]
            test_data.iloc[start_idx:start_idx + gap_size] = np.nan
            
            # Apply imputation
            imputed_values = imputation_method(test_data)
            
            # Evaluate imputation quality
            mae = np.mean(np.abs(true_values - imputed_values[start_idx:start_idx + gap_size]))
            rmse = np.sqrt(np.mean((true_values - imputed_values[start_idx:start_idx + gap_size]) ** 2))
            
            scores.append({'mae': mae, 'rmse': rmse})
        
        results[f'gap_size_{gap_size}'] = {
            'mean_mae': np.mean([s['mae'] for s in scores]),
            'mean_rmse': np.mean([s['rmse'] for s in scores]),
            'std_mae': np.std([s['mae'] for s in scores])
        }
    
    return results
