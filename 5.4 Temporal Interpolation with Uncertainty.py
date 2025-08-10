def outlier_interpolation_with_uncertainty(data, outlier_indices, method='spline'):
    """
    Replace outliers with interpolated values while tracking uncertainty.
    """
    from scipy.interpolate import interp1d
    
    treated_data = data.copy()
    uncertainty_estimates = np.zeros(len(data))
    
    for idx in outlier_indices:
        # Create interpolation from non-outlier neighbors
        neighbor_indices = []
        neighbor_values = []
        
        # Look for nearest non-outlier neighbors
        for offset in range(1, min(50, len(data))):
            for direction in [-1, 1]:
                neighbor_idx = idx + direction * offset
                if (0 <= neighbor_idx < len(data) and 
                    neighbor_idx not in outlier_indices):
                    neighbor_indices.append(neighbor_idx)
                    neighbor_values.append(data.iloc[neighbor_idx])
                    
                    if len(neighbor_indices) >= 4:  # Sufficient neighbors
                        break
            if len(neighbor_indices) >= 4:
                break
        
        if len(neighbor_indices) >= 2:
            # Perform interpolation
            if method == 'spline' and len(neighbor_indices) >= 4:
                interp_func = interp1d(neighbor_indices, neighbor_values, 
                                     kind='cubic', bounds_error=False, 
                                     fill_value='extrapolate')
            else:
                interp_func = interp1d(neighbor_indices, neighbor_values, 
                                     kind='linear', bounds_error=False, 
                                     fill_value='extrapolate')
            
            # Replace outlier with interpolated value
            interpolated_value = float(interp_func(idx))
            treated_data.iloc[idx] = interpolated_value
            
            # Estimate uncertainty based on neighbor variability
            neighbor_std = np.std(neighbor_values)
            uncertainty_estimates[idx] = neighbor_std
    
    return {
        'treated_data': treated_data,
        'uncertainty_estimates': uncertainty_estimates,
        'interpolation_method': method
    }
