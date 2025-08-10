def empirical_mode_decomposition(data, max_imfs=10, stopping_criterion='sd', sd_threshold=0.2):
    """
    Perform Empirical Mode Decomposition to extract Intrinsic Mode Functions.
    Particularly effective for non-linear and non-stationary time series.
    """
    
    def find_extrema(signal):
        """Find local maxima and minima."""
        from scipy.signal import find_peaks
        
        # Find peaks (maxima)
        peaks, _ = find_peaks(signal)
        
        # Find troughs (minima) by finding peaks of inverted signal
        troughs, _ = find_peaks(-signal)
        
        return peaks, troughs
    
    def cubic_spline_interpolation(x, y, x_new):
        """Cubic spline interpolation."""
        from scipy.interpolate import interp1d
        
        if len(x) < 4:  # Need at least 4 points for cubic spline
            if len(x) >= 2:
                f = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            else:
                return np.full_like(x_new, np.mean(y))
        else:
            f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        
        return f(x_new)
    
    def sifting_process(signal, max_iterations=1000):
        """Extract one Intrinsic Mode Function using sifting process."""
        
        h = signal.copy()
        
        for iteration in range(max_iterations):
            # Find extrema
            peaks, troughs = find_extrema(h)
            
            # Check if we can continue (need at least 2 peaks and 2 troughs)
            if len(peaks) < 2 or len(troughs) < 2:
                break
            
            # Create time indices
            time_indices = np.arange(len(h))
            
            # Interpolate upper and lower envelopes
            upper_envelope = cubic_spline_interpolation(peaks, h[peaks], time_indices)
            lower_envelope = cubic_spline_interpolation(troughs, h[troughs], time_indices)
            
            # Calculate mean envelope
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # Update h
            h_new = h - mean_envelope
            
            # Check stopping criterion
            if stopping_criterion == 'sd':
                # Standard deviation criterion
                sd = np.sum((h - h_new)**2) / np.sum(h**2)
                if sd < sd_threshold:
                    break
            elif stopping_criterion == 'consecutive_extrema':
                # Number of extrema criterion
                new_peaks, new_troughs = find_extrema(h_new)
                total_extrema = len(new_peaks) + len(new_troughs)
                zero_crossings = np.sum(np.diff(np.signbit(h_new)))
                
                if abs(total_extrema - zero_crossings) <= 1:
                    break
            
            h = h_new
        
        return h
    
    # Initialize
    imfs = []
    residual = data.values.copy()
    
    # Extract IMFs
    for i in range(max_imfs):
        # Perform sifting to extract IMF
        imf = sifting_process(residual)
        
        # Check if IMF is valid (should be oscillatory)
        peaks, troughs = find_extrema(imf)
        
        if len(peaks) < 2 and len(troughs) < 2:
            # No more oscillatory components
            break
        
        imfs.append(imf)
        
        # Update residual
        residual = residual - imf
        
        # Check if residual is monotonic (stopping criterion)
        peaks_res, troughs_res = find_extrema(residual)
        if len(peaks_res) + len(troughs_res) < 3:
            break
    
    # Add final residual as trend
    imfs.append(residual)
    
    # Convert to DataFrame for easier handling
    imf_df = pd.DataFrame(
        np.array(imfs).T, 
        index=data.index,
        columns=[f'IMF_{i+1}' for i in range(len(imfs)-1)] + ['Residual']
    )
    
    return imf_df

def emd_based_stationarity_transformation(data, target_imfs='adaptive'):
    """
    Use EMD to achieve stationarity by selecting appropriate IMF combinations.
    """
    
    # Perform EMD
    imf_df = empirical_mode_decomposition(data)
    
    # Analyze stationarity of each component
    stationarity_results = {}
    
    for col in imf_df.columns:
        component = imf_df[col].dropna()
        if len(component) > 10:
            adf_result = adfuller(component, autolag='AIC')
            stationarity_results[col] = {
                'adf_p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'component_data': component
            }
    
    # Select components based on strategy
    if target_imfs == 'adaptive':
        # Select stationary IMFs and difference non-stationary ones
        transformed_components = []
        
        for col, results in stationarity_results.items():
            if results['is_stationary'] or col == 'Residual':
                transformed_components.append(results['component_data'])
            else:
                # Apply differencing to non-stationary IMFs
                differenced = results['component_data'].diff().dropna()
                if len(differenced) > 0:
                    # Extend to match original length
                    extended_diff = pd.Series(
                        np.concatenate([[0], differenced]), 
                        index=results['component_data'].index
                    )
                    transformed_components.append(extended_diff)
        
        # Reconstruct series from transformed components
        min_length = min(len(comp) for comp in transformed_components)
        aligned_components = [comp.iloc[:min_length] for comp in transformed_components]
        
        transformed_series = sum(aligned_components)
        
    elif target_imfs == 'high_frequency_only':
        # Use only high-frequency IMFs (typically first few IMFs)
        n_hf_imfs = min(3, len(imf_df.columns) - 1)  # Exclude residual
        hf_cols = [f'IMF_{i+1}' for i in range(n_hf_imfs)]
        
        transformed_series = imf_df[hf_cols].sum(axis=1)
    
    else:
        # Use specified IMFs
        if isinstance(target_imfs, list):
            available_cols = [col for col in target_imfs if col in imf_df.columns]
            transformed_series = imf_df[available_cols].sum(axis=1)
        else:
            raise ValueError("target_imfs must be 'adaptive', 'high_frequency_only', or list of IMF names")
    
    return {
        'transformed_series': transformed_series,
        'imf_components': imf_df,
        'stationarity_analysis': stationarity_results,
        'transformation_method': target_imfs
    }
