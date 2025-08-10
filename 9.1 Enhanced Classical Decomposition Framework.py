class EnhancedClassicalDecomposition:
    """
    Advanced classical decomposition with robust estimation and 
    multiple decomposition models.
    """
    
    def __init__(self, period=None, model='auto', robust=True):
        self.period = period
        self.model = model  # 'additive', 'multiplicative', 'auto'
        self.robust = robust
        
    def decompose(self, data, timestamp_col=None):
        """
        Perform enhanced classical decomposition with automatic model selection.
        """
        series = data if isinstance(data, pd.Series) else data.iloc[:, 0]
        
        # Automatic period detection if not specified
        if self.period is None:
            self.period = self._detect_period(series)
        
        # Automatic model selection if not specified
        if self.model == 'auto':
            decomposition_model = self._select_model(series)
        else:
            decomposition_model = self.model
        
        # Perform decomposition based on selected model
        if decomposition_model == 'additive':
            decomposition = self._additive_decomposition(series)
        elif decomposition_model == 'multiplicative':
            decomposition = self._multiplicative_decomposition(series)
        else:  # pseudo-additive or mixed
            decomposition = self._pseudo_additive_decomposition(series)
        
        # Add diagnostic information
        decomposition['model_used'] = decomposition_model
        decomposition['period_detected'] = self.period
        decomposition['decomposition_quality'] = self._assess_decomposition_quality(
            series, decomposition
        )
        
        return decomposition
    
    def _detect_period(self, series, max_period=None):
        """Automatically detect the dominant period using multiple methods."""
        from scipy.signal import find_peaks
        from scipy.fft import fft, fftfreq
        
        if max_period is None:
            max_period = len(series) // 4
        
        # Method 1: Autocorrelation-based detection
        autocorr_period = self._autocorr_period_detection(series, max_period)
        
        # Method 2: FFT-based detection
        fft_period = self._fft_period_detection(series, max_period)
        
        # Method 3: Peak-based detection (for irregular patterns)
        peak_period = self._peak_period_detection(series, max_period)
        
        # Consensus-based period selection
        periods = [autocorr_period, fft_period, peak_period]
        period_weights = [0.4, 0.4, 0.2]  # Prefer autocorr and FFT methods
        
        # Weighted voting for period selection
        period_scores = {}
        for i, period in enumerate(periods):
            if period is not None:
                weight = period_weights[i]
                if period in period_scores:
                    period_scores[period] += weight
                else:
                    period_scores[period] = weight
        
        if period_scores:
            detected_period = max(period_scores.keys(), key=lambda k: period_scores[k])
        else:
            # Fallback to common periods
            detected_period = min(12, len(series) // 8)  # Default to monthly if unclear
        
        return max(2, detected_period)  # Ensure minimum period of 2
    
    def _autocorr_period_detection(self, series, max_period):
        """Detect period using autocorrelation function."""
        from statsmodels.tsa.stattools import acf
        
        # Calculate autocorrelation
        autocorr = acf(series.dropna(), nlags=max_period, missing='drop')
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(
            autocorr[1:],  # Skip lag 0
            height=0.1,    # Minimum correlation
            distance=2     # Minimum distance between peaks
        )
        
        if len(peaks) > 0:
            # Return the lag with highest autocorrelation
            best_peak = peaks[np.argmax(autocorr[peaks + 1])] + 1
            return best_peak
        
        return None
    
    def _fft_period_detection(self, series, max_period):
        """Detect period using Fourier transform."""
        from scipy.fft import fft, fftfreq
        
        # Compute FFT
        fft_values = fft(series.dropna().values)
        frequencies = fftfreq(len(fft_values))
        
        # Find dominant frequency (excluding DC component)
        magnitudes = np.abs(fft_values[1:len(fft_values)//2])
        dominant_freq_idx = np.argmax(magnitudes) + 1
        
        dominant_freq = frequencies[dominant_freq_idx]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
            if 2 <= period <= max_period:
                return period
        
        return None
    
    def _peak_period_detection(self, series, max_period):
        """Detect period based on peak patterns."""
        from scipy.signal import find_peaks
        
        # Find peaks in the series
        peaks, _ = find_peaks(series.values, height=series.mean())
        
        if len(peaks) > 2:
            # Calculate average distance between peaks
            peak_distances = np.diff(peaks)
            avg_distance = np.median(peak_distances)
            
            if 2 <= avg_distance <= max_period:
                return int(avg_distance)
        
        return None
    
    def _select_model(self, series):
        """Automatically select decomposition model based on data characteristics."""
        
        # Test for multiplicative patterns
        # If seasonal variation increases with trend level, use multiplicative
        trend_estimate = series.rolling(window=max(4, self.period)).mean()
        detrended = series - trend_estimate
        
        # Calculate seasonal variation at different trend levels
        trend_quantiles = trend_estimate.quantile([0.25, 0.75])
        
        low_trend_mask = trend_estimate <= trend_quantiles[0.25]
        high_trend_mask = trend_estimate >= trend_quantiles[0.75]
        
        low_trend_variation = detrended[low_trend_mask].std()
        high_trend_variation = detrended[high_trend_mask].std()
        
        # If variation increases significantly with trend level, use multiplicative
        if high_trend_variation > 1.5 * low_trend_variation:
            return 'multiplicative'
        else:
            return 'additive'
    
    def _additive_decomposition(self, series):
        """Perform robust additive decomposition."""
        
        # Step 1: Estimate trend using centered moving average
        if self.robust:
            trend = self._robust_trend_estimation(series)
        else:
            trend = series.rolling(window=self.period, center=True).mean()
        
        # Step 2: Detrend the series
        detrended = series - trend
        
        # Step 3: Estimate seasonal component
        seasonal = self._estimate_seasonal_component(detrended, model='additive')
        
        # Step 4: Calculate residual
        residual = series - trend - seasonal
        
        return {
            'original': series,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'model': 'additive'
        }
    
    def _multiplicative_decomposition(self, series):
        """Perform robust multiplicative decomposition."""
        
        # Ensure positive values for multiplicative model
        min_val = series.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            series_adj = series + offset
        else:
            series_adj = series
            offset = 0
        
        # Step 1: Estimate trend
        if self.robust:
            trend = self._robust_trend_estimation(series_adj)
        else:
            trend = series_adj.rolling(window=self.period, center=True).mean()
        
        # Step 2: Detrend by division
        detrended = series_adj / trend
        
        # Step 3: Estimate seasonal component
        seasonal = self._estimate_seasonal_component(detrended, model='multiplicative')
        
        # Step 4: Calculate residual
        residual = series_adj / (trend * seasonal)
        
        # Adjust back if offset was applied
        if offset > 0:
            trend = trend - offset
            # Note: seasonal and residual remain as ratios
        
        return {
            'original': series,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'model': 'multiplicative'
        }
    
    def _robust_trend_estimation(self, series):
        """Robust trend estimation using iterative filtering."""
        
        # Use Henderson moving average for robust trend estimation
        window = max(13, self.period)  # Ensure odd window size
        if window % 2 == 0:
            window += 1
        
        # Apply Henderson weights
        henderson_weights = self._henderson_weights(window)
        
        # Convolve with Henderson weights
        trend = np.convolve(series.values, henderson_weights, mode='same')
        
        # Handle boundary effects using local polynomial fitting
        boundary_size = window // 2
        
        # Left boundary
        for i in range(boundary_size):
            local_data = series.iloc[:window].values
            local_trend = np.polyfit(range(window), local_data, deg=2)
            trend[i] = np.polyval(local_trend, i)
        
        # Right boundary
        for i in range(len(series) - boundary_size, len(series)):
            local_data = series.iloc[-window:].values
            local_trend = np.polyfit(range(window), local_data, deg=2)
            trend[i] = np.polyval(local_trend, window - (len(series) - i))
        
        return pd.Series(trend, index=series.index)
    
    def _henderson_weights(self, n):
        """Generate Henderson moving average weights."""
        
        # Henderson weights formula
        m = (n - 1) // 2
        weights = np.zeros(n)
        
        for j in range(-m, m + 1):
            if abs(j) <= m:
                numerator = 315 * ((m + 1)**4 - 5*(m + 1)**2*j**2 + 4*j**4)
                denominator = 8 * (m + 1) * ((m + 1)**2 - 1) * (4*(m + 1)**2 - 1) * (4*(m + 1)**2 - 9)
                weights[j + m] = numerator / denominator
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _estimate_seasonal_component(self, detrended_series, model='additive'):
        """Estimate seasonal component using robust averaging."""
        
        seasonal_averages = np.zeros(self.period)
        
        # Calculate seasonal averages for each period position
        for i in range(self.period):
            period_values = []
            
            # Collect all values at this seasonal position
            for j in range(i, len(detrended_series), self.period):
                if not pd.isna(detrended_series.iloc[j]):
                    period_values.append(detrended_series.iloc[j])
            
            if period_values:
                if self.robust:
                    # Use median for robust estimation
                    seasonal_averages[i] = np.median(period_values)
                else:
                    seasonal_averages[i] = np.mean(period_values)
        
        # Ensure seasonal component sums to zero (additive) or averages to 1 (multiplicative)
        if model == 'additive':
            seasonal_averages = seasonal_averages - np.mean(seasonal_averages)
        else:  # multiplicative
            seasonal_averages = seasonal_averages / np.mean(seasonal_averages)
        
        # Expand to full series length
        seasonal_full = np.tile(seasonal_averages, len(detrended_series) // self.period + 1)
        seasonal_full = seasonal_full[:len(detrended_series)]
        
        return pd.Series(seasonal_full, index=detrended_series.index)
    
    def _assess_decomposition_quality(self, original, decomposition):
        """Assess the quality of decomposition."""
        
        # Reconstruction error
        if decomposition['model'] == 'additive':
            reconstructed = (decomposition['trend'] + 
                           decomposition['seasonal']).dropna()
        else:  # multiplicative
            reconstructed = (decomposition['trend'] * 
                           decomposition['seasonal']).dropna()
        
        # Align indices for comparison
        common_idx = original.index.intersection(reconstructed.index)
        orig_aligned = original.loc[common_idx]
        recon_aligned = reconstructed.loc[common_idx]
        
        # Calculate reconstruction metrics
        mae = np.mean(np.abs(orig_aligned - recon_aligned))
        rmse = np.sqrt(np.mean((orig_aligned - recon_aligned) ** 2))
        mape = np.mean(np.abs((orig_aligned - recon_aligned) / orig_aligned)) * 100
        
        # R-squared for reconstruction quality
        ss_res = np.sum((orig_aligned - recon_aligned) ** 2)
        ss_tot = np.sum((orig_aligned - np.mean(orig_aligned)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'reconstruction_mae': mae,
            'reconstruction_rmse': rmse,
            'reconstruction_mape': mape,
            'reconstruction_r2': r_squared,
            'residual_autocorr': self._residual_autocorrelation(decomposition['residual'])
        }
    
    def _residual_autocorrelation(self, residuals):
        """Check for remaining autocorrelation in residuals."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        clean_residuals = residuals.dropna()
        if len(clean_residuals) > 10:
            lb_test = acorr_ljungbox(clean_residuals, lags=min(10, len(clean_residuals)//4))
            return lb_test['lb_pvalue'].iloc[0]  # Return p-value for first lag
        else:
            return np.nan
