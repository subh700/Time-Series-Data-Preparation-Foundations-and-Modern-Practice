class AdvancedEMD:
    """
    Advanced Empirical Mode Decomposition with ensemble methods 
    and noise reduction techniques.
    """
    
    def __init__(self, method='eemd', n_ensembles=100, noise_std=0.2, max_imfs=None):
        self.method = method  # 'emd', 'eemd', 'ceemdan', 'eawd'
        self.n_ensembles = n_ensembles
        self.noise_std = noise_std
        self.max_imfs = max_imfs
        
    def decompose(self, data, sampling_rate=1.0):
        """
        Perform advanced EMD decomposition using specified method.
        """
        series = data if isinstance(data, np.ndarray) else data.values
        
        if self.method == 'emd':
            return self._standard_emd(series)
        elif self.method == 'eemd':
            return self._ensemble_emd(series)
        elif self.method == 'ceemdan':
            return self._ceemdan(series)
        elif self.method == 'eawd':
            return self._empirical_adaptive_wavelet_decomposition(series, sampling_rate)
        else:
            raise ValueError(f"Unknown EMD method: {self.method}")
    
    def _ensemble_emd(self, series):
        """
        Ensemble Empirical Mode Decomposition (EEMD) for noise reduction.
        """
        
        all_imfs = []
        
        # Generate ensemble of decompositions
        for i in range(self.n_ensembles):
            # Add white noise
            noise = np.random.normal(0, self.noise_std * np.std(series), len(series))
            noisy_series = series + noise
            
            # Perform standard EMD on noisy series
            imfs = self._standard_emd(noisy_series)
            all_imfs.append(imfs)
        
        # Average IMFs across ensemble
        n_imfs = min(len(imf) for imf in all_imfs)
        averaged_imfs = []
        
        for imf_idx in range(n_imfs):
            imf_stack = np.array([imfs[imf_idx] for imfs in all_imfs])
            averaged_imf = np.mean(imf_stack, axis=0)
            averaged_imfs.append(averaged_imf)
        
        return averaged_imfs
    
    def _ceemdan(self, series):
        """
        Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN).
        """
        
        imfs = []
        residual = series.copy()
        
        for imf_idx in range(self.max_imfs or 10):
            # Generate noise realizations
            noise_realizations = []
            for i in range(self.n_ensembles):
                noise = np.random.normal(0, self.noise_std * np.std(series), len(series))
                noise_realizations.append(noise)
            
            # Calculate modes from noise
            if imf_idx == 0:
                # First mode: add noise to signal
                mode_sum = np.zeros(len(series))
                for noise in noise_realizations:
                    noisy_signal = residual + noise
                    first_imf = self._extract_single_imf(noisy_signal)
                    mode_sum += first_imf
                
                current_imf = mode_sum / self.n_ensembles
            else:
                # Subsequent modes: add modes of noise
                mode_sum = np.zeros(len(series))
                for noise in noise_realizations:
                    noise_imfs = self._standard_emd(noise)
                    if len(noise_imfs) > imf_idx - 1:
                        mode_sum += noise_imfs[imf_idx - 1]
                
                noise_mode = mode_sum / self.n_ensembles
                current_imf = self._extract_single_imf(residual + noise_mode)
            
            imfs.append(current_imf)
            residual = residual - current_imf
            
            # Check stopping criteria
            if self._stopping_criterion(residual):
                break
        
        # Add final residual
        imfs.append(residual)
        
        return imfs
    
    def _empirical_adaptive_wavelet_decomposition(self, series, sampling_rate):
        """
        Empirical Adaptive Wavelet Decomposition (EAWD) combining EMD and EWT.
        Based on recent research combining EMD strengths with wavelet precision.
        """
        
        # Step 1: Perform standard EMD to get IMFs
        emd_imfs = self._standard_emd(series)
        
        # Step 2: Analyze spectral content of each IMF
        imf_spectra = []
        for imf in emd_imfs:
            spectrum = np.abs(np.fft.fft(imf))
            frequencies = np.fft.fftfreq(len(imf), d=1/sampling_rate)
            imf_spectra.append((frequencies[:len(frequencies)//2], 
                              spectrum[:len(spectrum)//2]))
        
        # Step 3: Design adaptive filter bank based on IMF spectra
        filter_bank = self._design_adaptive_filter_bank(imf_spectra, sampling_rate)
        
        # Step 4: Apply adaptive filters to original signal
        adaptive_components = []
        for filt in filter_bank:
            filtered_component = self._apply_filter(series, filt, sampling_rate)
            adaptive_components.append(filtered_component)
        
        return adaptive_components
    
    def _design_adaptive_filter_bank(self, imf_spectra, sampling_rate):
        """Design adaptive filter bank based on IMF spectral characteristics."""
        
        # Find frequency boundaries based on IMF spectral peaks
        boundaries = []
        
        for freqs, spectrum in imf_spectra:
            # Find dominant frequency in this IMF
            peak_idx = np.argmax(spectrum)
            peak_freq = freqs[peak_idx]
            boundaries.append(peak_freq)
        
        # Sort boundaries and create filter specifications
        boundaries = sorted(set(boundaries))
        filter_specs = []
        
        for i in range(len(boundaries)):
            if i == 0:
                # Low-pass filter for first component
                filter_specs.append({
                    'type': 'lowpass',
                    'cutoff': boundaries[i],
                    'order': 4
                })
            elif i == len(boundaries) - 1:
                # High-pass filter for last component
                filter_specs.append({
                    'type': 'highpass',
                    'cutoff': boundaries[i-1],
                    'order': 4
                })
            else:
                # Band-pass filter for middle components
                filter_specs.append({
                    'type': 'bandpass',
                    'low_cutoff': boundaries[i-1],
                    'high_cutoff': boundaries[i],
                    'order': 4
                })
        
        return filter_specs
    
    def _apply_filter(self, signal, filter_spec, sampling_rate):
        """Apply digital filter based on specifications."""
        from scipy.signal import butter, filtfilt
        
        nyquist = sampling_rate / 2
        
        if filter_spec['type'] == 'lowpass':
            b, a = butter(filter_spec['order'], 
                         filter_spec['cutoff'] / nyquist, 
                         btype='low')
        elif filter_spec['type'] == 'highpass':
            b, a = butter(filter_spec['order'], 
                         filter_spec['cutoff'] / nyquist, 
                         btype='high')
        elif filter_spec['type'] == 'bandpass':
            b, a = butter(filter_spec['order'], 
                         [filter_spec['low_cutoff'] / nyquist,
                          filter_spec['high_cutoff'] / nyquist], 
                         btype='band')
        
        # Apply zero-phase filtering
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def _standard_emd(self, series):
        """Standard EMD implementation with sifting process."""
        
        imfs = []
        residual = series.copy()
        
        for imf_count in range(self.max_imfs or 10):
            # Extract single IMF through sifting
            imf = self._extract_single_imf(residual)
            
            # Check if this is a valid IMF
            if not self._is_valid_imf(imf):
                break
            
            imfs.append(imf)
            residual = residual - imf
            
            # Check stopping criteria
            if self._stopping_criterion(residual):
                break
        
        # Add final residual
        imfs.append(residual)
        
        return imfs
    
    def _extract_single_imf(self, signal, max_iterations=1000):
        """Extract single IMF using sifting process."""
        
        h = signal.copy()
        
        for iteration in range(max_iterations):
            # Find extrema
            maxima_idx, minima_idx = self._find_extrema(h)
            
            # Check if we can continue
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            
            # Create envelopes
            upper_envelope = self._interpolate_envelope(maxima_idx, h[maxima_idx], len(h))
            lower_envelope = self._interpolate_envelope(minima_idx, h[minima_idx], len(h))
            
            # Calculate mean envelope
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # Update h
            h_new = h - mean_envelope
            
            # Check sifting stopping criterion
            if self._sifting_stopping_criterion(h, h_new):
                break
            
            h = h_new
        
        return h
    
    def _find_extrema(self, signal):
        """Find local maxima and minima in signal."""
        from scipy.signal import find_peaks
        
        # Find maxima
        maxima_idx, _ = find_peaks(signal)
        
        # Find minima (peaks of inverted signal)
        minima_idx, _ = find_peaks(-signal)
        
        return maxima_idx, minima_idx
    
    def _interpolate_envelope(self, extrema_idx, extrema_values, signal_length):
        """Interpolate envelope using cubic splines."""
        from scipy.interpolate import interp1d
        
        if len(extrema_idx) < 2:
            return np.zeros(signal_length)
        
        # Add boundary points for better interpolation
        extended_idx = np.concatenate([[0], extrema_idx, [signal_length - 1]])
        extended_values = np.concatenate([[extrema_values[0]], extrema_values, [extrema_values[-1]]])
        
        # Create interpolation function
        f = interp1d(extended_idx, extended_values, kind='cubic', 
                    bounds_error=False, fill_value='extrapolate')
        
        # Generate envelope
        envelope = f(np.arange(signal_length))
        
        return envelope
    
    def _sifting_stopping_criterion(self, h_old, h_new, threshold=0.2):
        """Check sifting stopping criterion using standard deviation."""
        
        sd = np.sum((h_old - h_new) ** 2) / np.sum(h_old ** 2)
        return sd < threshold
    
    def _is_valid_imf(self, imf, tolerance=1):
        """Check if extracted component is a valid IMF."""
        
        # Find extrema
        maxima_idx, minima_idx = self._find_extrema(imf)
        
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(imf)))
        
        # IMF criteria: number of extrema and zero crossings should differ by at most 1
        total_extrema = len(maxima_idx) + len(minima_idx)
        
        return abs(total_extrema - zero_crossings) <= tolerance
    
    def _stopping_criterion(self, residual):
        """Check global stopping criterion for EMD."""
        
        # Stop if residual is monotonic
        maxima_idx, minima_idx = self._find_extrema(residual)
        
        return len(maxima_idx) + len(minima_idx) < 3
