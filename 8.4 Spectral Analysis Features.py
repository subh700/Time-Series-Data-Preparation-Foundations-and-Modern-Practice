class FrequencyDomainFeatureEngineer:
    """
    Extract frequency domain features using spectral analysis techniques.
    """
    
    def __init__(self, sampling_rate=1.0, nperseg=None, noverlap=None):
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg
        self.noverlap = noverlap
        
    def create_frequency_features(self, data, target_col, window_type='hann'):
        """
        Create comprehensive frequency domain features.
        """
        from scipy import signal
        from scipy.fft import fft, fftfreq
        
        series = data[target_col].dropna()
        features = pd.DataFrame(index=data.index)
        
        # Basic FFT Features
        fft_features = self._extract_fft_features(series)
        features = pd.concat([features, fft_features], axis=1)
        
        # Power Spectral Density Features
        psd_features = self._extract_psd_features(series, window_type)
        features = pd.concat([features, psd_features], axis=1)
        
        # Spectral Statistics
        spectral_stats = self._extract_spectral_statistics(series)
        features = pd.concat([features, spectral_stats], axis=1)
        
        # Wavelet Features
        wavelet_features = self._extract_wavelet_features(series)
        features = pd.concat([features, wavelet_features], axis=1)
        
        return features
    
    def _extract_fft_features(self, series):
        """Extract features from Fast Fourier Transform."""
        
        # Compute FFT
        fft_values = fft(series.values)
        fft_freqs = fftfreq(len(series), d=1/self.sampling_rate)
        
        # Keep only positive frequencies
        pos_mask = fft_freqs > 0
        fft_magnitudes = np.abs(fft_values[pos_mask])
        fft_phases = np.angle(fft_values[pos_mask])
        freqs = fft_freqs[pos_mask]
        
        features = pd.DataFrame(index=series.index)
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitudes)
        features['fft_dominant_frequency'] = freqs[dominant_freq_idx]
        features['fft_dominant_magnitude'] = fft_magnitudes[dominant_freq_idx]
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)
        features['fft_spectral_centroid'] = spectral_centroid
        
        # Spectral spread (weighted standard deviation around centroid)
        spectral_spread = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * fft_magnitudes) / 
            np.sum(fft_magnitudes)
        )
        features['fft_spectral_spread'] = spectral_spread
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(fft_magnitudes ** 2)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0][0]
        features['fft_spectral_rolloff'] = freqs[rolloff_idx]
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(fft_magnitudes + 1e-10)))
        arithmetic_mean = np.mean(fft_magnitudes)
        features['fft_spectral_flatness'] = geometric_mean / arithmetic_mean
        
        # Zero crossing rate (in frequency domain - phase changes)
        phase_diff = np.diff(fft_phases)
        zero_crossings = np.sum(np.diff(np.sign(phase_diff)) != 0)
        features['fft_zero_crossing_rate'] = zero_crossings / len(phase_diff)
        
        # Peak count (number of local maxima in spectrum)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(fft_magnitudes, height=np.mean(fft_magnitudes))
        features['fft_peak_count'] = len(peaks)
        
        # Broadcast scalar features to match series length
        for col in features.columns:
            features[col] = features[col].fillna(features[col].iloc[0])
        
        return features
    
    def _extract_psd_features(self, series, window_type='hann'):
        """Extract features from Power Spectral Density."""
        from scipy import signal
        
        # Compute PSD using Welch's method
        nperseg = self.nperseg or min(256, len(series) // 4)
        noverlap = self.noverlap or nperseg // 2
        
        freqs, psd = signal.welch(
            series.values, 
            fs=self.sampling_rate,
            window=window_type,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        features = pd.DataFrame(index=series.index)
        
        # Total power
        total_power = np.trapz(psd, freqs)
        features['psd_total_power'] = total_power
        
        # Power in frequency bands
        # Low frequency (0-25% of Nyquist)
        low_freq_mask = freqs <= (freqs[-1] * 0.25)
        low_freq_power = np.trapz(psd[low_freq_mask], freqs[low_freq_mask])
        features['psd_low_freq_power'] = low_freq_power / total_power
        
        # Medium frequency (25-75% of Nyquist)
        med_freq_mask = (freqs > (freqs[-1] * 0.25)) & (freqs <= (freqs[-1] * 0.75))
        med_freq_power = np.trapz(psd[med_freq_mask], freqs[med_freq_mask])
        features['psd_med_freq_power'] = med_freq_power / total_power
        
        # High frequency (75-100% of Nyquist)
        high_freq_mask = freqs > (freqs[-1] * 0.75)
        high_freq_power = np.trapz(psd[high_freq_mask], freqs[high_freq_mask])
        features['psd_high_freq_power'] = high_freq_power / total_power
        
        # Peak frequency
        peak_freq_idx = np.argmax(psd)
        features['psd_peak_frequency'] = freqs[peak_freq_idx]
        features['psd_peak_power'] = psd[peak_freq_idx]
        
        # Spectral entropy
        psd_normalized = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))
        features['psd_spectral_entropy'] = spectral_entropy
        
        # Broadcast scalar features
        for col in features.columns:
            features[col] = features[col].fillna(features[col].iloc[0])
        
        return features
    
    def _extract_spectral_statistics(self, series):
        """Extract statistical measures from frequency domain."""
        from scipy.fft import fft
        
        fft_values = fft(series.values)
        magnitudes = np.abs(fft_values)
        
        features = pd.DataFrame(index=series.index)
        
        # Spectral statistics
        features['spectral_mean'] = np.mean(magnitudes)
        features['spectral_std'] = np.std(magnitudes)
        features['spectral_skewness'] = self._calculate_skewness(magnitudes)
        features['spectral_kurtosis'] = self._calculate_kurtosis(magnitudes)
        
        # Spectral energy
        features['spectral_energy'] = np.sum(magnitudes ** 2)
        
        # Spectral flux (rate of change in frequency domain)
        if len(series) > 1:
            prev_magnitudes = np.abs(fft(np.roll(series.values, 1)))
            spectral_flux = np.sum((magnitudes - prev_magnitudes) ** 2)
            features['spectral_flux'] = spectral_flux
        
        # Broadcast scalar features
        for col in features.columns:
            features[col] = features[col].fillna(features[col].iloc[0])
        
        return features
    
    def _extract_wavelet_features(self, series):
        """Extract features using wavelet transform."""
        try:
            import pywt
            
            # Perform discrete wavelet transform
            coeffs = pywt.wavedec(series.values, 'db4', level=4)
            
            features = pd.DataFrame(index=series.index)
            
            # Energy in each frequency band
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff ** 2)
                features[f'wavelet_energy_level_{i}'] = energy
                
                # Statistical measures of wavelet coefficients
                features[f'wavelet_mean_level_{i}'] = np.mean(coeff)
                features[f'wavelet_std_level_{i}'] = np.std(coeff)
                features[f'wavelet_max_level_{i}'] = np.max(np.abs(coeff))
            
            # Total wavelet energy
            total_energy = sum(np.sum(c ** 2) for c in coeffs)
            features['wavelet_total_energy'] = total_energy
            
            # Relative energy in each band
            for i in range(len(coeffs)):
                features[f'wavelet_relative_energy_level_{i}'] = (
                    features[f'wavelet_energy_level_{i}'] / total_energy
                )
            
            # Broadcast scalar features
            for col in features.columns:
                features[col] = features[col].fillna(features[col].iloc[0])
            
            return features
            
        except ImportError:
            print("PyWavelets not installed. Skipping wavelet features.")
            return pd.DataFrame(index=series.index)
    
    def _calculate_skewness(self, data):
        """Calculate skewness."""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis."""
        from scipy.stats import kurtosis
        return kurtosis(data)
