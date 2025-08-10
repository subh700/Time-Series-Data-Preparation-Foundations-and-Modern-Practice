class AdvancedWaveletDecomposition:
    """
    Advanced wavelet-based decomposition with multiple wavelet families
    and adaptive level selection.
    """
    
    def __init__(self, wavelet='db4', mode='symmetric', max_levels=None):
        self.wavelet = wavelet
        self.mode = mode
        self.max_levels = max_levels
        
    def decompose(self, data, method='dwt'):
        """
        Perform wavelet decomposition using specified method.
        
        Methods:
        - 'dwt': Discrete Wavelet Transform
        - 'cwt': Continuous Wavelet Transform  
        - 'swt': Stationary Wavelet Transform
        - 'wpt': Wavelet Packet Transform
        """
        
        series = data if isinstance(data, np.ndarray) else data.values
        
        if method == 'dwt':
            return self._discrete_wavelet_transform(series)
        elif method == 'cwt':
            return self._continuous_wavelet_transform(series)
        elif method == 'swt':
            return self._stationary_wavelet_transform(series)
        elif method == 'wpt':
            return self._wavelet_packet_transform(series)
        else:
            raise ValueError(f"Unknown wavelet method: {method}")
    
    def _discrete_wavelet_transform(self, series):
        """Discrete Wavelet Transform with adaptive level selection."""
        try:
            import pywt
            
            # Determine optimal decomposition levels
            if self.max_levels is None:
                max_possible_levels = pywt.dwt_max_levels(len(series), self.wavelet)
                optimal_levels = min(6, max_possible_levels)  # Practical limit
            else:
                optimal_levels = self.max_levels
            
            # Perform multi-level decomposition
            coeffs = pywt.wavedec(series, self.wavelet, level=optimal_levels, mode=self.mode)
            
            # Reconstruct components
            components = {}
            
            # Approximation coefficients (trend)
            approx_coeffs = coeffs[0]
            trend = pywt.upcoef('a', approx_coeffs, self.wavelet, level=optimal_levels, 
                              take=len(series), mode=self.mode)
            components['trend'] = trend
            
            # Detail coefficients (different frequency bands)
            details = []
            for i, detail_coeffs in enumerate(coeffs[1:], 1):
                detail = pywt.upcoef('d', detail_coeffs, self.wavelet, level=optimal_levels-i+1,
                                   take=len(series), mode=self.mode)
                details.append(detail)
                components[f'detail_{i}'] = detail
            
            # Frequency analysis of components
            frequency_bands = self._analyze_frequency_bands(details, len(series))
            components['frequency_analysis'] = frequency_bands
            
            # Quality metrics
            components['decomposition_quality'] = self._assess_wavelet_quality(
                series, trend, details
            )
            
            return components
            
        except ImportError:
            raise ImportError("PyWavelets (pywt) is required for wavelet decomposition")
    
    def _continuous_wavelet_transform(self, series, scales=None):
        """Continuous Wavelet Transform for time-frequency analysis."""
        try:
            import pywt
            
            if scales is None:
                # Generate scales for good frequency coverage
                scales = np.logspace(0, 3, 50)  # Logarithmic scale distribution
            
            # Perform CWT
            coefficients, frequencies = pywt.cwt(series, scales, self.wavelet)
            
            # Time-frequency representation
            time_freq_analysis = {
                'coefficients': coefficients,
                'scales': scales,
                'frequencies': frequencies,
                'time_points': np.arange(len(series)),
                'scalogram': np.abs(coefficients) ** 2  # Power scalogram
            }
            
            # Extract dominant patterns at different scales
            dominant_patterns = self._extract_dominant_patterns(coefficients, scales)
            time_freq_analysis['dominant_patterns'] = dominant_patterns
            
            return time_freq_analysis
            
        except ImportError:
            raise ImportError("PyWavelets (pywt) is required for wavelet decomposition")
    
    def _stationary_wavelet_transform(self, series):
        """Stationary Wavelet Transform (undecimated)."""
        try:
            import pywt
            
            # Determine levels
            if self.max_levels is None:
                max_possible_levels = pywt.swt_max_levels(len(series))
                levels = min(6, max_possible_levels)
            else:
                levels = self.max_levels
            
            # Perform SWT
            coeffs = pywt.swt(series, self.wavelet, level=levels, 
                             trim_approx=True, norm=True)
            
            # Reconstruct components
            components = {}
            
            # Each level gives approximation and detail coefficients
            for i, (approx, detail) in enumerate(coeffs):
                components[f'approx_level_{i+1}'] = approx
                components[f'detail_level_{i+1}'] = detail
            
            # Total approximation (trend)
            components['trend'] = coeffs[-1][0]  # Final approximation
            
            # Quality assessment
            components['decomposition_quality'] = self._assess_swt_quality(series, coeffs)
            
            return components
            
        except ImportError:
            raise ImportError("PyWavelets (pywt) is required for wavelet decomposition")
    
    def _wavelet_packet_transform(self, series):
        """Wavelet Packet Transform for complete frequency analysis."""
        try:
            import pywt
            
            # Create wavelet packet tree
            wp = pywt.WaveletPacket(data=series, wavelet=self.wavelet, mode=self.mode)
            
            # Determine optimal decomposition depth
            if self.max_levels is None:
                max_depth = min(6, int(np.log2(len(series))))
            else:
                max_depth = self.max_levels
            
            # Extract all nodes at maximum depth
            leaf_nodes = []
            node_names = []
            
            for node in wp.get_level(max_depth, 'natural'):
                if node.data.size > 0:
                    leaf_nodes.append(node.data)
                    node_names.append(node.path)
            
            # Reconstruct full-length signals for each packet
            components = {}
            for i, (node_data, node_name) in enumerate(zip(leaf_nodes, node_names)):
                # Reconstruct the packet to original length
                temp_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode=self.mode)
                temp_wp[node_name] = node_data
                reconstructed = temp_wp.reconstruct(update=False)
                
                # Trim or pad to original length
                if len(reconstructed) > len(series):
                    reconstructed = reconstructed[:len(series)]
                elif len(reconstructed) < len(series):
                    reconstructed = np.pad(reconstructed, (0, len(series) - len(reconstructed)))
                
                components[f'packet_{node_name}'] = reconstructed
            
            # Frequency analysis of packets
            components['frequency_analysis'] = self._analyze_packet_frequencies(
                leaf_nodes, node_names, len(series)
            )
            
            return components
            
        except ImportError:
            raise ImportError("PyWavelets (pywt) is required for wavelet decomposition")
    
    def _analyze_frequency_bands(self, details, signal_length):
        """Analyze frequency content of wavelet detail coefficients."""
        
        frequency_bands = {}
        
        for i, detail in enumerate(details):
            # Calculate power spectral density
            from scipy.signal import welch
            
            freqs, psd = welch(detail, nperseg=min(256, len(detail)//4))
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(psd)
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate frequency band characteristics
            total_power = np.trapz(psd, freqs)
            
            frequency_bands[f'detail_{i+1}'] = {
                'dominant_frequency': dominant_freq,
                'total_power': total_power,
                'power_spectrum': (freqs, psd),
                'relative_power': total_power / np.sum([np.trapz(*welch(d)) for d in details])
            }
        
        return frequency_bands
    
    def _extract_dominant_patterns(self, coefficients, scales):
        """Extract dominant patterns from CWT coefficients."""
        
        patterns = {}
        
        # For each scale, find time points with highest energy
        for i, scale in enumerate(scales):
            coeff_row = coefficients[i, :]
            power = np.abs(coeff_row) ** 2
            
            # Find peaks in power
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(power, height=np.percentile(power, 75))
            
            patterns[f'scale_{scale:.2f}'] = {
                'peak_times': peaks,
                'peak_powers': power[peaks],
                'mean_power': np.mean(power),
                'power_std': np.std(power)
            }
        
        return patterns
    
    def _assess_wavelet_quality(self, original, trend, details):
        """Assess quality of wavelet decomposition."""
        
        # Reconstruction error
        reconstructed = trend + np.sum(details, axis=0)
        mse = np.mean((original - reconstructed) ** 2)
        
        # Energy conservation
        original_energy = np.sum(original ** 2)
        reconstructed_energy = np.sum(reconstructed ** 2)
        energy_ratio = reconstructed_energy / original_energy
        
        # Component analysis
        trend_energy = np.sum(trend ** 2) / original_energy
        detail_energies = [np.sum(detail ** 2) / original_energy for detail in details]
        
        return {
            'reconstruction_mse': mse,
            'energy_conservation_ratio': energy_ratio,
            'trend_energy_ratio': trend_energy,
            'detail_energy_ratios': detail_energies,
            'total_components': len(details) + 1
        }
    
    def _assess_swt_quality(self, original, coeffs):
        """Assess quality of stationary wavelet transform."""
        
        # Reconstruct from final approximation and all details
        try:
            import pywt
            reconstructed = pywt.iswt(coeffs, self.wavelet)
            
            # Quality metrics
            mse = np.mean((original - reconstructed) ** 2)
            correlation = np.corrcoef(original, reconstructed)[0, 1]
            
            return {
                'reconstruction_mse': mse,
                'reconstruction_correlation': correlation,
                'decomposition_levels': len(coeffs)
            }
            
        except:
            return {'reconstruction_mse': np.nan, 'reconstruction_correlation': np.nan}
    
    def _analyze_packet_frequencies(self, leaf_nodes, node_names, signal_length):
        """Analyze frequency content of wavelet packets."""
        
        packet_analysis = {}
        
        for node_data, node_name in zip(leaf_nodes, node_names):
            if len(node_data) > 4:  # Minimum length for analysis
                # Calculate power spectrum
                from scipy.signal import welch
                
                freqs, psd = welch(node_data, nperseg=min(64, len(node_data)//2))
                
                packet_analysis[node_name] = {
                    'dominant_frequency': freqs[np.argmax(psd)],
                    'total_power': np.trapz(psd, freqs),
                    'bandwidth': freqs[-1] - freqs[0],
                    'packet_length': len(node_data)
                }
        
        return packet_analysis
