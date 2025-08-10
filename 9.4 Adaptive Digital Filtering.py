class AdaptiveSignalFilter:
    """
    Advanced signal filtering with adaptive algorithms and noise reduction.
    """
    
    def __init__(self, filter_type='adaptive', adaptation_method='lms'):
        self.filter_type = filter_type
        self.adaptation_method = adaptation_method  # 'lms', 'rls', 'kalman'
        
    def filter_signal(self, signal, reference_signal=None, **kwargs):
        """
        Apply adaptive filtering based on selected method.
        """
        
        if self.filter_type == 'adaptive':
            return self._adaptive_filter(signal, reference_signal, **kwargs)
        elif self.filter_type == 'wiener':
            return self._wiener_filter(signal, **kwargs)
        elif self.filter_type == 'kalman':
            return self._kalman_filter(signal, **kwargs)
        elif self.filter_type == 'median_hybrid':
            return self._median_hybrid_filter(signal, **kwargs)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
    
    def _adaptive_filter(self, signal, reference_signal, filter_length=32, step_size=0.01):
        """
        Adaptive filtering using LMS or RLS algorithms.
        """
        
        if reference_signal is None:
            # Create reference signal from delayed version
            reference_signal = np.roll(signal, 1)
            reference_signal[0] = signal[0]
        
        if self.adaptation_method == 'lms':
            return self._lms_adaptive_filter(signal, reference_signal, filter_length, step_size)
        elif self.adaptation_method == 'rls':
            return self._rls_adaptive_filter(signal, reference_signal, filter_length)
        else:
            raise ValueError(f"Unknown adaptation method: {self.adaptation_method}")
    
    def _lms_adaptive_filter(self, signal, reference_signal, filter_length, step_size):
        """Least Mean Squares adaptive filter."""
        
        n_samples = len(signal)
        
        # Initialize filter weights
        weights = np.zeros(filter_length)
        
        # Output signals
        filtered_signal = np.zeros(n_samples)
        error_signal = np.zeros(n_samples)
        
        # Create input buffer
        input_buffer = np.zeros(filter_length)
        
        for n in range(n_samples):
            # Update input buffer
            input_buffer[1:] = input_buffer[:-1]
            input_buffer[0] = reference_signal[n] if n < len(reference_signal) else 0
            
            # Filter output
            y_n = np.dot(weights, input_buffer)
            filtered_signal[n] = y_n
            
            # Error calculation
            error = signal[n] - y_n
            error_signal[n] = error
            
            # Weight update (LMS algorithm)
            weights += step_size * error * input_buffer
        
        return {
            'filtered_signal': filtered_signal,
            'error_signal': error_signal,
            'final_weights': weights,
            'filter_convergence': self._assess_convergence(error_signal)
        }
    
    def _rls_adaptive_filter(self, signal, reference_signal, filter_length, 
                           forgetting_factor=0.99, regularization=1e-4):
        """Recursive Least Squares adaptive filter."""
        
        n_samples = len(signal)
        
        # Initialize
        weights = np.zeros(filter_length)
        P = np.eye(filter_length) / regularization  # Inverse correlation matrix
        
        filtered_signal = np.zeros(n_samples)
        error_signal = np.zeros(n_samples)
        input_buffer = np.zeros(filter_length)
        
        for n in range(n_samples):
            # Update input buffer
            input_buffer[1:] = input_buffer[:-1]
            input_buffer[0] = reference_signal[n] if n < len(reference_signal) else 0
            
            # Filter output
            y_n = np.dot(weights, input_buffer)
            filtered_signal[n] = y_n
            
            # Error calculation
            error = signal[n] - y_n
            error_signal[n] = error
            
            # RLS weight update
            pi_n = np.dot(P, input_buffer)
            k_n = pi_n / (forgetting_factor + np.dot(input_buffer, pi_n))
            
            weights += k_n * error
            P = (P - np.outer(k_n, pi_n)) / forgetting_factor
        
        return {
            'filtered_signal': filtered_signal,
            'error_signal': error_signal,
            'final_weights': weights,
            'filter_convergence': self._assess_convergence(error_signal)
        }
    
    def _wiener_filter(self, signal, noise_variance=None, signal_variance=None):
        """Wiener filtering for optimal noise reduction."""
        
        if noise_variance is None:
            # Estimate noise variance from high-frequency components
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 0.1, btype='high')
            high_freq = filtfilt(b, a, signal)
            noise_variance = np.var(high_freq)
        
        if signal_variance is None:
            signal_variance = np.var(signal)
        
        # Compute power spectral density
        from scipy.signal import welch
        freqs, psd = welch(signal, nperseg=min(256, len(signal)//4))
        
        # Wiener filter transfer function
        signal_psd = psd - noise_variance  # Estimate signal PSD
        signal_psd = np.maximum(signal_psd, 0.1 * noise_variance)  # Avoid negative values
        
        wiener_response = signal_psd / (signal_psd + noise_variance)
        
        # Apply filter in frequency domain
        signal_fft = np.fft.fft(signal)
        freqs_full = np.fft.fftfreq(len(signal))
        
        # Interpolate Wiener response to full frequency grid
        wiener_full = np.interp(np.abs(freqs_full[:len(freqs_full)//2]), 
                               freqs, wiener_response)
        
        # Create symmetric response for negative frequencies
        wiener_symmetric = np.concatenate([wiener_full, wiener_full[::-1]])
        if len(wiener_symmetric) > len(signal_fft):
            wiener_symmetric = wiener_symmetric[:len(signal_fft)]
        
        # Apply filter
        filtered_fft = signal_fft * wiener_symmetric
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        
        return {
            'filtered_signal': filtered_signal,
            'wiener_response': wiener_response,
            'noise_variance_estimate': noise_variance,
            'frequencies': freqs
        }
    
    def _kalman_filter(self, signal, process_noise=1e-5, measurement_noise=1e-2):
        """Kalman filtering for time series smoothing."""
        
        n_samples = len(signal)
        
        # State space model: x(k) = x(k-1) + w(k), y(k) = x(k) + v(k)
        # Initialize
        x_hat = np.zeros(n_samples)  # State estimate
        P = np.ones(n_samples)       # Error covariance
        
        x_hat[0] = signal[0]
        P[0] = 1.0
        
        # Kalman filtering
        for k in range(1, n_samples):
            # Prediction step
            x_hat_minus = x_hat[k-1]  # Predicted state
            P_minus = P[k-1] + process_noise  # Predicted error covariance
            
            # Update step
            K = P_minus / (P_minus + measurement_noise)  # Kalman gain
            x_hat[k] = x_hat_minus + K * (signal[k] - x_hat_minus)
            P[k] = (1 - K) * P_minus
        
        return {
            'filtered_signal': x_hat,
            'error_covariance': P,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise
        }
    
    def _median_hybrid_filter(self, signal, window_size=5, alpha=0.3):
        """Hybrid median-exponential smoothing filter."""
        
        # Apply median filter for spike removal
        from scipy.signal import medfilt
        median_filtered = medfilt(signal, kernel_size=window_size)
        
        # Apply exponential smoothing
        exp_smoothed = np.zeros_like(signal)
        exp_smoothed[0] = median_filtered[0]
        
        for i in range(1, len(signal)):
            exp_smoothed[i] = alpha * median_filtered[i] + (1 - alpha) * exp_smoothed[i-1]
        
        return {
            'filtered_signal': exp_smoothed,
            'median_component': median_filtered,
            'window_size': window_size,
            'smoothing_factor': alpha
        }
    
    def _assess_convergence(self, error_signal, window_size=50):
        """Assess filter convergence based on error signal."""
        
        if len(error_signal) < window_size * 2:
            return {'converged': False, 'convergence_point': None}
        
        # Calculate moving variance of error
        error_variance = pd.Series(error_signal).rolling(window=window_size).var()
        
        # Find convergence point (where variance stabilizes)
        variance_diff = error_variance.diff().abs()
        stable_threshold = 0.1 * error_variance.std()
        
        stable_points = variance_diff < stable_threshold
        
        if stable_points.sum() > window_size:
            convergence_point = stable_points.idxmax()
            converged = True
        else:
            convergence_point = None
            converged = False
        
        return {
            'converged': converged,
            'convergence_point': convergence_point,
            'final_error_variance': error_variance.iloc[-1] if len(error_variance) > 0 else np.nan,
            'convergence_rate': self._calculate_convergence_rate(error_signal)
        }
    
    def _calculate_convergence_rate(self, error_signal):
        """Calculate exponential convergence rate."""
        
        # Fit exponential decay to absolute error
        abs_error = np.abs(error_signal)
        
        if len(abs_error) < 10:
            return np.nan
        
        # Use logarithmic fitting for exponential decay
        log_error = np.log(abs_error + 1e-10)  # Avoid log(0)
        time_indices = np.arange(len(log_error))
        
        try:
            # Linear fit in log space
            slope, intercept = np.polyfit(time_indices, log_error, 1)
            convergence_rate = -slope  # Negative slope indicates decay
        except:
            convergence_rate = np.nan
        
        return convergence_rate
