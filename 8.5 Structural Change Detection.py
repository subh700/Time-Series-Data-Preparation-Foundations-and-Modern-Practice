class ChangePointFeatureEngineer:
    """
    Extract features based on structural changes and regime shifts in time series.
    """
    
    def __init__(self, min_segment_length=10, max_change_points=10):
        self.min_segment_length = min_segment_length
        self.max_change_points = max_change_points
        
    def create_change_point_features(self, data, target_col, methods=None):
        """
        Create features based on detected change points using multiple methods.
        """
        if methods is None:
            methods = ['cusum', 'pelt', 'binary_segmentation', 'variance_change']
        
        features = pd.DataFrame(index=data.index)
        change_point_results = {}
        
        series = data[target_col].dropna()
        
        # CUSUM-based change point detection
        if 'cusum' in methods:
            cusum_features, cusum_cps = self._cusum_change_detection(series)
            features = pd.concat([features, cusum_features], axis=1)
            change_point_results['cusum'] = cusum_cps
        
        # PELT (Pruned Exact Linear Time)
        if 'pelt' in methods:
            pelt_features, pelt_cps = self._pelt_change_detection(series)
            features = pd.concat([features, pelt_features], axis=1)
            change_point_results['pelt'] = pelt_cps
        
        # Binary Segmentation
        if 'binary_segmentation' in methods:
            binseg_features, binseg_cps = self._binary_segmentation(series)
            features = pd.concat([features, binseg_features], axis=1)
            change_point_results['binary_segmentation'] = binseg_cps
        
        # Variance change detection
        if 'variance_change' in methods:
            var_features, var_cps = self._variance_change_detection(series)
            features = pd.concat([features, var_features], axis=1)
            change_point_results['variance'] = var_cps
        
        # Aggregate change point features
        features = self._create_aggregate_change_features(
            features, change_point_results, series
        )
        
        return features, change_point_results
    
    def _cusum_change_detection(self, series):
        """CUSUM-based change point detection."""
        
        # Calculate CUSUM statistics
        mean_val = series.mean()
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        
        h = 2 * series.std()  # Decision threshold
        
        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (series.iloc[i] - mean_val) - h/2)
            cusum_neg[i] = min(0, cusum_neg[i-1] + (series.iloc[i] - mean_val) + h/2)
        
        # Detect change points
        change_points = []
        for i in range(1, len(series)):
            if cusum_pos[i] > h or cusum_neg[i] < -h:
                change_points.append(i)
                # Reset CUSUM after detection
                cusum_pos[i:] = 0
                cusum_neg[i:] = 0
        
        # Create features
        features = pd.DataFrame(index=series.index)
        features['cusum_positive'] = cusum_pos
        features['cusum_negative'] = cusum_neg
        features['cusum_signal'] = cusum_pos + cusum_neg
        
        # Change point indicators
        change_indicator = np.zeros(len(series))
        for cp in change_points:
            if cp < len(change_indicator):
                change_indicator[cp] = 1
        features['cusum_change_point'] = change_indicator
        
        # Distance to nearest change point
        features['cusum_distance_to_change'] = self._calculate_distance_to_changes(
            change_points, len(series)
        )
        
        return features, change_points
    
    def _pelt_change_detection(self, series):
        """PELT (Pruned Exact Linear Time) change detection."""
        try:
            import ruptures as rpt
            
            # Apply PELT algorithm
            model = "rbf"  # Radial basis function kernel
            algo = rpt.Pelt(model=model, min_size=self.min_segment_length)
            algo.fit(series.values.reshape(-1, 1))
            change_points = algo.predict(pen=10)
            
            # Remove the last point (end of series)
            if change_points and change_points[-1] == len(series):
                change_points = change_points[:-1]
            
            features = pd.DataFrame(index=series.index)
            
            # Change point indicators
            change_indicator = np.zeros(len(series))
            for cp in change_points:
                if cp < len(change_indicator):
                    change_indicator[cp] = 1
            features['pelt_change_point'] = change_indicator
            
            # Segment means
            segment_means = self._calculate_segment_means(series.values, change_points)
            features['pelt_segment_mean'] = segment_means
            
            # Distance to change points
            features['pelt_distance_to_change'] = self._calculate_distance_to_changes(
                change_points, len(series)
            )
            
            return features, change_points
            
        except ImportError:
            print("ruptures package not available. Skipping PELT detection.")
            return pd.DataFrame(index=series.index), []
    
    def _binary_segmentation(self, series):
        """Binary segmentation change point detection."""
        
        def find_best_split(data, start, end):
            """Find the best split point in a segment."""
            if end - start < 2 * self.min_segment_length:
                return None
            
            best_split = None
            best_score = -np.inf
            
            for split in range(start + self.min_segment_length, 
                             end - self.min_segment_length + 1):
                
                left_data = data[start:split]
                right_data = data[split:end]
                
                # Calculate log-likelihood score
                left_ll = self._calculate_log_likelihood(left_data)
                right_ll = self._calculate_log_likelihood(right_data)
                combined_ll = self._calculate_log_likelihood(data[start:end])
                
                score = left_ll + right_ll - combined_ll
                
                if score > best_score:
                    best_score = score
                    best_split = split
            
            return best_split if best_score > 0 else None
        
        # Recursive binary segmentation
        segments_to_process = [(0, len(series))]
        change_points = []
        
        while segments_to_process and len(change_points) < self.max_change_points:
            start, end = segments_to_process.pop(0)
            split = find_best_split(series.values, start, end)
            
            if split is not None:
                change_points.append(split)
                segments_to_process.append((start, split))
                segments_to_process.append((split, end))
        
        change_points.sort()
        
        # Create features
        features = pd.DataFrame(index=series.index)
        
        change_indicator = np.zeros(len(series))
        for cp in change_points:
            if cp < len(change_indicator):
                change_indicator[cp] = 1
        features['binseg_change_point'] = change_indicator
        
        # Segment statistics
        segment_means = self._calculate_segment_means(series.values, change_points)
        features['binseg_segment_mean'] = segment_means
        
        return features, change_points
    
    def _variance_change_detection(self, series):
        """Detect changes in variance using rolling statistics."""
        
        window_size = max(self.min_segment_length, len(series) // 20)
        rolling_var = series.rolling(window=window_size).var()
        
        # Detect significant changes in variance
        var_changes = []
        threshold = 2 * rolling_var.std()
        
        for i in range(window_size, len(rolling_var) - window_size):
            current_var = rolling_var.iloc[i]
            
            # Compare with previous and next windows
            prev_var = rolling_var.iloc[i - window_size//2]
            next_var = rolling_var.iloc[i + window_size//2]
            
            if (abs(current_var - prev_var) > threshold or 
                abs(current_var - next_var) > threshold):
                var_changes.append(i)
        
        features = pd.DataFrame(index=series.index)
        features['variance_rolling'] = rolling_var
        
        # Change point indicators
        change_indicator = np.zeros(len(series))
        for cp in var_changes:
            if cp < len(change_indicator):
                change_indicator[cp] = 1
        features['variance_change_point'] = change_indicator
        
        # Variance ratio (current vs. historical)
        features['variance_ratio'] = (
            rolling_var / rolling_var.expanding().mean()
        )
        
        return features, var_changes
    
    def _calculate_log_likelihood(self, data):
        """Calculate log-likelihood assuming normal distribution."""
        if len(data) < 2:
            return -np.inf
        
        mean = np.mean(data)
        var = np.var(data, ddof=1)
        
        if var <= 0:
            return -np.inf
        
        ll = -0.5 * len(data) * (np.log(2 * np.pi * var) + 1)
        return ll
    
    def _calculate_segment_means(self, data, change_points):
        """Calculate mean for each segment defined by change points."""
        segment_means = np.zeros(len(data))
        
        # Add start and end points
        segments = [0] + sorted(change_points) + [len(data)]
        
        for i in range(len(segments) - 1):
            start = segments[i]
            end = segments[i + 1]
            segment_mean = np.mean(data[start:end])
            segment_means[start:end] = segment_mean
        
        return segment_means
    
    def _calculate_distance_to_changes(self, change_points, series_length):
        """Calculate distance to nearest change point for each observation."""
        distances = np.zeros(series_length)
        
        if not change_points:
            return distances
        
        for i in range(series_length):
            min_distance = min(abs(i - cp) for cp in change_points)
            distances[i] = min_distance
        
        return distances
    
    def _create_aggregate_change_features(self, features, change_results, series):
        """Create aggregate features from multiple change point methods."""
        
        # Count total detected change points
        total_changes = 0
        for method_cps in change_results.values():
            total_changes += len(method_cps)
        
        features['total_change_points'] = total_changes
        
        # Consensus change points (detected by multiple methods)
        all_change_points = []
        for method_cps in change_results.values():
            all_change_points.extend(method_cps)
        
        if all_change_points:
            # Find change points that are close to each other across methods
            consensus_threshold = max(5, len(series) // 100)
            consensus_changes = []
            
            for cp in set(all_change_points):
                nearby_count = sum(1 for other_cp in all_change_points 
                                 if abs(cp - other_cp) <= consensus_threshold)
                if nearby_count >= 2:  # At least 2 methods agree
                    consensus_changes.append(cp)
            
            # Consensus change indicators
            consensus_indicator = np.zeros(len(series))
            for cp in consensus_changes:
                if cp < len(consensus_indicator):
                    consensus_indicator[cp] = 1
            features['consensus_change_point'] = consensus_indicator
            
            # Stability measure (inverse of change frequency)
            features['stability_measure'] = 1 / (1 + len(consensus_changes) / len(series))
        
        return features
