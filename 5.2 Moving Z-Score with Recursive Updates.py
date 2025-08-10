class MovingZScoreDetector:
    """Online outlier detection using recursive moving statistics."""
    
    def __init__(self, window_size=100, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squares of differences from mean
        
    def update_and_detect(self, new_value):
        """Update statistics and detect outliers in real-time."""
        # Update count
        self.count += 1
        
        # Update mean and variance using Welford's online algorithm
        delta = new_value - self.mean
        self.mean += delta / min(self.count, self.window_size)
        delta2 = new_value - self.mean
        self.m2 += delta * delta2
        
        # Calculate current variance
        if self.count < 2:
            return False, 0.0
            
        n = min(self.count, self.window_size)
        variance = self.m2 / (n - 1)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return False, 0.0
            
        # Calculate Z-score
        z_score = abs(new_value - self.mean) / std_dev
        
        # Detect outlier
        is_outlier = z_score > self.threshold
        
        return is_outlier, z_score
