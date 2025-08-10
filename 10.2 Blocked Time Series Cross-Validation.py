class BlockedTimeSeriesCV(TemporalValidationBase):
    """
    Blocked cross-validation with margins to prevent data leakage.
    """
    
    def __init__(self, n_splits=5, test_size=0.2, train_size=0.6, gap=0.1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        """Generate blocked splits with margins."""
        
        n_samples = len(X)
        splits = []
        
        for i in range(self.n_splits):
            # Calculate block boundaries as proportions
            block_size = 1.0 / self.n_splits
            block_start = i * block_size
            
            # Train set
            train_start = int(block_start * n_samples)
            train_end = int((block_start + self.train_size * block_size) * n_samples)
            
            # Gap
            gap_end = int((block_start + (self.train_size + self.gap) * block_size) * n_samples)
            
            # Test set
            test_start = gap_end
            test_end = int((block_start + (self.train_size + self.gap + self.test_size) * block_size) * n_samples)
            
            # Ensure boundaries are within valid range
            train_start = max(0, min(train_start, n_samples))
            train_end = max(train_start, min(train_end, n_samples))
            test_start = max(train_end, min(test_start, n_samples))
            test_end = max(test_start, min(test_end, n_samples))
            
            if train_end > train_start and test_end > test_start:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))
        
        return splits
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
