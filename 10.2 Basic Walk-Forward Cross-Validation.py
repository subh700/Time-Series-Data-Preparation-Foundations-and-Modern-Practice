class WalkForwardValidator(TemporalValidationBase):
    """
    Walk-forward validation with expanding or sliding window options.
    """
    
    def __init__(self, n_splits=5, test_size=1, gap=0, max_train_size=None, 
                 expanding_window=True):
        super().__init__(test_size, gap, max_train_size)
        self.n_splits = n_splits
        self.expanding_window = expanding_window
        
    def split(self, X, y=None, groups=None):
        """Generate walk-forward splits."""
        
        n_samples = len(X)
        
        # Calculate split parameters
        if isinstance(self.test_size, float):
            test_size = int(n_samples * self.test_size)
        else:
            test_size = self.test_size
        
        # Determine starting position for first test set
        if self.expanding_window:
            min_train_size = max(1, n_samples // (self.n_splits + 1))
        else:
            # For sliding window, ensure consistent train size
            if self.max_train_size:
                min_train_size = self.max_train_size
            else:
                min_train_size = max(1, (n_samples - self.n_splits * test_size) // 2)
        
        splits = []
        
        for split_idx in range(self.n_splits):
            # Calculate test set boundaries
            test_start = min_train_size + split_idx * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_end <= test_start:
                break  # No more valid splits
            
            # Calculate train set boundaries
            if self.expanding_window:
                train_start = 0
                train_end = test_start
            else:  # sliding window
                if self.max_train_size:
                    train_start = max(0, test_start - self.max_train_size)
                else:
                    train_start = max(0, test_start - min_train_size)
                train_end = test_start
            
            # Apply gap
            train_end, test_start = self.create_gap(train_end, test_start, n_samples)
            
            # Validate split
            if train_end > train_start and test_end > test_start:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                if self.validate_temporal_ordering(train_indices, test_indices):
                    splits.append((train_indices, test_indices))
        
        return splits
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits
