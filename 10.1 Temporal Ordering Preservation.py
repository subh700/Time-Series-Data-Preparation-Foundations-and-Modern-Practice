class TemporalValidationBase:
    """
    Base class for temporal validation strategies ensuring proper time ordering.
    """
    
    def __init__(self, test_size=None, gap=0, max_train_size=None):
        self.test_size = test_size
        self.gap = gap  # Gap between train and test to prevent leakage
        self.max_train_size = max_train_size
        
    def validate_temporal_ordering(self, train_indices, test_indices):
        """Ensure temporal ordering is preserved."""
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            return True
            
        max_train_idx = max(train_indices)
        min_test_idx = min(test_indices)
        
        # Account for gap
        return max_train_idx + self.gap < min_test_idx
    
    def create_gap(self, train_end, test_start, total_length):
        """Create appropriate gap between train and test sets."""
        
        if self.gap == 0:
            return train_end, test_start
        
        # Adjust indices to create gap
        adjusted_train_end = min(train_end, test_start - self.gap)
        adjusted_test_start = max(test_start, train_end + self.gap)
        
        # Ensure we don't exceed bounds
        adjusted_train_end = max(0, min(adjusted_train_end, total_length))
        adjusted_test_start = max(0, min(adjusted_test_start, total_length))
        
        return adjusted_train_end, adjusted_test_start
