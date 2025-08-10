class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation for financial time series 
    to address overlapping samples and serial correlation.
    """
    
    def __init__(self, n_splits=6, n_test_splits=2, purge_length=0, embargo_length=0):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits  # Number of splits used for testing
        self.purge_length = purge_length    # Observations to purge around test set
        self.embargo_length = embargo_length # Forward-looking embargo
        
    def split(self, X, y=None, pred_times=None, eval_times=None):
        """
        Generate combinatorial purged splits.
        
        pred_times: Time when prediction is made
        eval_times: Time when outcome is evaluated
        """
        
        n_samples = len(X)
        
        if pred_times is None:
            pred_times = X.index if hasattr(X, 'index') else np.arange(n_samples)
        if eval_times is None:
            eval_times = pred_times
        
        # Create base splits
        split_size = n_samples // self.n_splits
        base_splits = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_samples)
            base_splits.append(np.arange(start_idx, end_idx))
        
        # Generate all combinations of test splits
        from itertools import combinations
        
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_split_indices in test_combinations:
            # Combine test splits
            test_indices = np.concatenate([base_splits[i] for i in test_split_indices])
            
            # Remaining splits form potential training set
            train_split_indices = [i for i in range(self.n_splits) if i not in test_split_indices]
            potential_train_indices = np.concatenate([base_splits[i] for i in train_split_indices])
            
            # Apply purging and embargo
            train_indices = self._apply_purging_embargo(
                potential_train_indices, test_indices, pred_times, eval_times
            )
            
            if len(train_indices) > 0:
                yield train_indices, test_indices
    
    def _apply_purging_embargo(self, train_indices, test_indices, pred_times, eval_times):
        """Apply purging and embargo to prevent data leakage."""
        
        # Convert indices to times if needed
        if hasattr(pred_times, 'iloc'):
            test_pred_times = pred_times.iloc[test_indices]
            test_eval_times = eval_times.iloc[test_indices]
            train_pred_times = pred_times.iloc[train_indices]
            train_eval_times = eval_times.iloc[train_indices]
        else:
            test_pred_times = pred_times[test_indices]
            test_eval_times = eval_times[test_indices]
            train_pred_times = pred_times[train_indices]
            train_eval_times = eval_times[train_indices]
        
        # Find overlapping observations
        purged_indices = []
        
        for train_idx, train_idx_orig in enumerate(train_indices):
            train_pred_time = train_pred_times.iloc[train_idx] if hasattr(train_pred_times, 'iloc') else train_pred_times[train_idx]
            train_eval_time = train_eval_times.iloc[train_idx] if hasattr(train_eval_times, 'iloc') else train_eval_times[train_idx]
            
            # Check for overlaps with test set
            overlaps = False
            
            for test_pred_time, test_eval_time in zip(test_pred_times, test_eval_times):
                # Purging condition: training eval time overlaps with test period
                if (train_eval_time >= test_pred_time - pd.Timedelta(seconds=self.purge_length) and
                    train_eval_time <= test_eval_time + pd.Timedelta(seconds=self.purge_length)):
                    overlaps = True
                    break
                
                # Embargo condition: training prediction is too close to test prediction
                if (train_pred_time >= test_pred_time - pd.Timedelta(seconds=self.embargo_length) and
                    train_pred_time <= test_pred_time):
                    overlaps = True
                    break
            
            if not overlaps:
                purged_indices.append(train_idx_orig)
        
        return np.array(purged_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Calculate number of possible combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
