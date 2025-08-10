class TimeSeriesCVDiagnostics:
    """
    Comprehensive diagnostics for time series cross-validation.
    """
    
    def __init__(self):
        self.results = {}
        
    def diagnose_cv_strategy(self, cv_strategy, X, y, model_class, **model_params):
        """
        Comprehensive diagnosis of cross-validation strategy effectiveness.
        """
        
        diagnostics = {
            'temporal_consistency': self._check_temporal_consistency(cv_strategy, X),
            'data_leakage': self._detect_data_leakage(cv_strategy, X, y),
            'performance_stability': self._assess_performance_stability(cv_strategy, X, y, model_class, **model_params),
            'fold_characteristics': self._analyze_fold_characteristics(cv_strategy, X, y),
            'coverage_analysis': self._analyze_temporal_coverage(cv_strategy, X)
        }
        
        return diagnostics
    
    def _check_temporal_consistency(self, cv_strategy, X):
        """Check if CV strategy maintains temporal ordering."""
        
        consistency_issues = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X)):
            # Check basic temporal ordering
            if len(train_idx) > 0 and len(test_idx) > 0:
                max_train = max(train_idx)
                min_test = min(test_idx)
                
                if max_train >= min_test:
                    consistency_issues.append({
                        'fold': fold_idx,
                        'issue': 'temporal_ordering_violation',
                        'max_train_idx': max_train,
                        'min_test_idx': min_test
                    })
            
            # Check for gaps in training data
            if len(train_idx) > 1:
                train_gaps = np.diff(np.sort(train_idx))
                large_gaps = train_gaps[train_gaps > 1]
                
                if len(large_gaps) > 0:
                    consistency_issues.append({
                        'fold': fold_idx,
                        'issue': 'training_data_gaps',
                        'gap_sizes': large_gaps.tolist()
                    })
        
        return {
            'is_consistent': len(consistency_issues) == 0,
            'issues': consistency_issues
        }
    
    def _detect_data_leakage(self, cv_strategy, X, y):
        """Detect potential data leakage in CV strategy."""
        
        leakage_tests = []
        
        # Test 1: Future information in training
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X)):
            future_in_train = np.intersect1d(train_idx, test_idx)
            
            if len(future_in_train) > 0:
                leakage_tests.append({
                    'fold': fold_idx,
                    'test': 'overlapping_indices',
                    'severity': 'high',
                    'description': f'Found {len(future_in_train)} overlapping indices'
                })
        
        # Test 2: Statistical independence test
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X
        
        independence_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X)):
            if len(train_idx) > 10 and len(test_idx) > 10:
                # Sample correlation test
                train_sample = X_values[train_idx[-10:]]  # Last 10 training samples
                test_sample = X_values[test_idx[:10]]     # First 10 test samples
                
                if train_sample.ndim == 1:
                    correlation = np.corrcoef(train_sample, test_sample)[0, 1]
                else:
                    # For multivariate data, use mean correlation
                    correlations = []
                    for col in range(train_sample.shape[1]):
                        corr = np.corrcoef(train_sample[:, col], test_sample[:, col])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    correlation = np.mean(correlations) if correlations else 0
                
                independence_scores.append(abs(correlation))
        
        avg_correlation = np.mean(independence_scores) if independence_scores else 0
        
        if avg_correlation > 0.3:  # Threshold for concerning correlation
            leakage_tests.append({
                'test': 'temporal_correlation',
                'severity': 'medium',
                'avg_correlation': avg_correlation,
                'description': f'High correlation ({avg_correlation:.3f}) between train/test boundaries'
            })
        
        return {
            'has_leakage': len(leakage_tests) > 0,
            'tests': leakage_tests
        }
    
    def _assess_performance_stability(self, cv_strategy, X, y, model_class, **model_params):
        """Assess stability of performance across folds."""
        
        fold_scores = []
        fold_predictions = []
        
        for train_idx, test_idx in cv_strategy.split(X):
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model.predict(X_test)
            
            # Calculate multiple metrics
            mae = np.mean(np.abs(y_test - y_pred))
            mse = np.mean((y_test - y_pred) ** 2)
            
            fold_scores.append({'mae': mae, 'mse': mse})
            fold_predictions.append((y_test, y_pred))
        
        # Analyze score stability
        mae_scores = [score['mae'] for score in fold_scores]
        mse_scores = [score['mse'] for score in fold_scores]
        
        mae_cv = np.std(mae_scores) / np.mean(mae_scores) if np.mean(mae_scores) > 0 else np.inf
        mse_cv = np.std(mse_scores) / np.mean(mse_scores) if np.mean(mse_scores) > 0 else np.inf
        
        return {
            'fold_scores': fold_scores,
            'mae_stability': {
                'mean': np.mean(mae_scores),
                'std': np.std(mae_scores),
                'cv': mae_cv
            },
            'mse_stability': {
                'mean': np.mean(mse_scores),
                'std': np.std(mse_scores),
                'cv': mse_cv
            },
            'is_stable': mae_cv < 0.5 and mse_cv < 0.5  # Reasonable stability threshold
        }
    
    def _analyze_fold_characteristics(self, cv_strategy, X, y):
        """Analyze characteristics of each fold."""
        
        fold_stats = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X)):
            train_size = len(train_idx)
            test_size = len(test_idx)
            
            # Training set statistics
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            train_mean = np.mean(y_train)
            train_std = np.std(y_train)
            
            # Test set statistics
            y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            test_mean = np.mean(y_test)
            test_std = np.std(y_test)
            
            # Distribution similarity
            from scipy.stats import ks_2samp
            ks_stat, ks_p_value = ks_2samp(y_train, y_test)
            
            fold_stats.append({
                'fold': fold_idx,
                'train_size': train_size,
                'test_size': test_size,
                'train_mean': train_mean,
                'train_std': train_std,
                'test_mean': test_mean,
                'test_std': test_std,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'distribution_similar': ks_p_value > 0.05
            })
        
        return fold_stats
    
    def _analyze_temporal_coverage(self, cv_strategy, X):
        """Analyze temporal coverage of CV strategy."""
        
        n_samples = len(X)
        coverage_matrix = np.zeros((n_samples, 2))  # 0: train, 1: test
        
        for train_idx, test_idx in cv_strategy.split(X):
            coverage_matrix[train_idx, 0] += 1
            coverage_matrix[test_idx, 1] += 1
        
        # Calculate coverage statistics
        train_coverage = coverage_matrix[:, 0]
        test_coverage = coverage_matrix[:, 1]
        
        return {
            'train_coverage_mean': np.mean(train_coverage),
            'train_coverage_std': np.std(train_coverage),
            'test_coverage_mean': np.mean(test_coverage),
            'test_coverage_std': np.std(test_coverage),
            'uncovered_samples': np.sum((train_coverage + test_coverage) == 0),
            'coverage_matrix': coverage_matrix
        }
