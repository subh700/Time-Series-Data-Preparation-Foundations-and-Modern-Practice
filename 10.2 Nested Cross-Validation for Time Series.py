class NestedTimeSeriesCV:
    """
    Nested cross-validation for hyperparameter tuning and model selection
    while preserving temporal ordering.
    """
    
    def __init__(self, outer_cv=None, inner_cv=None, scoring='neg_mean_squared_error'):
        self.outer_cv = outer_cv or WalkForwardValidator(n_splits=5, expanding_window=True)
        self.inner_cv = inner_cv or WalkForwardValidator(n_splits=3, expanding_window=True)
        self.scoring = scoring
        
    def validate(self, estimator, X, y, param_grid=None):
        """
        Perform nested cross-validation for unbiased performance estimation.
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import get_scorer
        
        scorer = get_scorer(self.scoring)
        
        outer_scores = []
        best_params_per_fold = []
        feature_importance_per_fold = []
        
        # Outer cross-validation loop
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(self.outer_cv.split(X)):
            X_outer_train = X.iloc[outer_train_idx] if hasattr(X, 'iloc') else X[outer_train_idx]
            y_outer_train = y.iloc[outer_train_idx] if hasattr(y, 'iloc') else y[outer_train_idx]
            X_outer_test = X.iloc[outer_test_idx] if hasattr(X, 'iloc') else X[outer_test_idx]
            y_outer_test = y.iloc[outer_test_idx] if hasattr(y, 'iloc') else y[outer_test_idx]
            
            if param_grid:
                # Inner cross-validation for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=self.inner_cv,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                
                grid_search.fit(X_outer_train, y_outer_train)
                best_estimator = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
            else:
                # No hyperparameter tuning
                best_estimator = estimator.fit(X_outer_train, y_outer_train)
                best_params = {}
            
            # Evaluate on outer test set
            y_pred = best_estimator.predict(X_outer_test)
            fold_score = scorer._score_func(y_outer_test, y_pred)
            
            outer_scores.append(fold_score)
            best_params_per_fold.append(best_params)
            
            # Extract feature importance if available
            if hasattr(best_estimator, 'feature_importances_'):
                feature_importance_per_fold.append(best_estimator.feature_importances_)
            elif hasattr(best_estimator, 'coef_'):
                feature_importance_per_fold.append(np.abs(best_estimator.coef_))
        
        return {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_per_fold': best_params_per_fold,
            'feature_importances': feature_importance_per_fold
        }
