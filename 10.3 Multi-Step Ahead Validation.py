class MultiStepAheadValidator:
    """
    Validation for multi-step ahead forecasting with different horizons.
    """
    
    def __init__(self, forecast_horizons=[1, 3, 7, 14], cv_strategy='expanding'):
        self.forecast_horizons = forecast_horizons
        self.cv_strategy = cv_strategy
        
    def validate_forecaster(self, forecaster, X, y, n_splits=5):
        """
        Validate multi-step ahead forecasting performance.
        """
        
        results = {}
        
        for horizon in self.forecast_horizons:
            horizon_results = self._validate_single_horizon(
                forecaster, X, y, horizon, n_splits
            )
            results[f'horizon_{horizon}'] = horizon_results
        
        # Aggregate results across horizons
        results['summary'] = self._summarize_multi_horizon_results(results)
        
        return results
    
    def _validate_single_horizon(self, forecaster, X, y, horizon, n_splits):
        """Validate forecasting for a specific horizon."""
        
        n_samples = len(X)
        fold_scores = []
        fold_predictions = []
        
        # Create validation splits
        if self.cv_strategy == 'expanding':
            base_cv = WalkForwardValidator(n_splits=n_splits, expanding_window=True)
        else:
            base_cv = WalkForwardValidator(n_splits=n_splits, expanding_window=False)
        
        for train_idx, test_idx in base_cv.split(X):
            # Adjust test indices for multi-step ahead prediction
            adjusted_test_idx = test_idx[test_idx + horizon < n_samples]
            
            if len(adjusted_test_idx) == 0:
                continue
            
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            
            # Train forecaster
            forecaster.fit(X_train, y_train)
            
            # Multi-step ahead predictions
            fold_preds = []
            fold_actuals = []
            
            for test_point in adjusted_test_idx:
                # Use data up to test_point for prediction
                X_context = X.iloc[:test_point+1] if hasattr(X, 'iloc') else X[:test_point+1]
                
                # Predict horizon steps ahead
                pred = forecaster.predict(X_context, steps=horizon)
                actual = y.iloc[test_point + horizon] if hasattr(y, 'iloc') else y[test_point + horizon]
                
                fold_preds.append(pred[-1] if isinstance(pred, (list, np.ndarray)) else pred)
                fold_actuals.append(actual)
            
            if fold_preds:
                fold_score = self._calculate_forecast_metrics(fold_actuals, fold_preds)
                fold_scores.append(fold_score)
                fold_predictions.append((fold_actuals, fold_preds))
        
        # Aggregate fold results
        if fold_scores:
            aggregated_scores = {}
            for metric in fold_scores[0].keys():
                aggregated_scores[f'{metric}_mean'] = np.mean([score[metric] for score in fold_scores])
                aggregated_scores[f'{metric}_std'] = np.std([score[metric] for score in fold_scores])
        else:
            aggregated_scores = {}
        
        return {
            'scores': aggregated_scores,
            'fold_predictions': fold_predictions,
            'n_folds': len(fold_scores)
        }
    
    def _calculate_forecast_metrics(self, y_true, y_pred):
        """Calculate comprehensive forecasting metrics."""
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic error metrics  
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Directional accuracy
        direction_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) if len(y_true) > 1 else np.nan
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'direction_accuracy': direction_accuracy
        }
    
    def _summarize_multi_horizon_results(self, results):
        """Summarize results across all forecast horizons."""
        
        horizon_keys = [k for k in results.keys() if k.startswith('horizon_')]
        
        summary = {}
        
        # Average performance across horizons
        metrics = ['mae_mean', 'rmse_mean', 'mape_mean', 'direction_accuracy_mean']
        
        for metric in metrics:
            metric_values = []
            for horizon_key in horizon_keys:
                if metric in results[horizon_key]['scores']:
                    metric_values.append(results[horizon_key]['scores'][metric])
            
            if metric_values:
                summary[f'avg_{metric}'] = np.mean(metric_values)
                summary[f'std_{metric}'] = np.std(metric_values)
        
        # Horizon-specific best performance
        for metric in ['mae_mean', 'rmse_mean', 'mape_mean']:
            best_horizon = None
            best_value = float('inf')
            
            for horizon_key in horizon_keys:
                if metric in results[horizon_key]['scores']:
                    value = results[horizon_key]['scores'][metric]
                    if value < best_value:
                        best_value = value
                        best_horizon = horizon_key
            
            summary[f'best_{metric}_horizon'] = best_horizon
            summary[f'best_{metric}_value'] = best_value
        
        return summary
