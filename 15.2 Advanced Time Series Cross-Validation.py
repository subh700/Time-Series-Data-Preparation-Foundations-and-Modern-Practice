import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Optional, Union, Callable
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod
import warnings

class TimeSeriesCrossValidator(ABC):
    """
    Abstract base class for time series cross-validation strategies.
    """
    
    @abstractmethod
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test splits for cross-validation."""
        pass
    
    @abstractmethod
    def get_n_splits(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Get number of splits."""
        pass


class WalkForwardValidator(TimeSeriesCrossValidator):
    """
    Walk-forward validation with expanding or rolling window.
    """
    
    def __init__(self, 
                 min_train_size: int,
                 test_size: int = 1,
                 max_train_size: Optional[int] = None,
                 gap: int = 0,
                 expanding_window: bool = True,
                 step_size: int = 1):
        """
        Initialize walk-forward validator.
        
        Args:
            min_train_size: Minimum size of training set
            test_size: Size of test set
            max_train_size: Maximum size of training set (for rolling window)
            gap: Gap between training and test sets
            expanding_window: If True, use expanding window; if False, use rolling window
            step_size: Step size for moving the window
        """
        
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.step_size = step_size
        
        # Validation
        if min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if test_size <= 0:
            raise ValueError("test_size must be positive")
        if gap < 0:
            raise ValueError("gap must be non-negative")
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate walk-forward splits."""
        
        n_samples = len(X)
        
        # Starting position for first test set
        start_test = self.min_train_size + self.gap
        
        while start_test + self.test_size <= n_samples:
            # Test set indices
            test_end = start_test + self.test_size
            test_indices = np.arange(start_test, test_end)
            
            # Training set indices
            if self.expanding_window:
                # Expanding window: use all data from beginning
                train_end = start_test - self.gap
                train_indices = np.arange(0, train_end)
            else:
                # Rolling window: use fixed-size window
                if self.max_train_size is None:
                    train_size = self.min_train_size
                else:
                    train_size = min(self.max_train_size, start_test - self.gap)
                
                train_start = start_test - self.gap - train_size
                train_end = start_test - self.gap
                train_indices = np.arange(max(0, train_start), train_end)
            
            # Ensure minimum training size
            if len(train_indices) >= self.min_train_size:
                yield train_indices, test_indices
            
            # Move to next position
            start_test += self.step_size
    
    def get_n_splits(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Get number of splits."""
        
        n_samples = len(X)
        start_test = self.min_train_size + self.gap
        
        if start_test + self.test_size > n_samples:
            return 0
        
        n_splits = (n_samples - start_test - self.test_size) // self.step_size + 1
        return max(0, n_splits)


class TimeSeriesBacktester:
    """
    Comprehensive backtesting framework for time series models.
    """
    
    def __init__(self, 
                 cv_strategy: TimeSeriesCrossValidator,
                 refit: bool = True,
                 scoring: Union[str, Callable] = 'mae',
                 return_forecasts: bool = True,
                 verbose: bool = True):
        """
        Initialize backtester.
        
        Args:
            cv_strategy: Cross-validation strategy
            refit: Whether to refit model for each fold
            scoring: Scoring function or metric name
            return_forecasts: Whether to return individual forecasts
            verbose: Whether to print progress
        """
        
        self.cv_strategy = cv_strategy
        self.refit = refit
        self.scoring = scoring
        self.return_forecasts = return_forecasts
        self.verbose = verbose
        
        self.results_history = []
        
        # Initialize scoring function
        if isinstance(scoring, str):
            self.scoring_func = self._get_scoring_function(scoring)
        else:
            self.scoring_func = scoring
    
    def backtest(self, 
                 model: BaseEstimator,
                 X: np.ndarray,
                 y: np.ndarray,
                 **fit_params) -> Dict:
        """
        Perform backtesting on time series model.
        
        Args:
            model: Forecasting model
            X: Feature matrix
            y: Target values
            **fit_params: Additional parameters for model fitting
            
        Returns:
            Dictionary containing backtest results
        """
        
        scores = []
        forecasts = []
        fold_results = []
        
        n_splits = self.cv_strategy.get_n_splits(X, y)
        
        if self.verbose:
            print(f"Starting backtesting with {n_splits} folds...")
        
        for fold, (train_idx, test_idx) in enumerate(self.cv_strategy.split(X, y)):
            
            if self.verbose and fold % max(1, n_splits // 10) == 0:
                print(f"Processing fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Fit model
                if self.refit or fold == 0:
                    model.fit(X_train, y_train, **fit_params)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate score
                score = self.scoring_func(y_test, y_pred)
                scores.append(score)
                
                # Store fold results
                fold_result = {
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'score': score,
                    'train_period': (train_idx[0], train_idx[-1]),
                    'test_period': (test_idx[0], test_idx[-1])
                }
                
                if self.return_forecasts:
                    fold_result['predictions'] = y_pred
                    fold_result['actuals'] = y_test
                
                fold_results.append(fold_result)
                
            except Exception as e:
                warning_msg = f"Error in fold {fold}: {str(e)}"
                warnings.warn(warning_msg)
                scores.append(np.nan)
                
                fold_results.append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'score': np.nan,
                    'error': str(e)
                })
        
        # Aggregate results
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        results = {
            'scores': scores,
            'mean_score': np.mean(valid_scores) if valid_scores else np.nan,
            'std_score': np.std(valid_scores) if valid_scores else np.nan,
            'n_splits': n_splits,
            'n_successful': len(valid_scores),
            'fold_results': fold_results
        }
        
        # Store results
        self.results_history.append({
            'timestamp': pd.Timestamp.now(),
            'model_name': type(model).__name__,
            'results': results
        })
        
        if self.verbose:
            print(f"Backtesting completed. Mean score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        return results
    
    def _get_scoring_function(self, scoring: str) -> Callable:
        """Get scoring function by name."""
        
        scoring_functions = {
            'mae': lambda y_true, y_pred: -np.mean(np.abs(y_true - y_pred)),
            'mse': lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2),
            'rmse': lambda y_true, y_pred: -np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mape': lambda y_true, y_pred: -np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        
        if scoring not in scoring_functions:
            raise ValueError(f"Unknown scoring function: {scoring}")
        
        return scoring_functions[scoring]
    
    def compare_models(self, 
                      models: Dict[str, BaseEstimator],
                      X: np.ndarray,
                      y: np.ndarray,
                      **fit_params) -> pd.DataFrame:
        """
        Compare multiple models using backtesting.
        
        Args:
            models: Dictionary of model name -> model instance
            X: Feature matrix
            y: Target values
            **fit_params: Additional parameters for model fitting
            
        Returns:
            DataFrame with comparison results
        """
        
        comparison_results = []
        
        for model_name, model in models.items():
            if self.verbose:
                print(f"\nBacktesting {model_name}...")
            
            results = self.backtest(model, X, y, **fit_params)
            
            comparison_results.append({
                'Model': model_name,
                'Mean_Score': results['mean_score'],
                'Std_Score': results['std_score'],
                'N_Splits': results['n_splits'],
                'N_Successful': results['n_successful'],
                'Success_Rate': results['n_successful'] / results['n_splits'] * 100
            })
        
        df_results = pd.DataFrame(comparison_results)
        
        # Sort by mean score (higher is better for our negative scoring)
        df_results = df_results.sort_values('Mean_Score', ascending=False)
        
        return df_results
    
    def plot_performance_over_time(self, results: Dict, 
                                 figsize: Tuple[int, int] = (12, 6)):
        """Plot model performance over time."""
        
        try:
            import matplotlib.pyplot as plt
            
            fold_results = results['fold_results']
            
            # Extract time periods and scores
            test_periods = [fr['test_period'][0] for fr in fold_results if 'score' in fr]
            scores = [fr['score'] for fr in fold_results if 'score' in fr and not np.isnan(fr['score'])]
            
            if not scores:
                print("No valid scores to plot")
                return
            
            plt.figure(figsize=figsize)
            plt.plot(test_periods, scores, marker='o', linewidth=2, markersize=4)
            plt.title('Model Performance Over Time')
            plt.xlabel('Time Period (Test Set Start)')
            plt.ylabel('Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


class BlockingTimeSeriesSplitter(TimeSeriesCrossValidator):
    """
    Blocking time series splitter for handling correlated observations.
    """
    
    def __init__(self, n_splits: int = 5, block_size: int = 1):
        """
        Initialize blocking splitter.
        
        Args:
            n_splits: Number of splits
            block_size: Size of each block
        """
        
        self.n_splits = n_splits
        self.block_size = block_size
    
    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate blocked splits."""
        
        n_samples = len(X)
        
        # Create blocks
        n_blocks = n_samples // self.block_size
        block_indices = np.arange(n_blocks)
        
        # Shuffle blocks
        np.random.shuffle(block_indices)
        
        # Create splits
        test_size = n_blocks // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_blocks
            
            test_blocks = block_indices[test_start:test_end]
            train_blocks = np.concatenate([
                block_indices[:test_start],
                block_indices[test_end:]
            ])
            
            # Convert block indices to sample indices
            test_indices = []
            for block_idx in test_blocks:
                start_idx = block_idx * self.block_size
                end_idx = min((block_idx + 1) * self.block_size, n_samples)
                test_indices.extend(range(start_idx, end_idx))
            
            train_indices = []
            for block_idx in train_blocks:
                start_idx = block_idx * self.block_size
                end_idx = min((block_idx + 1) * self.block_size, n_samples)
                train_indices.extend(range(start_idx, end_idx))
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> int:
        """Get number of splits."""
        return self.n_splits
