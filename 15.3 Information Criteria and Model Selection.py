import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy import stats
from sklearn.base import BaseEstimator
import warnings

class ModelSelectionFramework:
    """
    Comprehensive framework for time series model selection using
    information criteria, cross-validation, and statistical tests.
    """
    
    def __init__(self, 
                 criteria: List[str] = None,
                 cv_strategy: Optional[TimeSeriesCrossValidator] = None,
                 significance_level: float = 0.05):
        """
        Initialize model selection framework.
        
        Args:
            criteria: List of information criteria to use
            cv_strategy: Cross-validation strategy
            significance_level: Significance level for statistical tests
        """
        
        self.criteria = criteria or ['AIC', 'BIC', 'HQIC']
        self.cv_strategy = cv_strategy
        self.significance_level = significance_level
        
        # Results storage
        self.selection_results = []
        self.model_comparisons = []
        
        # Initialize criteria calculators
        self.criteria_calculators = {
            'AIC': self._calculate_aic,
            'BIC': self._calculate_bic,
            'HQIC': self._calculate_hqic,
            'FPE': self._calculate_fpe,
            'AICC': self._calculate_aicc
        }
    
    def select_best_model(self, 
                         models: Dict[str, BaseEstimator],
                         X: np.ndarray,
                         y: np.ndarray,
                         selection_method: str = 'information_criteria',
                         **kwargs) -> Dict[str, Any]:
        """
        Select best model using specified method.
        
        Args:
            models: Dictionary mapping model names to model instances
            X: Feature matrix
            y: Target values
            selection_method: Method for model selection
            **kwargs: Additional arguments for selection method
            
        Returns:
            Dictionary with selection results
        """
        
        if selection_method == 'information_criteria':
            return self._select_by_information_criteria(models, X, y, **kwargs)
        elif selection_method == 'cross_validation':
            return self._select_by_cross_validation(models, X, y, **kwargs)
        elif selection_method == 'statistical_tests':
            return self._select_by_statistical_tests(models, X, y, **kwargs)
        elif selection_method == 'ensemble':
            return self._select_by_ensemble_method(models, X, y, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
    
    def _select_by_information_criteria(self, 
                                       models: Dict[str, BaseEstimator],
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       **kwargs) -> Dict[str, Any]:
        """Select model using information criteria."""
        
        criteria_results = {}
        model_scores = {}
        
        for model_name, model in models.items():
            # Fit model
            model.fit(X, y)
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(model, X, y)
            n_params = self._count_parameters(model)
            n_obs = len(y)
            
            model_scores[model_name] = {}
            
            # Calculate each criterion
            for criterion in self.criteria:
                if criterion in self.criteria_calculators:
                    score = self.criteria_calculators[criterion](
                        log_likelihood, n_params, n_obs
                    )
                    model_scores[model_name][criterion] = score
        
        # Select best model for each criterion
        best_models = {}
        
        for criterion in self.criteria:
            criterion_scores = {name: scores.get(criterion, float('inf')) 
                              for name, scores in model_scores.items()}
            
            best_model = min(criterion_scores.items(), key=lambda x: x[1])
            best_models[criterion] = {
                'model_name': best_model[0],
                'score': best_model[1]
            }
        
        # Overall ranking using multiple criteria
        overall_ranking = self._rank_models_by_criteria(model_scores)
        
        result = {
            'method': 'information_criteria',
            'model_scores': model_scores,
            'best_by_criterion': best_models,
            'overall_ranking': overall_ranking,
            'recommended_model': overall_ranking[0]['model_name']
        }
        
        self.selection_results.append(result)
        
        return result
    
    def _select_by_cross_validation(self, 
                                   models: Dict[str, BaseEstimator],
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   scoring: str = 'mae',
                                   **kwargs) -> Dict[str, Any]:
        """Select model using cross-validation."""
        
        if self.cv_strategy is None:
            raise ValueError("Cross-validation strategy not specified")
        
        backtester = TimeSeriesBacktester(
            cv_strategy=self.cv_strategy,
            scoring=scoring,
            verbose=False
        )
        
        cv_results = {}
        
        for model_name, model in models.items():
            result = backtester.backtest(model, X, y)
            cv_results[model_name] = {
                'mean_score': result['mean_score'],
                'std_score': result['std_score'],
                'n_splits': result['n_splits'],
                'scores': result['scores']
            }
        
        # Select best model (highest score for our negative scoring)
        best_model = max(cv_results.items(), 
                        key=lambda x: x[1]['mean_score'])
        
        # Rank all models
        ranking = sorted(cv_results.items(), 
                        key=lambda x: x[1]['mean_score'], 
                        reverse=True)
        
        result = {
            'method': 'cross_validation',
            'cv_results': cv_results,
            'best_model': {
                'model_name': best_model[0],
                'mean_score': best_model[1]['mean_score'],
                'std_score': best_model[1]['std_score']
            },
            'ranking': [{'model_name': name, **scores} for name, scores in ranking],
            'recommended_model': best_model[0]
        }
        
        self.selection_results.append(result)
        
        return result
    
    def _select_by_statistical_tests(self, 
                                    models: Dict[str, BaseEstimator],
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    baseline_model: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """Select model using statistical significance tests."""
        
        # Fit all models and get predictions
        model_predictions = {}
        model_errors = {}
        
        for model_name, model in models.items():
            model.fit(X, y)
            predictions = model.predict(X)
            errors = y - predictions
            
            model_predictions[model_name] = predictions
            model_errors[model_name] = errors
        
        # Determine baseline model
        if baseline_model is None:
            # Use simplest model or first model as baseline
            baseline_model = list(models.keys())[0]
        
        if baseline_model not in model_errors:
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        
        # Perform pairwise statistical tests
        test_suite = StatisticalTestSuite()
        test_results = {}
        
        baseline_errors = model_errors[baseline_model]
        
        for model_name, errors in model_errors.items():
            if model_name != baseline_model:
                # Diebold-Mariano test
                dm_result = test_suite.diebold_mariano_test(
                    baseline_errors, errors
                )
                
                # Wilcoxon signed-rank test
                wilcoxon_result = test_suite.wilcoxon_signed_rank_test(
                    baseline_errors, errors
                )
                
                test_results[model_name] = {
                    'dm_test': dm_result,
                    'wilcoxon_test': wilcoxon_result,
                    'significantly_better': (
                        dm_result['p_value'] < self.significance_level and
                        dm_result['statistic'] > 0
                    )
                }
        
        # Identify significantly better models
        better_models = [name for name, results in test_results.items() 
                        if results['significantly_better']]
        
        # SPA test if multiple models
        if len(model_errors) > 2:
            alternative_errors = [errors for name, errors in model_errors.items() 
                                if name != baseline_model]
            
            spa_result = test_suite.superior_predictive_ability_test(
                baseline_errors, *alternative_errors
            )
        else:
            spa_result = None
        
        result = {
            'method': 'statistical_tests',
            'baseline_model': baseline_model,
            'test_results': test_results,
            'significantly_better_models': better_models,
            'spa_test': spa_result,
            'recommended_model': better_models[0] if better_models else baseline_model
        }
        
        self.selection_results.append(result)
        
        return result
    
    def _calculate_aic(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Calculate Akaike Information Criterion."""
        return -2 * log_likelihood + 2 * n_params
    
    def _calculate_bic(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Calculate Bayesian Information Criterion."""
        return -2 * log_likelihood + n_params * np.log(n_obs)
    
    def _calculate_hqic(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Calculate Hannan-Quinn Information Criterion."""
        return -2 * log_likelihood + 2 * n_params * np.log(np.log(n_obs))
    
    def _calculate_aicc(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Calculate corrected AIC for small samples."""
        aic = self._calculate_aic(log_likelihood, n_params, n_obs)
        correction = (2 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
        return aic + correction
    
    def _calculate_fpe(self, log_likelihood: float, n_params: int, n_obs: int) -> float:
        """Calculate Final Prediction Error."""
        return ((n_obs + n_params) / (n_obs - n_params)) * np.exp(-2 * log_likelihood / n_obs)
    
    def _calculate_log_likelihood(self, model: BaseEstimator, 
                                 X: np.ndarray, y: np.ndarray) -> float:
        """Calculate log-likelihood for fitted model."""
        
        # Get predictions
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Estimate variance
        sigma_squared = np.var(residuals, ddof=1)
        
        # Gaussian log-likelihood
        log_likelihood = -0.5 * len(y) * (np.log(2 * np.pi) + np.log(sigma_squared)) - \
                        0.5 * np.sum(residuals**2) / sigma_squared
        
        return log_likelihood
    
    def _count_parameters(self, model: BaseEstimator) -> int:
        """Estimate number of parameters in model."""
        
        # This is a simplified approach - in practice, this would need to be
        # implemented specifically for each model type
        
        if hasattr(model, 'coef_'):
            n_params = len(np.atleast_1d(model.coef_))
            if hasattr(model, 'intercept_'):
                n_params += 1
            return n_params
        elif hasattr(model, 'n_parameters_'):
            return model.n_parameters_
        else:
            # Default estimate
            warnings.warn("Could not determine number of parameters, using default estimate")
            return 3
    
    def _rank_models_by_criteria(self, model_scores: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Rank models using multiple criteria with Borda count."""
        
        # Get rankings for each criterion
        criterion_rankings = {}
        
        for criterion in self.criteria:
            # Sort models by criterion score (lower is better)
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1].get(criterion, float('inf'))
            )
            
            criterion_rankings[criterion] = {
                model_name: rank for rank, (model_name, _) in enumerate(sorted_models)
            }
        
        # Calculate Borda count for each model
        borda_scores = {}
        for model_name in model_scores.keys():
            borda_score = sum(
                criterion_rankings[criterion].get(model_name, len(model_scores))
                for criterion in self.criteria
            )
            borda_scores[model_name] = borda_score
        
        # Sort by Borda count (lower is better)
        final_ranking = sorted(borda_scores.items(), key=lambda x: x[1])
        
        return [
            {
                'model_name': model_name,
                'borda_score': score,
                'rank': rank + 1
            }
            for rank, (model_name, score) in enumerate(final_ranking)
        ]
    
    def get_selection_summary(self) -> pd.DataFrame:
        """Get summary of all model selection results."""
        
        if not self.selection_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.selection_results:
            summary_data.append({
                'Method': result['method'],
                'Recommended_Model': result['recommended_model'],
                'Timestamp': result.get('timestamp', pd.Timestamp.now())
            })
        
        return pd.DataFrame(summary_data)
