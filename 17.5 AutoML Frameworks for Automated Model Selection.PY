import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AutoMLConfig:
    """Configuration for AutoML time series forecasting."""
    
    # Search space and optimization
    optimization_metric: str = "mse"  # mse, mae, mape, smape
    optimization_direction: str = "minimize"
    max_trials: int = 100
    timeout_seconds: int = 3600  # 1 hour
    n_jobs: int = -1
    
    # Cross-validation settings
    n_splits: int = 5
    test_size: float = 0.2
    validation_method: str = "time_series_split"  # time_series_split, sliding_window
    
    # Model selection
    include_models: List[str] = field(default_factory=lambda: [
        "linear_regression", "ridge", "lasso", "random_forest", 
        "gradient_boosting", "xgboost", "lightgbm", "arima", "sarima"
    ])
    
    # Feature engineering
    enable_feature_engineering: bool = True
    max_lag_features: int = 20
    enable_rolling_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])
    enable_seasonal_features: bool = True
    
    # Ensemble settings
    enable_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # weighted_average, stacking
    max_ensemble_size: int = 5
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 20
    
    # Preprocessing
    scaling_method: str = "standard"  # standard, robust, minmax, none
    handle_missing: str = "interpolate"  # interpolate, forward_fill, drop
    
    # Advanced settings
    enable_hyperparameter_tuning: bool = True
    tuning_budget_ratio: float = 0.7  # Fraction of time for hyperparameter tuning
    enable_meta_learning: bool = False
    meta_features_path: Optional[str] = None


class TimeSeriesAutoML:
    """
    Comprehensive AutoML system for time series forecasting.
    Automates the entire pipeline from data preprocessing to model selection.
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.preprocessor = AutoMLPreprocessor(config)
        self.feature_engineer = AutoMLFeatureEngineer(config)
        self.model_selector = AutoMLModelSelector(config)
        self.ensemble_builder = AutoMLEnsembleBuilder(config)
        self.evaluator = AutoMLEvaluator(config)
        
        # Results storage
        self.best_model = None
        self.best_score = None
        self.experiment_history = []
        self.feature_importance = {}
        
        # Meta-learning (if enabled)
        if config.enable_meta_learning:
            self.meta_learner = MetaLearner(config)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for AutoML system."""
        
        logger = logging.getLogger("TimeSeriesAutoML")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            time_column: Optional[str] = None,
            validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Fit AutoML system to time series data.
        
        Args:
            train_data: Training time series data
            target_column: Name of target column to forecast
            time_column: Name of time column (optional)
            validation_data: Optional validation data
            
        Returns:
            Fitting results and best model information
        """
        
        start_time = time.time()
        self.logger.info("Starting AutoML time series forecasting pipeline")
        
        # Validate input data
        self._validate_input_data(train_data, target_column, time_column)
        
        # Data preprocessing
        self.logger.info("Step 1: Data preprocessing")
        preprocessed_data = self.preprocessor.fit_transform(
            train_data, target_column, time_column
        )
        
        # Feature engineering
        self.logger.info("Step 2: Feature engineering")
        engineered_data = self.feature_engineer.fit_transform(
            preprocessed_data, target_column
        )
        
        # Meta-learning initialization (if enabled)
        if self.config.enable_meta_learning:
            self.logger.info("Step 2.5: Meta-learning initialization")
            meta_features = self._extract_meta_features(engineered_data, target_column)
            recommended_models = self.meta_learner.recommend_models(meta_features)
            self.config.include_models = recommended_models
        
        # Model selection and hyperparameter optimization
        self.logger.info("Step 3: Model selection and hyperparameter tuning")
        
        if self.config.enable_hyperparameter_tuning:
            tuning_time_budget = int(
                self.config.timeout_seconds * self.config.tuning_budget_ratio
            )
            
            best_models = self.model_selector.search_and_optimize(
                engineered_data, 
                target_column,
                time_budget=tuning_time_budget,
                validation_data=validation_data
            )
        else:
            best_models = self.model_selector.train_default_models(
                engineered_data, target_column
            )
        
        # Ensemble building
        if self.config.enable_ensemble and len(best_models) > 1:
            self.logger.info("Step 4: Ensemble building")
            self.best_model = self.ensemble_builder.build_ensemble(
                best_models, engineered_data, target_column
            )
        else:
            self.best_model = best_models[0] if best_models else None
        
        # Final evaluation
        self.logger.info("Step 5: Final model evaluation")
        if self.best_model:
            evaluation_results = self.evaluator.comprehensive_evaluation(
                self.best_model, engineered_data, target_column
            )
            self.best_score = evaluation_results['primary_metric']
        
        # Store results
        total_time = time.time() - start_time
        
        results = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'total_time_seconds': total_time,
            'preprocessing_steps': self.preprocessor.get_steps_summary(),
            'feature_engineering_steps': self.feature_engineer.get_steps_summary(),
            'models_evaluated': len(self.experiment_history),
            'evaluation_results': evaluation_results if self.best_model else None
        }
        
        self.logger.info(f"AutoML pipeline completed in {total_time:.2f} seconds")
        self.logger.info(f"Best model: {type(self.best_model).__name__}")
        self.logger.info(f"Best score ({self.config.optimization_metric}): {self.best_score:.6f}")
        
        return results
    
    def predict(self, 
                test_data: pd.DataFrame,
                forecast_horizon: int = 1) -> np.ndarray:
        """
        Generate forecasts using the trained AutoML model.
        
        Args:
            test_data: Test data for forecasting
            forecast_horizon: Number of steps to forecast ahead
            
        Returns:
            Forecast predictions
        """
        
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Apply same preprocessing and feature engineering
        preprocessed_data = self.preprocessor.transform(test_data)
        engineered_data = self.feature_engineer.transform(preprocessed_data)
        
        # Generate predictions
        if hasattr(self.best_model, 'predict_multi_step'):
            predictions = self.best_model.predict_multi_step(
                engineered_data, forecast_horizon
            )
        else:
            predictions = self.best_model.predict(engineered_data)
        
        return predictions
    
    def explain_model(self) -> Dict[str, Any]:
        """
        Provide explanations for the selected model and features.
        
        Returns:
            Model explanation including feature importance and decision rationale
        """
        
        if self.best_model is None:
            return {"error": "No model trained"}
        
        explanations = {
            'model_type': type(self.best_model).__name__,
            'model_parameters': getattr(self.best_model, 'get_params', lambda: {})(),
            'feature_importance': self.feature_importance,
            'preprocessing_steps': self.preprocessor.get_steps_summary(),
            'feature_engineering_rationale': self.feature_engineer.get_rationale(),
            'selection_criteria': {
                'optimization_metric': self.config.optimization_metric,
                'cross_validation_method': self.config.validation_method,
                'final_score': self.best_score
            }
        }
        
        # Add model-specific explanations
        if hasattr(self.best_model, 'explain'):
            explanations['model_specific'] = self.best_model.explain()
        
        return explanations
    
    def _validate_input_data(self, 
                           data: pd.DataFrame, 
                           target_column: str, 
                           time_column: Optional[str]):
        """Validate input data format and content."""
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if data[target_column].isnull().all():
            raise ValueError("Target column contains only missing values")
        
        if len(data) < 10:
            raise ValueError("Insufficient data points for time series analysis")
        
        if time_column and time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
    
    def _extract_meta_features(self, 
                             data: pd.DataFrame, 
                             target_column: str) -> Dict[str, float]:
        """Extract meta-features for meta-learning."""
        
        target_series = data[target_column].dropna()
        
        meta_features = {
            'n_observations': len(target_series),
            'mean': target_series.mean(),
            'std': target_series.std(),
            'skewness': target_series.skew(),
            'kurtosis': target_series.kurtosis(),
            'trend_strength': self._calculate_trend_strength(target_series),
            'seasonal_strength': self._calculate_seasonal_strength(target_series),
            'stability': 1 / (1 + target_series.std() / abs(target_series.mean() + 1e-8)),
            'linearity': self._calculate_linearity(target_series),
            'n_features': len(data.columns) - 1
        }
        
        return meta_features
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength using linear regression."""
        
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series))
        try:
            correlation = np.corrcoef(x, series)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_seasonal_strength(self, series: pd.Series, period: int = 12) -> float:
        """Calculate seasonal strength using autocorrelation."""
        
        if len(series) < 2 * period:
            return 0.0
        
        try:
            autocorr = np.corrcoef(
                series[:-period].values, 
                series[period:].values
            )[0, 1]
            return abs(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _calculate_linearity(self, series: pd.Series) -> float:
        """Calculate linearity of the time series."""
        
        if len(series) < 3:
            return 0.0
        
        x = np.arange(len(series))
        try:
            # Fit linear and quadratic models
            linear_coef = np.polyfit(x, series, 1)
            quad_coef = np.polyfit(x, series, 2)
            
            linear_pred = np.polyval(linear_coef, x)
            quad_pred = np.polyval(quad_coef, x)
            
            linear_error = np.mean((series - linear_pred) ** 2)
            quad_error = np.mean((series - quad_pred) ** 2)
            
            # Linearity is how much linear model explains vs quadratic
            linearity = 1 - (linear_error / (quad_error + 1e-8))
            return max(0, min(1, linearity))
        except:
            return 0.0


class AutoMLPreprocessor:
    """Automated preprocessing for time series data."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.steps_applied = []
        self.scaler = None
        
    def fit_transform(self, 
                     data: pd.DataFrame, 
                     target_column: str, 
                     time_column: Optional[str] = None) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        
        processed_data = data.copy()
        
        # Handle time column
        if time_column:
            processed_data[time_column] = pd.to_datetime(processed_data[time_column])
            processed_data = processed_data.sort_values(time_column)
            self.steps_applied.append("sort_by_time")
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data, target_column)
        
        # Detect and handle outliers
        processed_data = self._handle_outliers(processed_data, target_column)
        
        # Scale features (but not target initially)
        feature_columns = [col for col in processed_data.columns 
                          if col != target_column and col != time_column]
        
        if feature_columns and self.config.scaling_method != "none":
            processed_data[feature_columns] = self._fit_transform_scaler(
                processed_data[feature_columns]
            )
        
        return processed_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor."""
        
        processed_data = data.copy()
        
        # Apply same transformations (simplified version)
        if self.scaler:
            feature_columns = [col for col in processed_data.columns 
                             if col in self.scaler.feature_names_in_]
            if feature_columns:
                processed_data[feature_columns] = self.scaler.transform(
                    processed_data[feature_columns]
                )
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        
        if self.config.handle_missing == "interpolate":
            # Time series interpolation
            data[target_column] = data[target_column].interpolate(method='time')
            
            # Forward fill remaining
            data = data.fillna(method='ffill')
            
            # Backward fill any remaining
            data = data.fillna(method='bfill')
            
            self.steps_applied.append("interpolate_missing")
            
        elif self.config.handle_missing == "forward_fill":
            data = data.fillna(method='ffill')
            self.steps_applied.append("forward_fill_missing")
            
        elif self.config.handle_missing == "drop":
            data = data.dropna()
            self.steps_applied.append("drop_missing")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Detect and handle outliers in target variable."""
        
        target_series = data[target_column]
        
        # Use IQR method for outlier detection
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = (target_series < lower_bound) | (target_series > upper_bound)
        n_outliers = outliers.sum()
        
        if n_outliers > len(data) * 0.05:  # More than 5% outliers
            # Cap extreme values instead of removing
            data[target_column] = np.clip(
                data[target_column], 
                lower_bound, 
                upper_bound
            )
            self.steps_applied.append(f"cap_outliers_{n_outliers}")
        
        return data
    
    def _fit_transform_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data using specified scaling method."""
        
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        elif self.config.scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        scaled_data = self.scaler.fit_transform(data)
        self.steps_applied.append(f"scaling_{self.config.scaling_method}")
        
        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    def get_steps_summary(self) -> List[str]:
        """Get summary of preprocessing steps applied."""
        return self.steps_applied.copy()


class AutoMLFeatureEngineer:
    """Automated feature engineering for time series."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.features_created = []
        self.target_column = None
        
    def fit_transform(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create time series features automatically."""
        
        self.target_column = target_column
        engineered_data = data.copy()
        
        if not self.config.enable_feature_engineering:
            return engineered_data
        
        # Lag features
        engineered_data = self._create_lag_features(engineered_data, target_column)
        
        # Rolling window features
        if self.config.enable_rolling_features:
            engineered_data = self._create_rolling_features(engineered_data, target_column)
        
        # Seasonal features
        if self.config.enable_seasonal_features:
            engineered_data = self._create_seasonal_features(engineered_data)
        
        # Difference features
        engineered_data = self._create_difference_features(engineered_data, target_column)
        
        # Remove rows with NaN created by feature engineering
        engineered_data = engineered_data.dropna()
        
        return engineered_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data with same feature engineering."""
        
        # Apply same feature engineering steps
        # This is a simplified version - in practice, you'd store
        # the exact transformation parameters
        
        return self.fit_transform(data, self.target_column)
    
    def _create_lag_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create lagged features of the target variable."""
        
        max_lags = min(self.config.max_lag_features, len(data) // 4)
        
        for lag in range(1, max_lags + 1):
            feature_name = f"{target_column}_lag_{lag}"
            data[feature_name] = data[target_column].shift(lag)
            self.features_created.append(feature_name)
        
        return data
    
    def _create_rolling_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create rolling window statistical features."""
        
        for window in self.config.rolling_windows:
            if window < len(data):
                # Rolling mean
                feature_name = f"{target_column}_rolling_mean_{window}"
                data[feature_name] = data[target_column].rolling(window=window).mean()
                self.features_created.append(feature_name)
                
                # Rolling std
                feature_name = f"{target_column}_rolling_std_{window}"
                data[feature_name] = data[target_column].rolling(window=window).std()
                self.features_created.append(feature_name)
                
                # Rolling min/max
                feature_name = f"{target_column}_rolling_min_{window}"
                data[feature_name] = data[target_column].rolling(window=window).min()
                self.features_created.append(feature_name)
                
                feature_name = f"{target_column}_rolling_max_{window}"
                data[feature_name] = data[target_column].rolling(window=window).max()
                self.features_created.append(feature_name)
        
        return data
    
    def _create_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal/cyclical features."""
        
        # Assume we have an index that represents time periods
        if isinstance(data.index, pd.DatetimeIndex):
            # Hour of day (if hourly data)
            if data.index.freq and 'H' in str(data.index.freq):
                data['hour_of_day'] = data.index.hour
                data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
                self.features_created.extend(['hour_of_day', 'hour_sin', 'hour_cos'])
            
            # Day of week
            data['day_of_week'] = data.index.dayofweek
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            self.features_created.extend(['day_of_week', 'is_weekend'])
            
            # Month
            data['month'] = data.index.month
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            self.features_created.extend(['month', 'month_sin', 'month_cos'])
        
        else:
            # Create simple cyclical features based on row index
            n_periods = 24  # Assume daily cycle
            data['cycle_sin'] = np.sin(2 * np.pi * np.arange(len(data)) / n_periods)
            data['cycle_cos'] = np.cos(2 * np.pi * np.arange(len(data)) / n_periods)
            self.features_created.extend(['cycle_sin', 'cycle_cos'])
        
        return data
    
    def _create_difference_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create differenced features."""
        
        # First difference
        feature_name = f"{target_column}_diff_1"
        data[feature_name] = data[target_column].diff(1)
        self.features_created.append(feature_name)
        
        # Second difference
        feature_name = f"{target_column}_diff_2"
        data[feature_name] = data[target_column].diff(2)
        self.features_created.append(feature_name)
        
        return data
    
    def get_steps_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering steps."""
        
        return {
            'features_created': len(self.features_created),
            'feature_types': {
                'lag_features': len([f for f in self.features_created if 'lag' in f]),
                'rolling_features': len([f for f in self.features_created if 'rolling' in f]),
                'seasonal_features': len([f for f in self.features_created if any(s in f for s in ['hour', 'day', 'month', 'cycle'])]),
                'difference_features': len([f for f in self.features_created if 'diff' in f])
            },
            'total_features': len(self.features_created)
        }
    
    def get_rationale(self) -> Dict[str, str]:
        """Get rationale for feature engineering decisions."""
        
        return {
            'lag_features': "Capture temporal dependencies and autoregressive patterns",
            'rolling_features': "Capture local trends and smooth out noise",
            'seasonal_features': "Capture cyclical patterns (daily, weekly, monthly)",
            'difference_features': "Capture rate of change and stationarity"
        }


class AutoMLModelSelector:
    """Automated model selection and hyperparameter optimization."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.study = None
        self.best_models = []
        
    def search_and_optimize(self, 
                          data: pd.DataFrame, 
                          target_column: str,
                          time_budget: int,
                          validation_data: Optional[pd.DataFrame] = None) -> List[Any]:
        """Search for best models with hyperparameter optimization."""
        
        # Create Optuna study
        self.study = optuna.create_study(direction=self.config.optimization_direction)
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, data, target_column, validation_data)
        
        # Optimize with time budget
        self.study.optimize(
            objective, 
            n_trials=self.config.max_trials,
            timeout=time_budget
        )
        
        # Get best models from top trials
        best_trials = sorted(
            self.study.trials, 
            key=lambda t: t.value if t.value is not None else float('inf')
        )[:self.config.max_ensemble_size]
        
        # Train final models with best parameters
        self.best_models = []
        for trial in best_trials:
            if trial.value is not None:
                model = self._create_model_from_trial(trial, data, target_column)
                self.best_models.append(model)
        
        return self.best_models
    
    def train_default_models(self, data: pd.DataFrame, target_column: str) -> List[Any]:
        """Train models with default parameters (no optimization)."""
        
        self.best_models = []
        
        # Prepare data
        X, y = self._prepare_data(data, target_column)
        
        # Train each model type with defaults
        for model_name in self.config.include_models:
            try:
                model = self._create_default_model(model_name)
                if model:
                    model.fit(X, y)
                    self.best_models.append({
                        'model': model,
                        'name': model_name,
                        'score': self._evaluate_model(model, X, y)
                    })
            except Exception as e:
                logging.warning(f"Failed to train {model_name}: {str(e)}")
        
        # Sort by score
        self.best_models.sort(key=lambda x: x['score'])
        
        return [m['model'] for m in self.best_models]
    
    def _objective_function(self, 
                          trial: optuna.Trial, 
                          data: pd.DataFrame, 
                          target_column: str,
                          validation_data: Optional[pd.DataFrame] = None) -> float:
        """Objective function for hyperparameter optimization."""
        
        try:
            # Select model type
            model_name = trial.suggest_categorical('model_type', self.config.include_models)
            
            # Create model with trial parameters
            model = self._create_model_with_trial_params(trial, model_name)
            
            if model is None:
                return float('inf')
            
            # Prepare data
            X, y = self._prepare_data(data, target_column)
            
            # Cross-validation evaluation
            score = self._cross_validate_model(model, X, y)
            
            return score
            
        except Exception as e:
            logging.warning(f"Trial failed: {str(e)}")
            return float('inf')
    
    def _create_model_with_trial_params(self, trial: optuna.Trial, model_name: str):
        """Create model with hyperparameters suggested by trial."""
        
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=trial.suggest_int('rf_n_estimators', 10, 200),
                max_depth=trial.suggest_int('rf_max_depth', 3, 20),
                min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
                random_state=42
            )
        
        elif model_name == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=trial.suggest_int('gb_n_estimators', 10, 200),
                max_depth=trial.suggest_int('gb_max_depth', 3, 10),
                learning_rate=trial.suggest_float('gb_learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('gb_subsample', 0.6, 1.0),
                random_state=42
            )
        
        elif model_name == "ridge":
            return Ridge(
                alpha=trial.suggest_float('ridge_alpha', 0.01, 100.0, log=True)
            )
        
        elif model_name == "lasso":
            return Lasso(
                alpha=trial.suggest_float('lasso_alpha', 0.01, 10.0, log=True)
            )
        
        elif model_name == "linear_regression":
            return LinearRegression()
        
        # Add more models as needed
        return None
    
    def _create_default_model(self, model_name: str):
        """Create model with default parameters."""
        
        model_defaults = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        return model_defaults.get(model_name)
    
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training."""
        
        # Remove target column and any non-numeric columns
        feature_columns = [col for col in data.columns 
                          if col != target_column and data[col].dtype in ['int64', 'float64']]
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        return X, y
    
    def _cross_validate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-validate model performance."""
        
        if self.config.validation_method == "time_series_split":
            cv = TimeSeriesSplit(n_splits=self.config.n_splits)
        else:
            # Default to time series split
            cv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate score
            if self.config.optimization_metric == "mse":
                score = mean_squared_error(y_val, y_pred)
            elif self.config.optimization_metric == "mae":
                score = mean_absolute_error(y_val, y_pred)
            else:
                score = mean_squared_error(y_val, y_pred)  # Default
            
            scores.append(score)
        
        return np.mean(scores)
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance."""
        
        # Simple train-test split evaluation
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if self.config.optimization_metric == "mse":
            return mean_squared_error(y_test, y_pred)
        elif self.config.optimization_metric == "mae":
            return mean_absolute_error(y_test, y_pred)
        else:
            return mean_squared_error(y_test, y_pred)
    
    def _create_model_from_trial(self, trial: optuna.Trial, data: pd.DataFrame, target_column: str):
        """Create final model from best trial."""
        
        model_name = trial.params['model_type']
        
        # Recreate model with best parameters
        model = self._create_model_with_trial_params(trial, model_name)
        
        if model:
            X, y = self._prepare_data(data, target_column)
            model.fit(X, y)
        
        return model


class AutoMLEnsembleBuilder:
    """Build ensemble models from individual models."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        
    def build_ensemble(self, models: List[Any], data: pd.DataFrame, target_column: str):
        """Build ensemble from individual models."""
        
        if self.config.ensemble_method == "weighted_average":
            return self._build_weighted_average_ensemble(models, data, target_column)
        elif self.config.ensemble_method == "stacking":
            return self._build_stacking_ensemble(models, data, target_column)
        else:
            # Default to weighted average
            return self._build_weighted_average_ensemble(models, data, target_column)
    
    def _build_weighted_average_ensemble(self, models: List[Any], data: pd.DataFrame, target_column: str):
        """Build weighted average ensemble."""
        
        X, y = self._prepare_data(data, target_column)
        
        # Calculate weights based on individual model performance
        weights = []
        for model in models:
            # Evaluate model performance
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            # Weight is inverse of MSE (better models get higher weight)
            weight = 1.0 / (mse + 1e-8)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return WeightedAverageEnsemble(models, weights)
    
    def _build_stacking_ensemble(self, models: List[Any], data: pd.DataFrame, target_column: str):
        """Build stacking ensemble."""
        
        # This would implement a more sophisticated stacking approach
        # For now, fall back to weighted average
        return self._build_weighted_average_ensemble(models, data, target_column)
    
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ensemble building."""
        
        feature_columns = [col for col in data.columns 
                          if col != target_column and data[col].dtype in ['int64', 'float64']]
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        return X, y


class WeightedAverageEnsemble:
    """Simple weighted average ensemble model."""
    
    def __init__(self, models: List[Any], weights: List[float]):
        self.models = models
        self.weights = weights
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def get_params(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        
        return {
            'ensemble_type': 'weighted_average',
            'n_models': len(self.models),
            'weights': self.weights,
            'model_types': [type(model).__name__ for model in self.models]
        }


class AutoMLEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        
    def comprehensive_evaluation(self, model, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        
        X, y = self._prepare_data(data, target_column)
        
        # Generate predictions
        y_pred = model.predict(X)
        
        # Calculate various metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100,
            'r2': 1 - np.var(y - y_pred) / np.var(y)
        }
        
        # Primary metric for comparison
        metrics['primary_metric'] = metrics[self.config.optimization_metric]
        
        # Residual analysis
        residuals = y - y_pred
        metrics['residual_analysis'] = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'residual_skewness': pd.Series(residuals).skew() if len(residuals) > 3 else 0
        }
        
        return metrics
    
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation."""
        
        feature_columns = [col for col in data.columns 
                          if col != target_column and data[col].dtype in ['int64', 'float64']]
        
        X = data[feature_columns].values
        y = data[target_column].values
        
        return X, y


class MetaLearner:
    """Meta-learning for model recommendation."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.meta_features_db = {}
        self.model_performance_db = {}
        
        # Load meta-features database if available
        if config.meta_features_path:
            self._load_meta_database(config.meta_features_path)
    
    def recommend_models(self, meta_features: Dict[str, float]) -> List[str]:
        """Recommend models based on meta-features."""
        
        # Simple rule-based recommendations for now
        # In practice, this would use a trained meta-model
        
        recommendations = []
        
        # Based on data size
        if meta_features['n_observations'] < 100:
            recommendations.extend(['linear_regression', 'ridge'])
        elif meta_features['n_observations'] < 1000:
            recommendations.extend(['random_forest', 'gradient_boosting'])
        else:
            recommendations.extend(['gradient_boosting', 'xgboost', 'lightgbm'])
        
        # Based on trend strength
        if meta_features['trend_strength'] > 0.7:
            recommendations.append('linear_regression')
        
        # Based on seasonal strength
        if meta_features['seasonal_strength'] > 0.5:
            recommendations.extend(['arima', 'sarima'])
        
        # Ensure we have at least some models
        if not recommendations:
            recommendations = ['random_forest', 'gradient_boosting']
        
        # Remove duplicates and limit
        recommendations = list(set(recommendations))
        return recommendations[:5]  # Max 5 recommendations
    
    def _load_meta_database(self, path: str):
        """Load meta-features database."""
        try:
            with open(path, 'r') as f:
                db = json.load(f)
                self.meta_features_db = db.get('meta_features', {})
                self.model_performance_db = db.get('model_performance', {})
        except Exception as e:
            logging.warning(f"Could not load meta-features database: {str(e)}")


# Example usage and demonstration
def demonstrate_automl_time_series():
    """Demonstrate AutoML time series forecasting."""
    
    print("Generating synthetic time series data...")
    
    # Generate synthetic time series
    np.random.seed(42)
    n_points = 1000
    
    # Create time index
    dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    # Generate time series with trend, seasonality, and noise
    t = np.arange(n_points)
    trend = 0.001 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 365.25) + np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 0.5, n_points)
    
    target = 100 + trend + seasonal + noise
    
    # Add some external features
    feature1 = np.random.normal(0, 1, n_points)
    feature2 = 0.5 * target + np.random.normal(0, 2, n_points)  # Correlated with target
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'target': target,
        'feature1': feature1,
        'feature2': feature2
    })
    
    # Add some missing values
    missing_indices = np.random.choice(n_points, size=50, replace=False)
    data.loc[missing_indices, 'target'] = np.nan
    
    print(f"Created dataset with {len(data)} observations")
    print(f"Missing values: {data['target'].isnull().sum()}")
    
    # Split into train and test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()
    
    print(f"Training set: {len(train_data)} observations")
    print(f"Test set: {len(test_data)} observations")
    
    # Configure AutoML
    config = AutoMLConfig(
        optimization_metric="mae",
        max_trials=50,
        timeout_seconds=300,  # 5 minutes
        include_models=["linear_regression", "ridge", "random_forest", "gradient_boosting"],
        enable_feature_engineering=True,
        enable_ensemble=True,
        enable_early_stopping=True
    )
    
    # Initialize and train AutoML
    print("\nStarting AutoML training...")
    automl = TimeSeriesAutoML(config)
    
    # Fit the model
    results = automl.fit(
        train_data=train_data,
        target_column='target',
        time_column='date'
    )
    
    print("\nAutoML Training Complete!")
    print(f"Best model: {type(results['best_model']).__name__}")
    print(f"Best score: {results['best_score']:.6f}")
    print(f"Total time: {results['total_time_seconds']:.2f} seconds")
    print(f"Models evaluated: {results['models_evaluated']}")
    
    # Generate predictions
    print("\nGenerating forecasts...")
    test_features = test_data.drop('target', axis=1)
    predictions = automl.predict(test_features)
    
    # Evaluate predictions
    actual_values = test_data['target'].dropna()
    pred_length = min(len(predictions), len(actual_values))
    
    test_mae = mean_absolute_error(actual_values[:pred_length], predictions[:pred_length])
    test_rmse = np.sqrt(mean_squared_error(actual_values[:pred_length], predictions[:pred_length]))
    
    print(f"\nTest Performance:")
    print(f"MAE: {test_mae:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    
    # Get model explanation
    print("\nModel Explanation:")
    explanation = automl.explain_model()
    print(f"Model type: {explanation['model_type']}")
    print(f"Preprocessing steps: {len(explanation['preprocessing_steps'])}")
    print(f"Features created: {explanation['feature_engineering_rationale']['total_features'] if 'feature_engineering_rationale' in explanation else 'N/A'}")
    
    # Compare with simple baseline
    print("\nComparing with simple baseline...")
    
    # Naive forecast (last value)
    naive_predictions = [train_data['target'].dropna().iloc[-1]] * pred_length
    naive_mae = mean_absolute_error(actual_values[:pred_length], naive_predictions)
    naive_rmse = np.sqrt(mean_squared_error(actual_values[:pred_length], naive_predictions))
    
    print(f"Naive forecast MAE: {naive_mae:.6f}")
    print(f"Naive forecast RMSE: {naive_rmse:.6f}")
    
    improvement_mae = (naive_mae - test_mae) / naive_mae * 100
    improvement_rmse = (naive_rmse - test_rmse) / naive_rmse * 100
    
    print(f"\nAutoML Improvement over naive forecast:")
    print(f"MAE improvement: {improvement_mae:.2f}%")
    print(f"RMSE improvement: {improvement_rmse:.2f}%")
    
    return {
        'automl_results': results,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'baseline_mae': naive_mae,
        'baseline_rmse': naive_rmse,
        'improvement': {'mae': improvement_mae, 'rmse': improvement_rmse}
    }


if __name__ == "__main__":
    results = demonstrate_automl_time_series()
    print("\nAutoML democratizes time series forecasting by automating")
    print("the complex pipeline of preprocessing, feature engineering,")
    print("model selection, and hyperparameter optimization.")
