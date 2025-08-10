class AutomatedPipelineBuilder:
    """
    Automated pipeline builder that selects and configures
    preprocessing components based on data characteristics.
    """
    
    def __init__(self, model_type: str = 'auto'):
        self.model_type = model_type
        self.data_profile = {}
        self.recommended_components = []
        self.pipeline_config = {}
        
    def analyze_data(self, data: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Analyze data characteristics to inform pipeline decisions."""
        
        analysis = {
            'basic_stats': self._get_basic_statistics(data),
            'temporal_characteristics': self._analyze_temporal_patterns(data),
            'data_quality': self._assess_data_quality(data),
            'seasonality': self._detect_seasonality_patterns(data, target_column),
            'stationarity': self._test_stationarity(data, target_column),
            'distribution': self._analyze_distributions(data)
        }
        
        self.data_profile = analysis
        return analysis
    
    def recommend_pipeline(self, data: pd.DataFrame, target_column: str,
                          model_type: str = None) -> TimeSeriesPreprocessingPipeline:
        """Recommend optimal preprocessing pipeline based on data analysis."""
        
        if model_type:
            self.model_type = model_type
        
        # Analyze data if not done already
        if not self.data_profile:
            self.analyze_data(data, target_column)
        
        # Auto-detect model type if needed
        if self.model_type == 'auto':
            self.model_type = self._recommend_model_type()
        
        # Build pipeline based on analysis and model type
        pipeline = TimeSeriesPreprocessingPipeline(
            name=f"auto_pipeline_{self.model_type}",
            config={'model_type': self.model_type, 'target_column': target_column}
        )
        
        # Add components based on recommendations
        if self._needs_data_cleaning():
            pipeline.add_component(
                DataCleaningComponent(
                    config=self._get_cleaning_config()
                )
            )
        
        if self._needs_outlier_treatment():
            pipeline.add_component(
                OutlierTreatmentComponent(
                    config=self._get_outlier_config()
                ),
                dependencies=['data_cleaning'] if self._needs_data_cleaning() else []
            )
        
        if self._needs_missing_value_imputation():
            pipeline.add_component(
                MissingValueImputationComponent(
                    config=self._get_imputation_config()
                ),
                dependencies=self._get_previous_component_names(pipeline)
            )
        
        # Model-specific components
        if self.model_type in ['arima', 'sarima']:
            pipeline.add_component(
                ARIMAPreprocessor(
                    config=self._get_arima_config(target_column)
                ),
                dependencies=self._get_previous_component_names(pipeline)
            )
        
        elif self.model_type in ['lstm', 'gru', 'neural_network']:
            pipeline.add_component(
                NeuralNetworkPreprocessor(
                    config=self._get_nn_config(target_column)
                ),
                dependencies=self._get_previous_component_names(pipeline)
            )
        
        elif self.model_type in ['random_forest', 'xgboost', 'lightgbm']:
            pipeline.add_component(
                MachineLearningPreprocessor(
                    config=self._get_ml_config(target_column)
                ),
                dependencies=self._get_previous_component_names(pipeline)
            )
        
        # Feature engineering components
        if self._needs_feature_engineering():
            pipeline.add_component(
                FeatureEngineeringComponent(
                    config=self._get_feature_engineering_config()
                ),
                dependencies=self._get_previous_component_names(pipeline)
            )
        
        return pipeline
    
    def _get_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistical summary of the data."""
        
        return {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': data.describe().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'duplicate_rows': data.duplicated().sum()
        }
    
    def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal characteristics of the data."""
        
        temporal_analysis = {}
        
        # Check if there's a datetime column or index
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if isinstance(data.index, pd.DatetimeIndex):
            dt_index = data.index
            temporal_analysis['has_datetime_index'] = True
        elif datetime_cols:
            dt_index = pd.to_datetime(data[datetime_cols[0]])
            temporal_analysis['has_datetime_index'] = False
            temporal_analysis['datetime_column'] = datetime_cols[0]
        else:
            return {'has_temporal_info': False}
        
        # Analyze frequency
        temporal_analysis.update({
            'has_temporal_info': True,
            'frequency': pd.infer_freq(dt_index),
            'time_span': dt_index.max() - dt_index.min(),
            'data_points': len(dt_index),
            'regular_intervals': self._check_regular_intervals(dt_index),
            'gaps': self._detect_time_gaps(dt_index)
        })
        
        return temporal_analysis
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        quality_metrics = {
            'completeness': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'uniqueness': data.nunique().sum() / len(data),
            'consistency': self._check_data_consistency(data),
            'outlier_ratio': self._estimate_outlier_ratio(numeric_data),
            'data_types_consistent': self._check_dtype_consistency(data)
        }
        
        return quality_metrics
    
    def _detect_seasonality_patterns(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect seasonal patterns in the target variable."""
        
        if not target_column or target_column not in data.columns:
            return {'seasonality_detected': False}
        
        series = data[target_column].dropna()
        
        if len(series) < 50:  # Need sufficient data for seasonality detection
            return {'seasonality_detected': False, 'reason': 'insufficient_data'}
        
        seasonality_results = {}
        
        # Test common seasonal periods
        seasonal_periods = [7, 12, 24, 30, 52, 365] if len(series) > 365 else [7, 12, 24, 30]
        
        for period in seasonal_periods:
            if len(series) >= 2 * period:
                autocorr = series.autocorr(lag=period)
                seasonality_results[f'period_{period}'] = {
                    'autocorrelation': autocorr,
                    'significant': autocorr > 0.3
                }
        
        # Determine if any seasonality detected
        significant_periods = [
            period for period, result in seasonality_results.items()
            if result.get('significant', False)
        ]
        
        return {
            'seasonality_detected': len(significant_periods) > 0,
            'seasonal_periods': significant_periods,
            'detailed_results': seasonality_results
        }
    
    def _test_stationarity(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Test stationarity of the target variable."""
        
        if not target_column or target_column not in data.columns:
            return {'stationarity_tested': False}
        
        try:
            from statsmodels.tsa.stattools import adfuller
            
            series = data[target_column].dropna()
            
            if len(series) < 20:
                return {'stationarity_tested': False, 'reason': 'insufficient_data'}
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series)
            
            return {
                'stationarity_tested': True,
                'is_stationary': adf_result[1] < 0.05,
                'adf_statistic': adf_result[0],
                'adf_p_value': adf_result[1],
                'critical_values': adf_result[4]
            }
            
        except ImportError:
            return {'stationarity_tested': False, 'reason': 'statsmodels_not_available'}
    
    def _analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distribution characteristics of numeric columns."""
        
        numeric_data = data.select_dtypes(include=[np.number])
        distribution_analysis = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) > 0:
                distribution_analysis[col] = {
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'normality_likely': abs(series.skew()) < 1 and abs(series.kurtosis()) < 3,
                    'zero_values': (series == 0).sum(),
                    'negative_values': (series < 0).sum(),
                    'range': series.max() - series.min()
                }
        
        return distribution_analysis
    
    def _recommend_model_type(self) -> str:
        """Recommend model type based on data characteristics."""
        
        data_size = self.data_profile['basic_stats']['shape'][0]
        has_seasonality = self.data_profile.get('seasonality', {}).get('seasonality_detected', False)
        is_stationary = self.data_profile.get('stationarity', {}).get('is_stationary', True)
        
        # Decision logic for model recommendation
        if data_size < 100:
            return 'simple_exponential_smoothing'
        elif data_size < 500:
            if has_seasonality:
                return 'sarima'
            else:
                return 'arima'
        elif data_size < 2000:
            return 'random_forest'
        else:
            return 'lstm'
    
    def _needs_data_cleaning(self) -> bool:
        """Determine if data cleaning component is needed."""
        
        quality = self.data_profile.get('data_quality', {})
        completeness = quality.get('completeness', 1.0)
        
        return completeness < 0.95 or quality.get('outlier_ratio', 0) > 0.05
    
    def _needs_outlier_treatment(self) -> bool:
        """Determine if outlier treatment is needed."""
        
        quality = self.data_profile.get('data_quality', {})
        return quality.get('outlier_ratio', 0) > 0.03
    
    def _needs_missing_value_imputation(self) -> bool:
        """Determine if missing value imputation is needed."""
        
        basic_stats = self.data_profile.get('basic_stats', {})
        missing_values = basic_stats.get('missing_values', {})
        
        return any(count > 0 for count in missing_values.values())
    
    def _needs_feature_engineering(self) -> bool:
        """Determine if feature engineering is needed."""
        
        # Feature engineering is generally beneficial for ML models
        return self.model_type in ['random_forest', 'xgboost', 'lightgbm', 'lstm', 'gru']
    
    def _get_previous_component_names(self, pipeline: TimeSeriesPreprocessingPipeline) -> List[str]:
        """Get names of existing components in pipeline."""
        return [c.name for c in pipeline.components]
    
    # Configuration methods for different components
    def _get_cleaning_config(self) -> Dict[str, Any]:
        """Get configuration for data cleaning component."""
        return {
            'remove_duplicates': True,
            'handle_missing_timestamps': True,
            'validate_data_types': True
        }
    
    def _get_outlier_config(self) -> Dict[str, Any]:
        """Get configuration for outlier treatment."""
        return {
            'method': 'iqr',
            'treatment': 'clip',
            'threshold': 1.5
        }
    
    def _get_imputation_config(self) -> Dict[str, Any]:
        """Get configuration for missing value imputation."""
        return {
            'strategy': 'interpolate',
            'method': 'linear'
        }
    
    def _get_arima_config(self, target_column: str) -> Dict[str, Any]:
        """Get configuration for ARIMA preprocessor."""
        
        seasonality = self.data_profile.get('seasonality', {})
        seasonal_periods = []
        
        if seasonality.get('seasonality_detected'):
            seasonal_periods = [
                int(period.split('_')[1]) for period in seasonality.get('seasonal_periods', [])
            ]
        
        return {
            'target_column': target_column,
            'seasonal_periods': seasonal_periods,
            'log_transform': 'auto',
            'max_differencing_order': 2
        }
    
    def _get_nn_config(self, target_column: str) -> Dict[str, Any]:
        """Get configuration for neural network preprocessor."""
        
        data_size = self.data_profile['basic_stats']['shape'][0]
        
        # Adjust sequence length based on data size
        if data_size < 500:
            sequence_length = min(30, data_size // 10)
        elif data_size < 2000:
            sequence_length = 60
        else:
            sequence_length = 120
        
        return {
            'target_columns': [target_column],
            'sequence_length': sequence_length,
            'prediction_horizon': 1,
            'scaling_method': 'minmax',
            'create_cyclical_features': True,
            'add_trend_features': True
        }
    
    def _get_ml_config(self, target_column: str) -> Dict[str, Any]:
        """Get configuration for machine learning preprocessor."""
        return {
            'target_column': target_column,
            'create_lag_features': True,
            'max_lags': 10,
            'create_rolling_features': True,
            'rolling_windows': [3, 7, 14],
            'scaling_method': 'standard'
        }
    
    def _get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get configuration for feature engineering."""
        return {
            'create_temporal_features': True,
            'create_interaction_features': False,
            'polynomial_features': False,
            'feature_selection': True,
            'selection_method': 'mutual_info'
        }
    
    # Helper methods for data analysis
    def _check_regular_intervals(self, dt_index: pd.DatetimeIndex) -> bool:
        """Check if time series has regular intervals."""
        
        if len(dt_index) < 3:
            return True
        
        intervals = dt_index.to_series().diff().dropna()
        mode_interval = intervals.mode()
        
        if len(mode_interval) == 0:
            return False
        
        # Check if most intervals match the mode
        regular_ratio = (intervals == mode_interval.iloc[0]).sum() / len(intervals)
        return regular_ratio > 0.8
    
    def _detect_time_gaps(self, dt_index: pd.DatetimeIndex) -> Dict[str, Any]:
        """Detect gaps in time series."""
        
        if len(dt_index) < 2:
            return {'gaps_detected': False}
        
        intervals = dt_index.to_series().diff().dropna()
        
        if len(intervals) == 0:
            return {'gaps_detected': False}
        
        # Estimate expected interval
        expected_interval = intervals.mode().iloc[0] if len(intervals) > 0 else intervals.median()
        
        # Detect large gaps (more than 2x expected interval)
        large_gaps = intervals[intervals > 2 * expected_interval]
        
        return {
            'gaps_detected': len(large_gaps) > 0,
            'gap_count': len(large_gaps),
            'largest_gap': large_gaps.max() if len(large_gaps) > 0 else pd.Timedelta(0),
            'expected_interval': expected_interval
        }
    
    def _check_data_consistency(self, data: pd.DataFrame) -> float:
        """Check consistency across data types and formats."""
        
        consistency_scores = []
        
        # Check numeric columns for consistent scale
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) > 0:
                # Check for consistent scale (coefficient of variation)
                cv = series.std() / abs(series.mean()) if series.mean() != 0 else 0
                # Lower CV indicates more consistent scale
                consistency_scores.append(max(0, 1 - min(cv, 1)))
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _estimate_outlier_ratio(self, numeric_data: pd.DataFrame) -> float:
        """Estimate ratio of outliers in numeric data."""
        
        if numeric_data.empty:
            return 0.0
        
        outlier_ratios = []
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            if len(series) > 4:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((series < lower_bound) | (series > upper_bound)).sum()
                outlier_ratio = outliers / len(series)
                outlier_ratios.append(outlier_ratio)
        
        return np.mean(outlier_ratios) if outlier_ratios else 0.0
    
    def _check_dtype_consistency(self, data: pd.DataFrame) -> bool:
        """Check if data types are consistent and appropriate."""
        
        # Check if numeric columns contain string values
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Check if conversion to numeric would change the data
                try:
                    converted = pd.to_numeric(data[col], errors='coerce')
                    if converted.isna().sum() > data[col].isna().sum():
                        return False
                except:
                    return False
        
        return True
