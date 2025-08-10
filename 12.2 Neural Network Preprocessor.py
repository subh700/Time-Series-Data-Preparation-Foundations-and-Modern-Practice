class NeuralNetworkPreprocessor(TimeSeriesPipelineComponent):
    """
    Specialized preprocessing for neural network time series models.
    """
    
    def __init__(self, name: str = "nn_preprocessor", config: Dict[str, Any] = None):
        default_config = {
            'target_columns': ['value'],
            'feature_columns': [],
            'sequence_length': 60,
            'prediction_horizon': 1,
            'scaling_method': 'minmax',  # 'minmax', 'standard', 'robust'
            'handle_missing': 'interpolate',
            'outlier_treatment': 'clip',
            'create_cyclical_features': True,
            'add_trend_features': True,
            'normalize_by_group': None,  # Column to group by for normalization
            'validation_split': 0.2
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        # Fitted components
        self.scalers = {}
        self.feature_names = []
        self.cyclical_encoders = {}
        self.trend_coefficients = {}
        self.data_stats = {}
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'NeuralNetworkPreprocessor':
        """Fit neural network preprocessor."""
        
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        self.validate_data(data)
        
        # 1. Handle missing values
        clean_data = self._handle_missing_values(data)
        
        # 2. Create additional features
        enriched_data = self._create_features(clean_data)
        
        # 3. Handle outliers
        outlier_treated_data = self._handle_outliers(enriched_data)
        
        # 4. Set up scalers for each numeric column
        numeric_columns = outlier_treated_data.select_dtypes(include=[np.number]).columns
        
        scaling_method = self.config['scaling_method']
        
        for col in numeric_columns:
            if scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
            
            # Fit scaler
            scaler.fit(outlier_treated_data[[col]])
            self.scalers[col] = scaler
        
        # 5. Calculate data statistics
        self.data_stats = {
            'shape': outlier_treated_data.shape,
            'numeric_columns': list(numeric_columns),
            'categorical_columns': list(outlier_treated_data.select_dtypes(include=['object']).columns),
            'datetime_columns': list(outlier_treated_data.select_dtypes(include=['datetime64']).columns),
            'missing_value_counts': outlier_treated_data.isnull().sum().to_dict()
        }
        
        # 6. Store feature names
        self.feature_names = list(outlier_treated_data.columns)
        
        self.metadata.update({
            'sequence_length': self.config['sequence_length'],
            'prediction_horizon': self.config['prediction_horizon'],
            'scaling_method': scaling_method,
            'feature_count': len(self.feature_names),
            'data_stats': self.data_stats
        })
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Transform data for neural network input."""
        
        if not self.is_fitted:
            raise RuntimeError("NeuralNetworkPreprocessor must be fitted before transformation")
        
        # Apply same preprocessing steps
        clean_data = self._handle_missing_values(data)
        enriched_data = self._create_features(clean_data)
        outlier_treated_data = self._handle_outliers(enriched_data)
        
        # Scale numeric features
        scaled_data = outlier_treated_data.copy()
        
        for col, scaler in self.scalers.items():
            if col in scaled_data.columns:
                scaled_data[col] = scaler.transform(scaled_data[[col]]).flatten()
        
        # Create sequences for neural network
        sequences = self._create_sequences(scaled_data)
        
        return sequences
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        
        strategy = self.config['handle_missing']
        result = data.copy()
        
        if strategy == 'interpolate':
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].interpolate(method='linear')
            
            # Forward fill any remaining missing values
            result = result.fillna(method='ffill')
            
        elif strategy == 'drop':
            result = result.dropna()
            
        elif strategy == 'zero':
            result = result.fillna(0)
            
        elif strategy == 'median':
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result[numeric_cols] = result[numeric_cols].fillna(
                result[numeric_cols].median()
            )
        
        return result
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for neural network."""
        
        result = data.copy()
        
        # Create cyclical time features if enabled
        if self.config['create_cyclical_features']:
            result = self._add_cyclical_features(result)
        
        # Add trend features if enabled
        if self.config['add_trend_features']:
            result = self._add_trend_features(result)
        
        return result
    
    def _add_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for time-based features."""
        
        result = data.copy()
        
        # If index is datetime, create cyclical features
        if isinstance(result.index, pd.DatetimeIndex):
            dt_index = result.index
            
            # Hour of day (if hourly data)
            if dt_index.freq and 'H' in str(dt_index.freq):
                result['hour_sin'] = np.sin(2 * np.pi * dt_index.hour / 24)
                result['hour_cos'] = np.cos(2 * np.pi * dt_index.hour / 24)
            
            # Day of week
            result['dow_sin'] = np.sin(2 * np.pi * dt_index.dayofweek / 7)
            result['dow_cos'] = np.cos(2 * np.pi * dt_index.dayofweek / 7)
            
            # Day of year
            result['doy_sin'] = np.sin(2 * np.pi * dt_index.dayofyear / 365)
            result['doy_cos'] = np.cos(2 * np.pi * dt_index.dayofyear / 365)
            
            # Month
            result['month_sin'] = np.sin(2 * np.pi * dt_index.month / 12)
            result['month_cos'] = np.cos(2 * np.pi * dt_index.month / 12)
        
        return result
    
    def _add_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based features."""
        
        result = data.copy()
        
        # Linear time trend
        result['time_trend'] = np.arange(len(result))
        
        # For each target column, add rolling statistics
        target_cols = self.config['target_columns']
        
        for col in target_cols:
            if col in result.columns:
                # Rolling means with different windows
                for window in [5, 10, 20]:
                    if len(result) > window:
                        result[f'{col}_ma_{window}'] = (
                            result[col].rolling(window=window, min_periods=1).mean()
                        )
                
                # Exponential weighted moving average
                result[f'{col}_ewm'] = result[col].ewm(span=10).mean()
                
                # Rate of change
                result[f'{col}_pct_change'] = result[col].pct_change().fillna(0)
                
                # Lag features
                for lag in [1, 2, 3, 7]:
                    if len(result) > lag:
                        result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        
        treatment = self.config['outlier_treatment']
        
        if treatment == 'none':
            return data
        
        result = data.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if treatment == 'clip':
                # Clip to 1st and 99th percentiles
                lower_bound = result[col].quantile(0.01)
                upper_bound = result[col].quantile(0.99)
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif treatment == 'remove':
                # Remove rows with outliers (using IQR method)
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (result[col] >= lower_bound) & (result[col] <= upper_bound)
                result = result[outlier_mask]
        
        return result
    
    def _create_sequences(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create sequences for neural network training."""
        
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config['prediction_horizon']
        target_columns = self.config['target_columns']
        
        # Ensure we have enough data
        min_required_length = sequence_length + prediction_horizon
        if len(data) < min_required_length:
            raise ValueError(f"Data length ({len(data)}) insufficient for sequence_length "
                           f"({sequence_length}) + prediction_horizon ({prediction_horizon})")
        
        # Prepare feature matrix
        feature_columns = [col for col in data.columns if col not in target_columns]
        
        X_sequences = []
        y_sequences = []
        
        # Create sequences
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence (features + targets)
            X_seq = data.iloc[i:i + sequence_length][feature_columns + target_columns].values
            X_sequences.append(X_seq)
            
            # Target sequence (only target columns)
            y_seq = data.iloc[i + sequence_length:i + sequence_length + prediction_horizon][target_columns].values
            y_sequences.append(y_seq)
        
        return {
            'X': np.array(X_sequences),
            'y': np.array(y_sequences),
            'feature_names': feature_columns + target_columns,
            'target_names': target_columns,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon
        }
    
    def _get_state(self) -> Dict[str, Any]:
        """Get component state for serialization."""
        return {
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'cyclical_encoders': self.cyclical_encoders,
            'trend_coefficients': self.trend_coefficients,
            'data_stats': self.data_stats
        }
    
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set component state from deserialization."""
        self.scalers = state['scalers']
        self.feature_names = state['feature_names']
        self.cyclical_encoders = state['cyclical_encoders']
        self.trend_coefficients = state['trend_coefficients']
        self.data_stats = state['data_stats']
