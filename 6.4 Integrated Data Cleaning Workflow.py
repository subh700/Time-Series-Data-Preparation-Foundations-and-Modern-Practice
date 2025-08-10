class TimeSeriesDataCleaner:
    """Comprehensive time series data cleaning pipeline."""
    
    def __init__(self, domain='general', validation_rules=None):
        self.domain = domain
        self.validation_rules = validation_rules or {}
        self.cleaning_history = []
        
        # Initialize domain-specific validators
        if domain == 'financial':
            self.domain_validator = FinancialDataValidator()
        elif domain == 'iot':
            self.domain_validator = IoTSensorValidator()
        else:
            self.domain_validator = None
    
    def clean_data(self, data, timestamp_col='timestamp', value_cols=None, 
                   cleaning_strategy='comprehensive'):
        """
        Execute comprehensive data cleaning pipeline.
        """
        if value_cols is None:
            value_cols = [col for col in data.columns if col != timestamp_col]
        
        cleaning_steps = []
        cleaned_data = data.copy()
        
        # Step 1: Timestamp validation and repair
        if cleaning_strategy in ['comprehensive', 'temporal']:
            timestamp_issues = validate_temporal_ordering(cleaned_data, timestamp_col)
            if any(timestamp_issues.values()):
                cleaned_data, repair_log = repair_timestamp_issues(
                    cleaned_data, timestamp_col, strategy='conservative'
                )
                cleaning_steps.extend(repair_log)
        
        # Step 2: Data type standardization
        if cleaning_strategy in ['comprehensive', 'format']:
            cleaned_data, type_log = standardize_data_types(cleaned_data)
            cleaning_steps.extend(type_log)
        
        # Step 3: Domain-specific validation
        if self.domain_validator and cleaning_strategy in ['comprehensive', 'domain']:
            for value_col in value_cols:
                violations = self.domain_validator.validate_data(cleaned_data, value_col)
                if any(violations.values()):
                    cleaned_data, domain_log = self.domain_validator.clean_violations(
                        cleaned_data, violations, strategy='conservative'
                    )
                    cleaning_steps.extend(domain_log)
        
        # Step 4: Statistical outlier detection and treatment
        if cleaning_strategy in ['comprehensive', 'outliers']:
            for value_col in value_cols:
                outlier_results = stl_outlier_detection(cleaned_data[value_col])
                if len(outlier_results['outlier_indices']) > 0:
                    treatment_results = outlier_interpolation_with_uncertainty(
                        cleaned_data[value_col], 
                        outlier_results['outlier_indices']
                    )
                    cleaned_data[value_col] = treatment_results['treated_data']
                    cleaning_steps.append(f"Treated {len(outlier_results['outlier_indices'])} outliers in {value_col}")
        
        # Step 5: Missing data imputation
        if cleaning_strategy in ['comprehensive', 'missing']:
            for value_col in value_cols:
                missing_count = cleaned_data[value_col].isnull().sum()
                if missing_count > 0:
                    # Choose imputation method based on missing pattern
                    if missing_count / len(cleaned_data) < 0.05:  # < 5% missing
                        cleaned_data[value_col] = cleaned_data[value_col].interpolate(method='time')
                        cleaning_steps.append(f"Interpolated {missing_count} missing values in {value_col}")
                    else:
                        # Use more sophisticated imputation for higher missing rates
                        imputed_values = seasonal_knn_imputation(cleaned_data[value_col])
                        cleaned_data[value_col] = imputed_values
                        cleaning_steps.append(f"Applied seasonal KNN imputation for {missing_count} missing values in {value_col}")
        
        # Step 6: Final validation
        final_quality_score = self._calculate_quality_score(cleaned_data, value_cols)
        
        # Record cleaning history
        self.cleaning_history.append({
            'timestamp': pd.Timestamp.now(),
            'strategy': cleaning_strategy,
            'steps_performed': cleaning_steps,
            'initial_shape': data.shape,
            'final_shape': cleaned_data.shape,
            'quality_score': final_quality_score
        })
        
        return {
            'cleaned_data': cleaned_data,
            'cleaning_log': cleaning_steps,
            'quality_score': final_quality_score,
            'data_reduction': 1 - (len(cleaned_data) / len(data))
        }
    
    def _calculate_quality_score(self, data, value_cols):
        """Calculate overall data quality score (0-1 scale)."""
        scores = []
        
        for col in value_cols:
            if col not in data.columns:
                continue
                
            series = data[col]
            
            # Completeness score
            completeness = 1 - (series.isnull().sum() / len(series))
            
            # Consistency score (based on outlier rate)
            outlier_results = stl_outlier_detection(series.dropna())
            consistency = 1 - (len(outlier_results['outlier_indices']) / len(series.dropna()))
            
            # Temporal regularity score
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                regularity = calculate_regularity_score(time_diffs)
            else:
                regularity = 1.0
            
            col_score = (completeness + consistency + regularity) / 3
            scores.append(col_score)
        
        return np.mean(scores) if scores else 0.0
