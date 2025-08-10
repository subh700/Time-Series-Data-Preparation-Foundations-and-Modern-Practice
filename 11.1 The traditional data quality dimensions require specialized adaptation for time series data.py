class TimeSeriesDataQualityFramework:
    """
    Comprehensive data quality assessment framework for time series data.
    """
    
    def __init__(self, domain='general', quality_weights=None):
        self.domain = domain
        self.quality_weights = quality_weights or self._default_weights()
        self.quality_dimensions = self._initialize_dimensions()
        
    def _default_weights(self):
        """Default weights for quality dimensions by domain."""
        return {
            'general': {
                'completeness': 0.15,
                'timeliness': 0.20,
                'consistency': 0.15,
                'accuracy': 0.15,
                'validity': 0.10,
                'uniqueness': 0.10,
                'temporal_coherence': 0.15
            },
            'financial': {
                'completeness': 0.10,
                'timeliness': 0.25,  # Critical for trading
                'consistency': 0.15,
                'accuracy': 0.20,    # Critical for monetary values
                'validity': 0.10,
                'uniqueness': 0.05,
                'temporal_coherence': 0.15
            },
            'iot': {
                'completeness': 0.20,  # Sensor data often incomplete
                'timeliness': 0.15,
                'consistency': 0.20,   # Multiple sensors must agree
                'accuracy': 0.10,
                'validity': 0.15,      # Sensor range validation
                'uniqueness': 0.05,
                'temporal_coherence': 0.15
            }
        }
    
    def _initialize_dimensions(self):
        """Initialize quality dimension calculators."""
        return {
            'completeness': CompletenessAssessor(),
            'timeliness': TimelinessAssessor(),
            'consistency': ConsistencyAssessor(),
            'accuracy': AccuracyAssessor(),
            'validity': ValidityAssessor(),
            'uniqueness': UniquenessAssessor(),
            'temporal_coherence': TemporalCoherenceAssessor()
        }
    
    def assess_quality(self, data, timestamp_col='timestamp', value_cols=None, 
                      reference_data=None, schema=None):
        """
        Comprehensive data quality assessment.
        
        Args:
            data: Time series data
            timestamp_col: Timestamp column name
            value_cols: Value columns to assess
            reference_data: Reference data for accuracy assessment
            schema: Expected data schema for validity assessment
        """
        
        if value_cols is None:
            value_cols = [col for col in data.columns if col != timestamp_col]
        
        quality_scores = {}
        detailed_results = {}
        
        # Assess each quality dimension
        for dimension, assessor in self.quality_dimensions.items():
            try:
                if dimension == 'accuracy' and reference_data is not None:
                    score, details = assessor.assess(
                        data, timestamp_col, value_cols, reference_data=reference_data
                    )
                elif dimension == 'validity' and schema is not None:
                    score, details = assessor.assess(
                        data, timestamp_col, value_cols, schema=schema
                    )
                else:
                    score, details = assessor.assess(data, timestamp_col, value_cols)
                
                quality_scores[dimension] = score
                detailed_results[dimension] = details
                
            except Exception as e:
                print(f"Warning: Failed to assess {dimension}: {str(e)}")
                quality_scores[dimension] = 0.0
                detailed_results[dimension] = {'error': str(e)}
        
        # Calculate weighted overall quality score
        weights = self.quality_weights.get(self.domain, self.quality_weights['general'])
        overall_score = sum(
            quality_scores.get(dim, 0) * weight 
            for dim, weight in weights.items()
            if dim in quality_scores
        )
        
        return {
            'overall_score': overall_score,
            'dimension_scores': quality_scores,
            'detailed_results': detailed_results,
            'assessment_timestamp': pd.Timestamp.now(),
            'domain': self.domain
        }
