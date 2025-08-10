class ProductionTimeSeriesPipeline:
    """
    Production-ready wrapper for time series preprocessing pipelines
    with monitoring, logging, and error handling capabilities.
    """
    
    def __init__(self, pipeline: TimeSeriesPreprocessingPipeline, 
                 config: Dict[str, Any] = None):
        self.pipeline = pipeline
        self.config = config or {}
        
        # Production settings
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.performance_tracking = self.config.get('performance_tracking', True)
        self.error_recovery = self.config.get('error_recovery', True)
        self.data_validation = self.config.get('data_validation', True)
        
        # Monitoring components
        self.performance_metrics = []
        self.error_log = []
        self.data_quality_history = []
        
        # Setup logging
        self.logger = logging.getLogger(f"ProductionPipeline.{pipeline.name}")
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_latency_seconds': self.config.get('max_latency_seconds', 30),
            'min_throughput_records_per_second': self.config.get('min_throughput', 100),
            'max_memory_usage_mb': self.config.get('max_memory_mb', 1000),
            'min_data_quality_score': self.config.get('min_quality_score', 0.8)
        }
    
    def process_batch(self, data: pd.DataFrame, 
                     batch_id: str = None) -> Dict[str, Any]:
        """
        Process a batch of data with full production monitoring.
        """
        
        batch_id = batch_id or f"batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = pd.Timestamp.now()
        
        self.logger.info(f"Starting batch processing: {batch_id}")
        
        # Initialize result structure
        result = {
            'batch_id': batch_id,
            'status': 'processing',
            'start_time': start_time,
            'input_shape': data.shape,
            'transformed_data': None,
            'processing_time': None,
            'errors': [],
            'warnings': [],
            'quality_metrics': {},
            'performance_metrics': {}
        }
        
        try:
            # 1. Input validation
            if self.data_validation:
                validation_result = self._validate_input_data(data)
                
                if not validation_result['is_valid']:
                    result['status'] = 'failed'
                    result['errors'].extend(validation_result['errors'])
                    return result
                
                result['warnings'].extend(validation_result.get('warnings', []))
            
            # 2. Memory usage check
            memory_usage_mb = self._get_memory_usage_mb()
            
            if memory_usage_mb > self.performance_thresholds['max_memory_usage_mb']:
                warning_msg = f"High memory usage: {memory_usage_mb}MB"
                result['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
            
            # 3. Process data through pipeline
            transformed_data = self.pipeline.transform(data)
            
            # 4. Calculate processing metrics
            end_time = pd.Timestamp.now()
            processing_time = (end_time - start_time).total_seconds()
            throughput = len(data) / processing_time if processing_time > 0 else 0
            
            # 5. Performance validation
            performance_issues = self._validate_performance(processing_time, throughput)
            result['warnings'].extend(performance_issues)
            
            # 6. Data quality assessment
            if self.monitoring_enabled:
                quality_metrics = self._assess_output_quality(transformed_data)
                result['quality_metrics'] = quality_metrics
                
                if quality_metrics['overall_score'] < self.performance_thresholds['min_data_quality_score']:
                    warning_msg = f"Low data quality score: {quality_metrics['overall_score']:.3f}"
                    result['warnings'].append(warning_msg)
            
            # 7. Update result with success
            result.update({
                'status': 'completed',
                'end_time': end_time,
                'processing_time': processing_time,
                'transformed_data': transformed_data,
                'output_shape': transformed_data.shape if hasattr(transformed_data, 'shape') else None,
                'performance_metrics': {
                    'processing_time_seconds': processing_time,
                    'throughput_records_per_second': throughput,
                    'memory_usage_mb': memory_usage_mb
                }
            })
            
            self.logger.info(f"Batch {batch_id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            # Error handling
            error_msg = f"Pipeline processing failed: {str(e)}"
            result['status'] = 'failed'
            result['errors'].append(error_msg)
            result['end_time'] = pd.Timestamp.now()
            result['processing_time'] = (result['end_time'] - start_time).total_seconds()
            
            self.logger.error(f"Batch {batch_id} failed: {error_msg}")
            
            # Error recovery attempt if enabled
            if self.error_recovery:
                recovery_result = self._attempt_error_recovery(data, str(e))
                result['recovery_attempted'] = recovery_result
        
        # 8. Update monitoring history
        if self.performance_tracking:
            self._update_performance_history(result)
        
        return result
    
    def process_stream(self, data_stream, batch_size: int = 1000, 
                      max_batches: int = None) -> Generator[Dict[str, Any], None, None]:
        """
        Process streaming data in batches.
        """
        
        batch_count = 0
        current_batch = []
        
        for record in data_stream:
            current_batch.append(record)
            
            # Process when batch is full
            if len(current_batch) >= batch_size:
                batch_df = pd.DataFrame(current_batch)
                batch_result = self.process_batch(
                    batch_df, 
                    batch_id=f"stream_batch_{batch_count}"
                )
                
                yield batch_result
                
                # Reset for next batch
                current_batch = []
                batch_count += 1
                
                # Check max batches limit
                if max_batches and batch_count >= max_batches:
                    break
        
        # Process remaining records
        if current_batch:
            batch_df = pd.DataFrame(current_batch)
            batch_result = self.process_batch(
                batch_df, 
                batch_id=f"stream_batch_{batch_count}_final"
            )
            yield batch_result
    
    def _validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data meets pipeline requirements."""
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check basic requirements
        if data.empty:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Input data is empty")
            return validation_result
        
        # Check expected schema if available
        expected_schema = self.pipeline.metadata.get('input_schema')
        
        if expected_schema:
            expected_columns = expected_schema.get('columns', [])
            missing_columns = set(expected_columns) - set(data.columns)
            
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            expected_dtypes = expected_schema.get('dtypes', {})
            for col, expected_dtype in expected_dtypes.items():
                if col in data.columns and str(data[col].dtype) != expected_dtype:
                    validation_result['warnings'].append(
                        f"Column '{col}' has dtype {data[col].dtype}, expected {expected_dtype}"
                    )
        
        # Check for excessive missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.5:
            validation_result['warnings'].append(
                f"High missing value ratio: {missing_ratio:.1%}"
            )
        
        return validation_result
    
    def _validate_performance(self, processing_time: float, 
                            throughput: float) -> List[str]:
        """Validate processing performance against thresholds."""
        
        issues = []
        
        # Check latency
        if processing_time > self.performance_thresholds['max_latency_seconds']:
            issues.append(
                f"High latency: {processing_time:.2f}s > "
                f"{self.performance_thresholds['max_latency_seconds']}s"
            )
        
        # Check throughput
        min_throughput = self.performance_thresholds['min_throughput_records_per_second']
        if throughput < min_throughput:
            issues.append(
                f"Low throughput: {throughput:.1f} records/s < {min_throughput} records/s"
            )
        
        return issues
    
    def _assess_output_quality(self, output_data) -> Dict[str, Any]:
        """Assess quality of output data."""
        
        if isinstance(output_data, pd.DataFrame):
            # Standard DataFrame quality assessment
            completeness = 1 - (output_data.isnull().sum().sum() / 
                               (len(output_data) * len(output_data.columns)))
            
            # Check for infinite or extreme values
            numeric_cols = output_data.select_dtypes(include=[np.number]).columns
            validity_scores = []
            
            for col in numeric_cols:
                col_data = output_data[col]
                finite_ratio = np.isfinite(col_data).sum() / len(col_data)
                validity_scores.append(finite_ratio)
            
            validity = np.mean(validity_scores) if validity_scores else 1.0
            
            return {
                'overall_score': (completeness + validity) / 2,
                'completeness': completeness,
                'validity': validity,
                'output_shape': output_data.shape
            }
            
        elif isinstance(output_data, dict):
            # Handle structured output (e.g., from neural network preprocessor)
            return {
                'overall_score': 1.0,  # Assume good quality for structured output
                'output_type': 'structured',
                'output_keys': list(output_data.keys())
            }
        
        else:
            return {
                'overall_score': 0.5,  # Unknown format
                'output_type': str(type(output_data))
            }
    
    def _attempt_error_recovery(self, data: pd.DataFrame, error_msg: str) -> Dict[str, Any]:
        """Attempt to recover from processing errors."""
        
        recovery_result = {
            'attempted': True,
            'successful': False,
            'strategy': None,
            'message': None
        }
        
        try:
            # Strategy 1: Try with smaller data sample
            if len(data) > 1000:
                sample_data = data.sample(n=min(1000, len(data) // 2))
                sample_result = self.pipeline.transform(sample_data)
                
                recovery_result.update({
                    'successful': True,
                    'strategy': 'data_sampling',
                    'message': f'Processed sample of {len(sample_data)} records',
                    'sample_result': sample_result
                })
                
                return recovery_result
        
        except Exception as recovery_error:
            recovery_result['message'] = f'Recovery failed: {str(recovery_error)}'
        
        return recovery_result
    
    def _update_performance_history(self, batch_result: Dict[str, Any]) -> None:
        """Update performance monitoring history."""
        
        performance_record = {
            'timestamp': batch_result['start_time'],
            'batch_id': batch_result['batch_id'],
            'status': batch_result['status'],
            'processing_time': batch_result.get('processing_time'),
            'input_records': batch_result['input_shape'][0] if batch_result['input_shape'] else 0,
            'quality_score': batch_result.get('quality_metrics', {}).get('overall_score'),
            'error_count': len(batch_result.get('errors', [])),
            'warning_count': len(batch_result.get('warnings', []))
        }
        
        self.performance_metrics.append(performance_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: estimate based on pipeline size
            return 100.0  # Default assumption
    
    def get_performance_summary(self, last_n_batches: int = 100) -> Dict[str, Any]:
        """Get performance summary for recent batches."""
        
        if not self.performance_metrics:
            return {'status': 'no_data'}
        
        recent_metrics = self.performance_metrics[-last_n_batches:]
        successful_batches = [m for m in recent_metrics if m['status'] == 'completed']
        
        if not successful_batches:
            return {'status': 'no_successful_batches'}
        
        processing_times = [m['processing_time'] for m in successful_batches if m['processing_time']]
        quality_scores = [m['quality_score'] for m in successful_batches if m['quality_score']]
        
        return {
            'total_batches': len(recent_metrics),
            'successful_batches': len(successful_batches),
            'success_rate': len(successful_batches) / len(recent_metrics),
            'avg_processing_time': np.mean(processing_times) if processing_times else None,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else None,
            'error_rate': sum(m['error_count'] for m in recent_metrics) / len(recent_metrics),
            'warning_rate': sum(m['warning_count'] for m in recent_metrics) / len(recent_metrics)
        }
    
    def export_monitoring_data(self, filepath: str) -> None:
        """Export monitoring data for analysis."""
        
        monitoring_data = {
            'performance_metrics': self.performance_metrics,
            'error_log': self.error_log,
            'pipeline_metadata': self.pipeline.get_pipeline_summary(),
            'config': self.config,
            'export_timestamp': pd.Timestamp.now()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(monitoring_data, f, default=str, indent=2)
        
        self.logger.info(f"Monitoring data exported to {filepath}")
