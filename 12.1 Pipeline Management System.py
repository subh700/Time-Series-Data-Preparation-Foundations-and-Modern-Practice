class TimeSeriesPreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline for time series data with
    automated orchestration and monitoring capabilities.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.components: List[TimeSeriesPipelineComponent] = []
        self.component_graph = {}
        self.execution_history = []
        self.is_fitted = False
        self.logger = logging.getLogger(f"Pipeline.{name}")
        
        # Pipeline metadata
        self.metadata = {
            'creation_time': pd.Timestamp.now(),
            'version': '1.0.0',
            'input_schema': None,
            'output_schema': None,
            'performance_metrics': {}
        }
    
    def add_component(self, component: TimeSeriesPipelineComponent, 
                     dependencies: List[str] = None) -> 'TimeSeriesPreprocessingPipeline':
        """Add component to pipeline with optional dependencies."""
        
        if any(c.name == component.name for c in self.components):
            raise ValueError(f"Component '{component.name}' already exists in pipeline")
        
        # Validate dependencies exist
        dependencies = dependencies or []
        existing_names = [c.name for c in self.components]
        
        for dep in dependencies:
            if dep not in existing_names:
                raise ValueError(f"Dependency '{dep}' not found in pipeline")
        
        self.components.append(component)
        self.component_graph[component.name] = dependencies
        
        self.logger.info(f"Added component '{component.name}' with dependencies: {dependencies}")
        
        return self
    
    def remove_component(self, name: str) -> 'TimeSeriesPreprocessingPipeline':
        """Remove component from pipeline."""
        
        # Check if component is a dependency for others
        dependents = [comp_name for comp_name, deps in self.component_graph.items() 
                     if name in deps]
        
        if dependents:
            raise ValueError(f"Cannot remove '{name}': required by {dependents}")
        
        # Remove component
        self.components = [c for c in self.components if c.name != name]
        del self.component_graph[name]
        
        self.logger.info(f"Removed component '{name}'")
        return self
    
    def _get_execution_order(self) -> List[str]:
        """Determine optimal execution order using topological sort."""
        
        # Topological sort using Kahn's algorithm
        in_degree = {name: 0 for name in self.component_graph}
        
        for deps in self.component_graph.values():
            for dep in deps:
                in_degree[dep] += 1
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees
            for name, deps in self.component_graph.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
        
        if len(execution_order) != len(self.components):
            raise ValueError("Circular dependency detected in pipeline")
        
        return execution_order
    
    def fit(self, data: pd.DataFrame, validation_data: pd.DataFrame = None, 
            **kwargs) -> 'TimeSeriesPreprocessingPipeline':
        """Fit entire pipeline to training data."""
        
        start_time = pd.Timestamp.now()
        self.logger.info(f"Starting pipeline fitting with {len(data)} samples")
        
        # Store input schema
        self.metadata['input_schema'] = {
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'shape': data.shape,
            'index_type': str(type(data.index))
        }
        
        execution_order = self._get_execution_order()
        current_data = data.copy()
        
        # Fit components in order
        for component_name in execution_order:
            component = next(c for c in self.components if c.name == component_name)
            
            step_start = pd.Timestamp.now()
            self.logger.info(f"Fitting component: {component_name}")
            
            try:
                # Fit component
                component.fit(current_data, **kwargs)
                
                # Transform data for next component
                current_data = component.transform(current_data, **kwargs)
                
                step_duration = pd.Timestamp.now() - step_start
                
                self.execution_history.append({
                    'component': component_name,
                    'operation': 'fit',
                    'start_time': step_start,
                    'duration': step_duration,
                    'input_shape': current_data.shape,
                    'success': True
                })
                
                self.logger.info(f"Component '{component_name}' fitted successfully in {step_duration}")
                
            except Exception as e:
                self.logger.error(f"Error fitting component '{component_name}': {str(e)}")
                
                self.execution_history.append({
                    'component': component_name,
                    'operation': 'fit',
                    'start_time': step_start,
                    'duration': pd.Timestamp.now() - step_start,
                    'error': str(e),
                    'success': False
                })
                
                raise RuntimeError(f"Pipeline fitting failed at component '{component_name}': {str(e)}")
        
        # Store output schema
        self.metadata['output_schema'] = {
            'columns': list(current_data.columns),
            'dtypes': current_data.dtypes.to_dict(),
            'shape': current_data.shape
        }
        
        # Validate with validation data if provided
        if validation_data is not None:
            validation_result = self._validate_pipeline(validation_data)
            self.metadata['validation_metrics'] = validation_result
        
        total_duration = pd.Timestamp.now() - start_time
        self.is_fitted = True
        
        self.logger.info(f"Pipeline fitting completed in {total_duration}")
        
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transformation")
        
        start_time = pd.Timestamp.now()
        execution_order = self._get_execution_order()
        current_data = data.copy()
        
        # Transform through each component
        for component_name in execution_order:
            component = next(c for c in self.components if c.name == component_name)
            
            step_start = pd.Timestamp.now()
            
            try:
                current_data = component.transform(current_data, **kwargs)
                
                step_duration = pd.Timestamp.now() - step_start
                
                self.execution_history.append({
                    'component': component_name,
                    'operation': 'transform',
                    'start_time': step_start,
                    'duration': step_duration,
                    'input_shape': data.shape,
                    'output_shape': current_data.shape,
                    'success': True
                })
                
            except Exception as e:
                self.logger.error(f"Error transforming with '{component_name}': {str(e)}")
                
                self.execution_history.append({
                    'component': component_name,
                    'operation': 'transform',
                    'start_time': step_start,
                    'duration': pd.Timestamp.now() - step_start,
                    'error': str(e),
                    'success': False
                })
                
                raise RuntimeError(f"Pipeline transformation failed at '{component_name}': {str(e)}")
        
        total_duration = pd.Timestamp.now() - start_time
        self.logger.info(f"Pipeline transformation completed in {total_duration}")
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, validation_data: pd.DataFrame = None, 
                     **kwargs) -> pd.DataFrame:
        """Fit pipeline and transform data in one step."""
        return self.fit(data, validation_data, **kwargs).transform(data, **kwargs)
    
    def _validate_pipeline(self, validation_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate pipeline performance on validation data."""
        
        start_time = pd.Timestamp.now()
        
        try:
            transformed_data = self.transform(validation_data)
            
            validation_metrics = {
                'validation_success': True,
                'validation_duration': pd.Timestamp.now() - start_time,
                'input_shape': validation_data.shape,
                'output_shape': transformed_data.shape,
                'data_quality_score': self._calculate_data_quality_score(transformed_data)
            }
            
        except Exception as e:
            validation_metrics = {
                'validation_success': False,
                'validation_error': str(e),
                'validation_duration': pd.Timestamp.now() - start_time
            }
        
        return validation_metrics
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        scores = []
        
        # Completeness score
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        scores.append(completeness)
        
        # Numeric data range score (check for reasonable values)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            range_scores = []
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Check for infinite or extremely large values
                    valid_ratio = np.isfinite(col_data).sum() / len(col_data)
                    range_scores.append(valid_ratio)
            
            if range_scores:
                scores.append(np.mean(range_scores))
        
        # Uniqueness score (avoid too many duplicates)
        uniqueness_scores = []
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            uniqueness_scores.append(min(1.0, unique_ratio * 2))  # Cap at 1.0
        
        if uniqueness_scores:
            scores.append(np.mean(uniqueness_scores))
        
        return np.mean(scores) if scores else 0.0
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        
        execution_stats = {}
        if self.execution_history:
            successful_operations = [h for h in self.execution_history if h['success']]
            
            if successful_operations:
                durations = [h['duration'].total_seconds() for h in successful_operations]
                execution_stats = {
                    'total_operations': len(self.execution_history),
                    'successful_operations': len(successful_operations),
                    'success_rate': len(successful_operations) / len(self.execution_history),
                    'avg_duration_seconds': np.mean(durations),
                    'total_duration_seconds': sum(durations)
                }
        
        return {
            'name': self.name,
            'component_count': len(self.components),
            'components': [c.name for c in self.components],
            'is_fitted': self.is_fitted,
            'metadata': self.metadata,
            'execution_stats': execution_stats,
            'component_graph': self.component_graph
        }
    
    def save_pipeline(self, path: Path) -> None:
        """Save entire pipeline to disk."""
        
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before saving")
        
        pipeline_dir = Path(path)
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each component
        components_dir = pipeline_dir / 'components'
        components_dir.mkdir(exist_ok=True)
        
        for component in self.components:
            component_path = components_dir / f"{component.name}.pkl"
            component.save(component_path)
        
        # Save pipeline metadata
        pipeline_metadata = {
            'name': self.name,
            'config': self.config,
            'component_graph': self.component_graph,
            'execution_history': self.execution_history,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted,
            'component_names': [c.name for c in self.components],
            'component_types': [type(c).__name__ for c in self.components]
        }
        
        metadata_path = pipeline_dir / 'pipeline_metadata.pkl'
        joblib.dump(pipeline_metadata, metadata_path)
        
        self.logger.info(f"Pipeline saved to {pipeline_dir}")
    
    @classmethod
    def load_pipeline(cls, path: Path) -> 'TimeSeriesPreprocessingPipeline':
        """Load pipeline from disk."""
        
        pipeline_dir = Path(path)
        metadata_path = pipeline_dir / 'pipeline_metadata.pkl'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Pipeline metadata not found at {metadata_path}")
        
        # Load pipeline metadata
        pipeline_metadata = joblib.load(metadata_path)
        
        # Create pipeline instance
        pipeline = cls(
            name=pipeline_metadata['name'],
            config=pipeline_metadata['config']
        )
        
        # Restore pipeline state
        pipeline.component_graph = pipeline_metadata['component_graph']
        pipeline.execution_history = pipeline_metadata['execution_history']
        pipeline.metadata = pipeline_metadata['metadata']
        pipeline.is_fitted = pipeline_metadata['is_fitted']
        
        # Load components
        components_dir = pipeline_dir / 'components'
        component_names = pipeline_metadata['component_names']
        component_types = pipeline_metadata['component_types']
        
        # Import component classes dynamically
        import importlib
        
        for name, type_name in zip(component_names, component_types):
            component_path = components_dir / f"{name}.pkl"
            
            # This assumes components are available in current namespace
            # In practice, you'd need proper module importing
            component = TimeSeriesPipelineComponent.load(component_path)
            pipeline.components.append(component)
        
        return pipeline
