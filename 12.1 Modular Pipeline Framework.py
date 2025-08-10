from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging
from pathlib import Path

class TimeSeriesPipelineComponent(ABC):
    """
    Abstract base class for time series pipeline components.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.metadata = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'TimeSeriesPipelineComponent':
        """Fit the component to training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform data using fitted component."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit component and transform data in one step."""
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def save(self, path: Path) -> None:
        """Save fitted component to disk."""
        if not self.is_fitted:
            raise RuntimeError(f"Component '{self.name}' must be fitted before saving")
        
        component_data = {
            'name': self.name,
            'config': self.config,
            'metadata': self.metadata,
            'is_fitted': self.is_fitted,
            'component_state': self._get_state()
        }
        
        joblib.dump(component_data, path)
        self.logger.info(f"Component saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'TimeSeriesPipelineComponent':
        """Load fitted component from disk."""
        component_data = joblib.load(path)
        
        instance = cls(
            name=component_data['name'],
            config=component_data['config']
        )
        
        instance.metadata = component_data['metadata']
        instance.is_fitted = component_data['is_fitted']
        instance._set_state(component_data['component_state'])
        
        return instance
    
    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """Get component state for serialization."""
        pass
    
    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Set component state from deserialization."""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Get names of output features."""
        return self.metadata.get('feature_names', [])
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and content."""
        required_columns = self.config.get('required_columns', [])
        
        if not all(col in data.columns for col in required_columns):
            missing = set(required_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        return True
