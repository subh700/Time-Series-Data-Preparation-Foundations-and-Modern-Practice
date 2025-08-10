import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

class NeuralProphetModel(nn.Module):
    """
    Advanced Neural Prophet implementation with probabilistic forecasting
    and uncertainty quantification capabilities.
    """
    
    def __init__(self, growth: str = 'linear', 
                 seasonalities: Dict[str, Dict] = None,
                 num_hidden_layers: int = 0,
                 d_hidden: int = 4,
                 ar_reg: Optional[float] = None,
                 learning_rate: float = 0.001,
                 quantiles: List[float] = None,
                 uncertainty_samples: int = 1000):
        super(NeuralProphetModel, self).__init__()
        
        self.growth = growth
        self.seasonalities = seasonalities or {'yearly': {'period': 365.25, 'fourier_order': 10}}
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = d_hidden
        self.ar_reg = ar_reg
        self.learning_rate = learning_rate
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.uncertainty_samples = uncertainty_samples
        
        # Initialize components
        self._initialize_components()
        
        # Uncertainty quantification components
        self.uncertainty_estimator = UncertaintyEstimator(uncertainty_samples)
        self.conformal_predictor = None  # Will be initialized during fitting
        
    def _initialize_components(self):
        """Initialize model components."""
        
        # Trend component
        if self.growth == 'linear':
            self.trend_layer = LinearTrend()
        elif self.growth == 'logistic':
            self.trend_layer = LogisticTrend()
        else:
            raise ValueError(f"Unknown growth type: {self.growth}")
        
        # Seasonality components
        self.seasonality_layers = nn.ModuleDict()
        for name, config in self.seasonalities.items():
            self.seasonality_layers[name] = FourierSeasonality(
                period=config['period'],
                fourier_order=config['fourier_order']
            )
        
        # Autoregressive component (if specified)
        if self.ar_reg is not None:
            self.ar_layer = AutoregressiveLayer(
                lags=self.ar_reg,
                hidden_size=self.d_hidden,
                num_layers=self.num_hidden_layers
            )
        else:
            self.ar_layer = None
        
        # Neural network component for covariates
        if self.num_hidden_layers > 0:
            layers = []
            input_size = 1  # Will be adjusted based on actual input
            
            for i in range(self.num_hidden_layers):
                layers.extend([
                    nn.Linear(input_size if i == 0 else self.d_hidden, self.d_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            
            layers.append(nn.Linear(self.d_hidden, 1))
            self.nn_component = nn.Sequential(*layers)
        else:
            self.nn_component = None
        
        # Quantile regression heads
        if len(self.quantiles) > 1:
            self.quantile_heads = nn.ModuleDict({
                f'q_{q}': nn.Linear(1, 1) for q in self.quantiles
            })
        else:
            self.quantile_heads = None
    
    def forward(self, t: torch.Tensor, features: Optional[torch.Tensor] = None,
               ar_inputs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Neural Prophet model.
        
        Args:
            t: Time tensor
            features: Optional additional features
            ar_inputs: Optional autoregressive inputs
            
        Returns:
            Dictionary containing predictions and components
        """
        
        # Trend component
        trend = self.trend_layer(t)
        
        # Seasonality components
        seasonality = torch.zeros_like(trend)
        seasonality_components = {}
        
        for name, layer in self.seasonality_layers.items():
            seasonal_component = layer(t)
            seasonality += seasonal_component
            seasonality_components[name] = seasonal_component
        
        # Autoregressive component
        ar_component = torch.zeros_like(trend)
        if self.ar_layer is not None and ar_inputs is not None:
            ar_component = self.ar_layer(ar_inputs)
        
        # Neural network component for additional features
        nn_component = torch.zeros_like(trend)
        if self.nn_component is not None and features is not None:
            nn_component = self.nn_component(features)
        
        # Combine components
        base_prediction = trend + seasonality + ar_component + nn_component
        
        # Generate quantile predictions if specified
        quantile_predictions = {}
        if self.quantile_heads is not None:
            for q_name, q_head in self.quantile_heads.items():
                quantile_predictions[q_name] = q_head(base_prediction)
        else:
            quantile_predictions['q_0.5'] = base_prediction
        
        return {
            'predictions': quantile_predictions,
            'trend': trend,
            'seasonality': seasonality,
            'seasonality_components': seasonality_components,
            'ar_component': ar_component,
            'nn_component': nn_component
        }
    
    def fit(self, df: pd.DataFrame, freq: str = 'D',
           validation_df: Optional[pd.DataFrame] = None,
           epochs: int = 100) -> Dict[str, List[float]]:
        """
        Fit Neural Prophet model to data.
        
        Args:
            df: Training dataframe with 'ds' and 'y' columns
            freq: Data frequency
            validation_df: Optional validation data
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(df, validation_df, freq)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(**batch['inputs'])
                
                # Calculate loss
                loss = self._calculate_loss(outputs, batch['targets'])
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            
            # Validation phase
            if val_loader:
                self.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = self(**batch['inputs'])
                        loss = self._calculate_loss(outputs, batch['targets'])
                        val_loss += loss.item()
                
                val_losses.append(val_loss / len(val_loader))
                scheduler.step(val_losses[-1])
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}" +
                      (f", Val Loss = {val_losses[-1]:.4f}" if val_losses else ""))
        
        # Initialize conformal prediction after training
        if validation_df is not None:
            self._initialize_conformal_prediction(df, validation_df, freq)
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def predict(self, df: pd.DataFrame, freq: str = 'D',
               uncertainty: bool = True, 
               decompose: bool = True) -> pd.DataFrame:
        """
        Generate predictions with uncertainty quantification.
        
        Args:
            df: Prediction dataframe with 'ds' column
            freq: Data frequency
            uncertainty: Whether to include uncertainty estimates
            decompose: Whether to decompose prediction components
            
        Returns:
            DataFrame with predictions and uncertainties
        """
        
        self.eval()
        
        # Prepare prediction data
        pred_data = self._prepare_prediction_data(df, freq)
        
        results = []
        
        with torch.no_grad():
            for batch in pred_data:
                # Base predictions
                outputs = self(**batch)
                
                # Extract quantile predictions
                predictions = {}
                for q_key, q_pred in outputs['predictions'].items():
                    quantile = float(q_key.replace('q_', ''))
                    predictions[f'yhat_{quantile}'] = q_pred.numpy().flatten()
                
                # Add component decomposition if requested
                if decompose:
                    predictions['trend'] = outputs['trend'].numpy().flatten()
                    predictions['seasonal'] = outputs['seasonality'].numpy().flatten()
                    
                    for name, component in outputs['seasonality_components'].items():
                        predictions[f'seasonal_{name}'] = component.numpy().flatten()
                    
                    if outputs['ar_component'] is not None:
                        predictions['ar'] = outputs['ar_component'].numpy().flatten()
                
                results.append(predictions)
        
        # Combine results
        result_df = pd.DataFrame()
        for i, batch_results in enumerate(results):
            batch_df = pd.DataFrame(batch_results)
            result_df = pd.concat([result_df, batch_df], ignore_index=True)
        
        # Add uncertainty estimates if requested
        if uncertainty and self.conformal_predictor is not None:
            uncertainty_estimates = self._generate_uncertainty_estimates(result_df)
            result_df = pd.concat([result_df, uncertainty_estimates], axis=1)
        
        # Add time index
        result_df['ds'] = df['ds'].values
        
        return result_df
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], 
                       targets: torch.Tensor) -> torch.Tensor:
        """Calculate quantile regression loss."""
        
        if len(self.quantiles) == 1:
            # Standard MSE loss
            predictions = outputs['predictions']['q_0.5']
            return nn.MSELoss()(predictions.squeeze(), targets.squeeze())
        
        # Quantile regression loss
        total_loss = 0.0
        
        for q_key, q_pred in outputs['predictions'].items():
            quantile = float(q_key.replace('q_', ''))
            
            # Quantile loss
            errors = targets.squeeze() - q_pred.squeeze()
            loss = torch.maximum(
                quantile * errors,
                (quantile - 1) * errors
            )
            
            total_loss += loss.mean()
        
        return total_loss
    
    def _initialize_conformal_prediction(self, train_df: pd.DataFrame,
                                       val_df: pd.DataFrame, freq: str):
        """Initialize conformal prediction using validation data."""
        
        from sklearn.dummy import DummyRegressor
        
        # Create a dummy base predictor for conformal prediction
        base_predictor = DummyRegressor(strategy='mean')
        
        self.conformal_predictor = TimeSeriesConformalPredictor(
            base_predictor=base_predictor,
            method='adaptive_copula',
            coverage_level=0.9
        )
        
        # Generate predictions on validation set for calibration
        val_predictions = self.predict(val_df, freq, uncertainty=False, decompose=False)
        
        # Calibrate conformal predictor
        X_cal = val_predictions[['yhat_0.5']].values
        y_cal = val_df['y'].values
        
        self.conformal_predictor.calibrate(X_cal, y_cal)
    
    def _generate_uncertainty_estimates(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive uncertainty estimates."""
        
        uncertainty_results = {}
        
        # Conformal prediction intervals
        if self.conformal_predictor is not None:
            base_predictions = predictions_df[['yhat_0.5']].values
            lower_bounds, upper_bounds = self.conformal_predictor.predict_interval(
                base_predictions
            )
            
            uncertainty_results['yhat_lower'] = lower_bounds.flatten()
            uncertainty_results['yhat_upper'] = upper_bounds.flatten()
            uncertainty_results['yhat_width'] = upper_bounds.flatten() - lower_bounds.flatten()
        
        # Quantile-based uncertainty (if available)
        if 'yhat_0.1' in predictions_df.columns and 'yhat_0.9' in predictions_df.columns:
            uncertainty_results['quantile_lower'] = predictions_df['yhat_0.1'].values
            uncertainty_results['quantile_upper'] = predictions_df['yhat_0.9'].values
            uncertainty_results['quantile_width'] = (
                predictions_df['yhat_0.9'].values - predictions_df['yhat_0.1'].values
            )
        
        # Monte Carlo uncertainty estimation
        mc_uncertainty = self.uncertainty_estimator.estimate_uncertainty(
            predictions_df['yhat_0.5'].values
        )
        
        uncertainty_results['mc_std'] = mc_uncertainty['std']
        uncertainty_results['mc_lower'] = mc_uncertainty['lower']
        uncertainty_results['mc_upper'] = mc_uncertainty['upper']
        
        return pd.DataFrame(uncertainty_results)


class LinearTrend(nn.Module):
    """Linear trend component."""
    
    def __init__(self):
        super(LinearTrend, self).__init__()
        self.k = nn.Parameter(torch.tensor(1.0))  # Growth rate
        self.m = nn.Parameter(torch.tensor(0.0))  # Offset
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.k * t + self.m


class LogisticTrend(nn.Module):
    """Logistic trend component with carrying capacity."""
    
    def __init__(self):
        super(LogisticTrend, self).__init__()
        self.k = nn.Parameter(torch.tensor(1.0))  # Growth rate
        self.m = nn.Parameter(torch.tensor(0.0))  # Offset
        self.C = nn.Parameter(torch.tensor(10.0))  # Carrying capacity
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.C / (1 + torch.exp(-self.k * (t - self.m)))


class FourierSeasonality(nn.Module):
    """Fourier-based seasonality component."""
    
    def __init__(self, period: float, fourier_order: int):
        super(FourierSeasonality, self).__init__()
        
        self.period = period
        self.fourier_order = fourier_order
        
        # Fourier coefficients
        self.beta = nn.Parameter(torch.randn(fourier_order * 2))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Fourier seasonality."""
        
        # Create Fourier features
        features = []
        
        for i in range(1, self.fourier_order + 1):
            # Sine and cosine components
            sin_feature = torch.sin(2 * np.pi * i * t / self.period)
            cos_feature = torch.cos(2 * np.pi * i * t / self.period)
            
            features.extend([sin_feature, cos_feature])
        
        # Stack features
        X = torch.stack(features, dim=-1)  # (batch_size, fourier_order * 2)
        
        # Linear combination with learned coefficients
        return torch.matmul(X, self.beta)


class AutoregressiveLayer(nn.Module):
    """Autoregressive neural network layer."""
    
    def __init__(self, lags: int, hidden_size: int = 10, num_layers: int = 1):
        super(AutoregressiveLayer, self).__init__()
        
        self.lags = lags
        
        # Neural network for autoregressive component
        layers = []
        input_size = lags
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoregressive network."""
        return self.network(x)


class UncertaintyEstimator:
    """Monte Carlo-based uncertainty estimator."""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
    
    def estimate_uncertainty(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate uncertainty using Monte Carlo sampling."""
        
        # Simple Monte Carlo uncertainty estimation
        # In practice, this would involve model ensemble or dropout-based sampling
        
        base_predictions = predictions
        
        # Simulate prediction variance (placeholder implementation)
        prediction_std = 0.1 * np.abs(base_predictions) + 0.05
        
        # Generate samples
        samples = np.random.normal(
            base_predictions[:, np.newaxis],
            prediction_std[:, np.newaxis],
            size=(len(base_predictions), self.num_samples)
        )
        
        # Calculate statistics
        mc_mean = np.mean(samples, axis=1)
        mc_std = np.std(samples, axis=1)
        mc_lower = np.percentile(samples, 5, axis=1)
        mc_upper = np.percentile(samples, 95, axis=1)
        
        return {
            'mean': mc_mean,
            'std': mc_std,
            'lower': mc_lower,
            'upper': mc_upper
        }
