import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings

@dataclass
class CausalForecastingConfig:
    """Configuration for causal time series forecasting."""
    
    # Model architecture
    backbone_model: str = "tft"  # tft, transformer, lstm
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_layers: int = 4
    
    # Causal inference settings
    treatment_encoding: str = "one_hot"  # one_hot, linear, cumulative
    outcome_type: str = "continuous"  # continuous, binary, count
    confounding_adjustment: str = "orthogonal"  # orthogonal, ipw, dr
    
    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    cross_fitting_folds: int = 2
    
    # Evaluation settings
    rdd_bandwidth: float = 5.0
    min_support_points: int = 3
    confidence_level: float = 0.95


class CausalTimeSeriesModel(nn.Module, ABC):
    """Abstract base class for causal time series forecasting models."""
    
    def __init__(self, config: CausalForecastingConfig):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, 
                static_features: torch.Tensor,
                temporal_features: torch.Tensor,
                treatment_history: torch.Tensor,
                outcome_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for causal model."""
        pass
    
    @abstractmethod
    def predict_counterfactual(self,
                             input_data: Dict[str, torch.Tensor],
                             treatment_intervention: torch.Tensor) -> torch.Tensor:
        """Predict counterfactual outcomes under treatment intervention."""
        pass


class OrthogonalCausalForecaster(CausalTimeSeriesModel):
    """
    Causal forecasting model using orthogonal statistical learning.
    Based on the approach in Crasson et al. (2024).
    """
    
    def __init__(self, config: CausalForecastingConfig, 
                 input_dim: int, treatment_dim: int, output_dim: int):
        super().__init__(config)
        
        self.input_dim = input_dim
        self.treatment_dim = treatment_dim
        self.output_dim = output_dim
        
        # Three models for orthogonal learning
        self.outcome_model = self._build_outcome_model()  # E[Y|W]
        self.treatment_model = self._build_treatment_model()  # E[T|W]
        self.treatment_effect_model = self._build_treatment_effect_model()  # θ(W)
        
        # Treatment encoding
        if config.treatment_encoding == "one_hot":
            self.treatment_encoder = OneHotTreatmentEncoder(treatment_dim)
        elif config.treatment_encoding == "linear":
            self.treatment_encoder = LinearTreatmentEncoder(treatment_dim)
        elif config.treatment_encoding == "cumulative":
            self.treatment_encoder = CumulativeTreatmentEncoder(treatment_dim)
        
    def _build_outcome_model(self) -> nn.Module:
        """Build model for E[Y|W] - expected outcome given context."""
        
        return nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, self.output_dim)
        )
    
    def _build_treatment_model(self) -> nn.Module:
        """Build model for E[T|W] - expected treatment given context."""
        
        if self.config.outcome_type == "continuous":
            output_activation = nn.Identity()
        elif self.config.outcome_type == "binary":
            output_activation = nn.Sigmoid()
        else:  # count
            output_activation = nn.Softplus()
        
        return nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, self.treatment_dim),
            output_activation
        )
    
    def _build_treatment_effect_model(self) -> nn.Module:
        """Build model for θ(W) - treatment effects given context."""
        
        treatment_effect_dim = self.treatment_dim
        if self.config.treatment_encoding == "one_hot":
            treatment_effect_dim = self.treatment_dim
        
        return nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2), 
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 2, treatment_effect_dim * self.output_dim)
        )
    
    def forward(self, 
                static_features: torch.Tensor,
                temporal_features: torch.Tensor,
                treatment_history: torch.Tensor,
                outcome_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for orthogonal causal model.
        
        Args:
            static_features: Static context features [batch, static_dim]
            temporal_features: Temporal context features [batch, seq_len, temporal_dim]
            treatment_history: Historical treatments [batch, seq_len, treatment_dim]
            outcome_history: Historical outcomes [batch, seq_len, output_dim]
            
        Returns:
            Dictionary with model outputs
        """
        
        batch_size = static_features.size(0)
        
        # Combine context features
        # Flatten temporal features for simplicity (could use RNN/attention)
        temporal_flat = temporal_features.reshape(batch_size, -1)
        context_features = torch.cat([static_features, temporal_flat], dim=1)
        
        # Model predictions
        outcome_pred = self.outcome_model(context_features)  # E[Y|W]
        treatment_pred = self.treatment_model(context_features)  # E[T|W]
        
        # Treatment effects
        treatment_effects_raw = self.treatment_effect_model(context_features)
        treatment_effects = treatment_effects_raw.reshape(
            batch_size, self.treatment_dim, self.output_dim
        )
        
        return {
            'outcome_prediction': outcome_pred,
            'treatment_prediction': treatment_pred,
            'treatment_effects': treatment_effects,
            'context_features': context_features
        }
    
    def compute_orthogonal_loss(self,
                               outputs: Dict[str, torch.Tensor],
                               actual_treatments: torch.Tensor,
                               actual_outcomes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute orthogonal (doubly robust) loss for causal estimation.
        
        Args:
            outputs: Model outputs from forward pass
            actual_treatments: Actual treatment values [batch, treatment_dim]
            actual_outcomes: Actual outcome values [batch, output_dim]
            
        Returns:
            Dictionary with loss components
        """
        
        # Extract predictions
        m_pred = outputs['outcome_prediction']  # E[Y|W]
        e_pred = outputs['treatment_prediction']  # E[T|W]
        theta_pred = outputs['treatment_effects']  # θ(W)
        
        # Encode treatments
        encoded_treatments = self.treatment_encoder.encode(actual_treatments)
        
        # Orthogonal loss (R-loss)
        treatment_residual = actual_treatments - e_pred  # T - E[T|W]
        
        # Calculate treatment effect prediction
        if self.config.treatment_encoding == "one_hot":
            # For categorical treatments
            treatment_effect_pred = torch.sum(
                theta_pred * encoded_treatments.unsqueeze(-1), dim=1
            )
        else:
            # For continuous treatments
            treatment_effect_pred = torch.sum(
                theta_pred * treatment_residual.unsqueeze(-1), dim=1
            )
        
        # Doubly robust residual
        outcome_residual = actual_outcomes - m_pred - treatment_effect_pred
        
        # R-loss: minimize residual correlation with treatment residual
        r_loss = torch.mean(outcome_residual * torch.sum(treatment_residual, dim=1, keepdim=True))
        
        # Auxiliary losses
        outcome_loss = nn.MSELoss()(m_pred, actual_outcomes)
        
        if self.config.outcome_type == "continuous":
            treatment_loss = nn.MSELoss()(e_pred, actual_treatments)
        elif self.config.outcome_type == "binary":
            treatment_loss = nn.BCELoss()(e_pred, actual_treatments)
        else:  # count
            treatment_loss = nn.MSELoss()(e_pred, actual_treatments)
        
        return {
            'r_loss': r_loss,
            'outcome_loss': outcome_loss,
            'treatment_loss': treatment_loss,
            'total_loss': r_loss + 0.1 * (outcome_loss + treatment_loss)
        }
    
    def predict_counterfactual(self,
                             input_data: Dict[str, torch.Tensor],
                             treatment_intervention: torch.Tensor) -> torch.Tensor:
        """Predict counterfactual outcomes under treatment intervention."""
        
        with torch.no_grad():
            outputs = self.forward(**input_data)
            
            # Baseline prediction
            baseline_outcome = outputs['outcome_prediction']
            
            # Treatment effect for intervention
            encoded_intervention = self.treatment_encoder.encode(treatment_intervention)
            
            if self.config.treatment_encoding == "one_hot":
                treatment_effect = torch.sum(
                    outputs['treatment_effects'] * encoded_intervention.unsqueeze(-1), 
                    dim=1
                )
            else:
                expected_treatment = outputs['treatment_prediction']
                treatment_residual = treatment_intervention - expected_treatment
                treatment_effect = torch.sum(
                    outputs['treatment_effects'] * treatment_residual.unsqueeze(-1), 
                    dim=1
                )
            
            # Counterfactual prediction
            counterfactual_outcome = baseline_outcome + treatment_effect
            
            return counterfactual_outcome


class TreatmentEncoder(ABC):
    """Abstract base class for treatment encoding schemes."""
    
    @abstractmethod
    def encode(self, treatments: torch.Tensor) -> torch.Tensor:
        """Encode treatments for model input."""
        pass


class OneHotTreatmentEncoder(TreatmentEncoder):
    """One-hot encoding for categorical treatments."""
    
    def __init__(self, num_categories: int):
        self.num_categories = num_categories
    
    def encode(self, treatments: torch.Tensor) -> torch.Tensor:
        """One-hot encode categorical treatments."""
        # Assume treatments are already one-hot encoded
        return treatments


class LinearTreatmentEncoder(TreatmentEncoder):
    """Linear encoding for continuous treatments."""
    
    def __init__(self, treatment_dim: int):
        self.treatment_dim = treatment_dim
    
    def encode(self, treatments: torch.Tensor) -> torch.Tensor:
        """Return treatments as-is for linear encoding."""
        return treatments


class CumulativeTreatmentEncoder(TreatmentEncoder):
    """Cumulative encoding for sequential treatments."""
    
    def __init__(self, treatment_dim: int):
        self.treatment_dim = treatment_dim
    
    def encode(self, treatments: torch.Tensor) -> torch.Tensor:
        """Encode treatments cumulatively."""
        return torch.cumsum(treatments, dim=1)


class RegressionDiscontinuityEvaluator:
    """
    Evaluate causal models using Regression Discontinuity Design.
    Creates ground truth treatment effects for model evaluation.
    """
    
    def __init__(self, config: CausalForecastingConfig):
        self.config = config
        
    def identify_treatment_switches(self, 
                                   treatment_history: np.ndarray,
                                   time_index: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify treatment discontinuities for RDD evaluation.
        
        Args:
            treatment_history: Treatment values over time [time_steps, treatment_dim]
            time_index: Time indices corresponding to treatments
            
        Returns:
            List of switching events with metadata
        """
        
        switching_events = []
        
        for t in range(1, len(treatment_history)):
            # Check for treatment changes
            current_treatment = treatment_history[t]
            previous_treatment = treatment_history[t-1]
            
            # Detect discontinuity
            if not np.allclose(current_treatment, previous_treatment):
                # Ensure sufficient data before and after
                if (t >= self.config.min_support_points and 
                    t <= len(treatment_history) - self.config.min_support_points):
                    
                    switching_events.append({
                        'switch_time': time_index[t],
                        'switch_index': t,
                        'treatment_before': previous_treatment.copy(),
                        'treatment_after': current_treatment.copy(),
                        'treatment_change': current_treatment - previous_treatment
                    })
        
        return switching_events
    
    def estimate_treatment_effect_rdd(self,
                                     outcome_history: np.ndarray,
                                     time_index: np.ndarray,
                                     switch_event: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate treatment effect using RDD around switching event.
        
        Args:
            outcome_history: Outcome values over time
            time_index: Time indices
            switch_event: Information about treatment switch
            
        Returns:
            Treatment effect estimate with confidence intervals
        """
        
        switch_time = switch_event['switch_time']
        switch_index = switch_event['switch_index']
        
        # Define bandwidth around switch
        bandwidth = self.config.rdd_bandwidth
        
        # Select data within bandwidth
        time_diff = np.abs(time_index - switch_time)
        within_bandwidth = time_diff <= bandwidth
        
        if np.sum(within_bandwidth) < 2 * self.config.min_support_points:
            return {'effect': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        
        # Separate before and after switch
        before_switch = (time_index < switch_time) & within_bandwidth
        after_switch = (time_index >= switch_time) & within_bandwidth
        
        if np.sum(before_switch) < self.config.min_support_points or \
           np.sum(after_switch) < self.config.min_support_points:
            return {'effect': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
        
        # Linear regression on both sides
        before_outcomes = outcome_history[before_switch]
        before_times = time_index[before_switch]
        
        after_outcomes = outcome_history[after_switch]
        after_times = time_index[after_switch]
        
        # Fit linear trends
        if len(before_outcomes) > 1:
            before_coef = np.polyfit(before_times - switch_time, before_outcomes, 1)
            before_intercept = np.polyval(before_coef, 0)  # Value at switch
        else:
            before_intercept = before_outcomes[0]
        
        if len(after_outcomes) > 1:
            after_coef = np.polyfit(after_times - switch_time, after_outcomes, 1)
            after_intercept = np.polyval(after_coef, 0)  # Value at switch
        else:
            after_intercept = after_outcomes[0]
        
        # Treatment effect is discontinuity at switch
        treatment_effect = after_intercept - before_intercept
        
        # Simple confidence interval using bootstrap
        bootstrap_effects = []
        n_bootstrap = 1000
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            before_sample = np.random.choice(before_outcomes, size=len(before_outcomes), replace=True)
            after_sample = np.random.choice(after_outcomes, size=len(after_outcomes), replace=True)
            
            bootstrap_effect = np.mean(after_sample) - np.mean(before_sample)
            bootstrap_effects.append(bootstrap_effect)
        
        bootstrap_effects = np.array(bootstrap_effects)
        alpha = 1 - self.config.confidence_level
        
        ci_lower = np.percentile(bootstrap_effects, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_effects, 100 * (1 - alpha / 2))
        
        return {
            'effect': treatment_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_before': len(before_outcomes),
            'n_after': len(after_outcomes)
        }
    
    def create_causal_test_set(self,
                              time_series_data: pd.DataFrame,
                              treatment_column: str,
                              outcome_column: str,
                              time_column: str) -> List[Dict[str, Any]]:
        """
        Create test set of causal treatment effects using RDD.
        
        Args:
            time_series_data: DataFrame with time series data
            treatment_column: Name of treatment column
            outcome_column: Name of outcome column
            time_column: Name of time column
            
        Returns:
            List of causal test examples
        """
        
        causal_test_set = []
        
        # Convert to numpy for processing
        treatment_history = time_series_data[treatment_column].values
        outcome_history = time_series_data[outcome_column].values
        time_index = time_series_data[time_column].values
        
        # Identify switching events
        switching_events = self.identify_treatment_switches(
            treatment_history.reshape(-1, 1),  # Reshape for multivariate compatibility
            time_index
        )
        
        # Estimate treatment effects for each switching event
        for switch_event in switching_events:
            effect_estimate = self.estimate_treatment_effect_rdd(
                outcome_history,
                time_index,
                switch_event
            )
            
            if not np.isnan(effect_estimate['effect']):
                causal_test_example = {
                    'switch_event': switch_event,
                    'true_treatment_effect': effect_estimate['effect'],
                    'effect_confidence_interval': (
                        effect_estimate['ci_lower'], 
                        effect_estimate['ci_upper']
                    ),
                    'sample_size': effect_estimate['n_before'] + effect_estimate['n_after']
                }
                
                causal_test_set.append(causal_test_example)
        
        return causal_test_set


class CausalForecastingTrainer:
    """Trainer for causal time series forecasting models."""
    
    def __init__(self, model: OrthogonalCausalForecaster, config: CausalForecastingConfig):
        self.model = model
        self.config = config
        
        # Separate optimizers for different model components
        self.outcome_optimizer = torch.optim.Adam(
            self.model.outcome_model.parameters(),
            lr=config.learning_rate
        )
        
        self.treatment_optimizer = torch.optim.Adam(
            self.model.treatment_model.parameters(),
            lr=config.learning_rate
        )
        
        self.effect_optimizer = torch.optim.Adam(
            self.model.treatment_effect_model.parameters(),
            lr=config.learning_rate
        )
        
        self.rdd_evaluator = RegressionDiscontinuityEvaluator(config)
        
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with cross-fitting."""
        
        # Extract batch data
        static_features = batch_data['static_features']
        temporal_features = batch_data['temporal_features']
        treatment_history = batch_data['treatment_history']
        outcome_history = batch_data['outcome_history']
        current_treatments = batch_data['current_treatments']
        current_outcomes = batch_data['current_outcomes']
        
        batch_size = static_features.size(0)
        
        # Cross-fitting: split batch for orthogonal learning
        fold_size = batch_size // self.config.cross_fitting_folds
        total_loss = 0.0
        loss_components = {'r_loss': 0.0, 'outcome_loss': 0.0, 'treatment_loss': 0.0}
        
        for fold in range(self.config.cross_fitting_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.config.cross_fitting_folds - 1 else batch_size
            
            # Split data
            fold_indices = torch.arange(start_idx, end_idx)
            complement_indices = torch.cat([
                torch.arange(0, start_idx),
                torch.arange(end_idx, batch_size)
            ])
            
            # Train auxiliary models on complement set
            self._train_auxiliary_models(batch_data, complement_indices)
            
            # Train treatment effect model on fold set
            fold_data = {key: tensor[fold_indices] for key, tensor in batch_data.items()}
            fold_loss = self._train_treatment_effect_model(fold_data)
            
            total_loss += fold_loss['total_loss']
            for key in loss_components:
                loss_components[key] += fold_loss[key]
        
        # Average losses across folds
        avg_loss = total_loss / self.config.cross_fitting_folds
        avg_components = {key: val / self.config.cross_fitting_folds 
                         for key, val in loss_components.items()}
        
        return {'total_loss': avg_loss, **avg_components}
    
    def _train_auxiliary_models(self, batch_data: Dict[str, torch.Tensor], 
                               indices: torch.Tensor):
        """Train outcome and treatment models on complement set."""
        
        # Extract subset
        subset_data = {key: tensor[indices] for key, tensor in batch_data.items()}
        
        # Forward pass
        outputs = self.model(
            subset_data['static_features'],
            subset_data['temporal_features'],
            subset_data['treatment_history'],
            subset_data['outcome_history']
        )
        
        # Train outcome model
        self.outcome_optimizer.zero_grad()
        outcome_loss = nn.MSELoss()(
            outputs['outcome_prediction'], 
            subset_data['current_outcomes']
        )
        outcome_loss.backward(retain_graph=True)
        self.outcome_optimizer.step()
        
        # Train treatment model
        self.treatment_optimizer.zero_grad()
        if self.config.outcome_type == "continuous":
            treatment_loss = nn.MSELoss()(
                outputs['treatment_prediction'], 
                subset_data['current_treatments']
            )
        else:
            treatment_loss = nn.BCELoss()(
                outputs['treatment_prediction'], 
                subset_data['current_treatments']
            )
        
        treatment_loss.backward()
        self.treatment_optimizer.step()
    
    def _train_treatment_effect_model(self, fold_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train treatment effect model on fold data."""
        
        self.effect_optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            fold_data['static_features'],
            fold_data['temporal_features'],
            fold_data['treatment_history'],
            fold_data['outcome_history']
        )
        
        # Compute orthogonal loss
        losses = self.model.compute_orthogonal_loss(
            outputs,
            fold_data['current_treatments'],
            fold_data['current_outcomes']
        )
        
        # Backward pass
        losses['total_loss'].backward()
        self.effect_optimizer.step()
        
        return {key: loss.item() for key, loss in losses.items()}
    
    def evaluate_causal_performance(self, 
                                   test_data: pd.DataFrame,
                                   treatment_column: str,
                                   outcome_column: str,
                                   time_column: str) -> Dict[str, float]:
        """Evaluate model using causal test set from RDD."""
        
        # Create causal test set
        causal_test_set = self.rdd_evaluator.create_causal_test_set(
            test_data,
            treatment_column,
            outcome_column,
            time_column
        )
        
        if len(causal_test_set) == 0:
            return {'causal_rmse': np.nan, 'causal_mae': np.nan, 'num_test_cases': 0}
        
        predicted_effects = []
        true_effects = []
        
        for test_case in causal_test_set:
            switch_event = test_case['switch_event']
            true_effect = test_case['true_treatment_effect']
            
            # Get model prediction for this switching event
            # This would require preparing the input data around the switch time
            # For simplicity, we'll use a placeholder
            predicted_effect = self._predict_treatment_effect_for_switch(
                test_data, switch_event
            )
            
            if not np.isnan(predicted_effect):
                predicted_effects.append(predicted_effect)
                true_effects.append(true_effect)
        
        if len(predicted_effects) == 0:
            return {'causal_rmse': np.nan, 'causal_mae': np.nan, 'num_test_cases': 0}
        
        # Calculate causal prediction metrics
        predicted_effects = np.array(predicted_effects)
        true_effects = np.array(true_effects)
        
        causal_rmse = np.sqrt(np.mean((predicted_effects - true_effects) ** 2))
        causal_mae = np.mean(np.abs(predicted_effects - true_effects))
        
        return {
            'causal_rmse': causal_rmse,
            'causal_mae': causal_mae,
            'num_test_cases': len(predicted_effects),
            'causal_r2': 1 - np.var(predicted_effects - true_effects) / np.var(true_effects)
        }
    
    def _predict_treatment_effect_for_switch(self,
                                           test_data: pd.DataFrame,
                                           switch_event: Dict[str, Any]) -> float:
        """Predict treatment effect for a specific switching event."""
        
        # This is a simplified placeholder
        # In practice, you would:
        # 1. Extract features around the switch time
        # 2. Convert to model input format
        # 3. Use model to predict treatment effect
        # 4. Return the prediction
        
        # For now, return a random prediction
        return np.random.normal(0, 1)


# Example usage and evaluation
def demonstrate_causal_forecasting():
    """Demonstrate causal time series forecasting."""
    
    print("Generating synthetic causal time series data...")
    
    # Generate synthetic data with causal structure
    n_time_steps = 1000
    n_features = 5
    
    np.random.seed(42)
    
    # Generate confounders (affecting both treatment and outcome)
    confounders = np.random.randn(n_time_steps, n_features)
    
    # Generate treatments influenced by confounders
    treatment_logits = 0.5 * np.sum(confounders, axis=1) + np.random.randn(n_time_steps) * 0.5
    treatments = (treatment_logits > 0).astype(float)
    
    # Generate outcomes with causal effect
    true_treatment_effect = 2.0
    noise = np.random.randn(n_time_steps) * 0.3
    
    outcomes = (0.3 * np.sum(confounders, axis=1) +  # Confounding effect
               true_treatment_effect * treatments +   # Causal effect
               noise)                                 # Noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': np.arange(n_time_steps),
        'treatment': treatments,
        'outcome': outcomes,
        **{f'confounder_{i}': confounders[:, i] for i in range(n_features)}
    })
    
    # Add some treatment switches for RDD evaluation
    switch_times = [200, 400, 600, 800]
    for switch_time in switch_times:
        if switch_time < len(data):
            # Create artificial switch
            data.loc[switch_time:switch_time+50, 'treatment'] = 1 - data.loc[switch_time:switch_time+50, 'treatment']
    
    print(f"Generated {len(data)} time steps with {np.sum(data['treatment'])} treated observations")
    
    # Configuration
    config = CausalForecastingConfig(
        backbone_model="tft",
        hidden_size=128,
        treatment_encoding="linear",
        learning_rate=1e-3,
        batch_size=64,
        rdd_bandwidth=10.0
    )
    
    # Create model
    input_dim = n_features + 1  # confounders + time trend
    treatment_dim = 1
    output_dim = 1
    
    model = OrthogonalCausalForecaster(config, input_dim, treatment_dim, output_dim)
    trainer = CausalForecastingTrainer(model, config)
    
    print("\nEvaluating causal performance using RDD...")
    
    # Evaluate causal performance
    causal_metrics = trainer.evaluate_causal_performance(
        data,
        'treatment',
        'outcome', 
        'time'
    )
    
    print(f"Causal evaluation results:")
    print(f"  Number of test cases: {causal_metrics['num_test_cases']}")
    print(f"  Causal RMSE: {causal_metrics['causal_rmse']:.4f}")
    print(f"  Causal MAE: {causal_metrics['causal_mae']:.4f}")
    
    # Compare with traditional forecasting approach
    print("\nComparing with traditional non-causal forecasting...")
    
    # Simple baseline: predict outcome without considering causality
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Prepare features (confounders + treatment)
    X = np.column_stack([confounders, treatments])
    y = outcomes
    
    # Train-test split (temporal)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Traditional model
    traditional_model = RandomForestRegressor(n_estimators=100, random_state=42)
    traditional_model.fit(X_train, y_train)
    
    y_pred_traditional = traditional_model.predict(X_test)
    
    traditional_rmse = np.sqrt(mean_squared_error(y_test, y_pred_traditional))
    traditional_mae = mean_absolute_error(y_test, y_pred_traditional)
    
    print(f"Traditional forecasting RMSE: {traditional_rmse:.4f}")
    print(f"Traditional forecasting MAE: {traditional_mae:.4f}")
    
    print(f"\nTrue causal effect: {true_treatment_effect:.2f}")
    print("Causal models enable understanding of intervention effects,")
    print("while traditional models only predict correlational patterns.")
    
    return {
        'causal_metrics': causal_metrics,
        'traditional_rmse': traditional_rmse,
        'traditional_mae': traditional_mae,
        'true_effect': true_treatment_effect
    }


if __name__ == "__main__":
    results = demonstrate_causal_forecasting()
    print("\nCausal forecasting enables decision-makers to understand")
    print("the impact of interventions, not just predict future values.")
