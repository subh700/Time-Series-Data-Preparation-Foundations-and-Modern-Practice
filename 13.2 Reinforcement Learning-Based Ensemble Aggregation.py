import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class RLEnsembleAggregator:
    """
    Reinforcement learning-based ensemble aggregator that learns optimal
    combination weights through interaction with forecasting environment.
    """
    
    def __init__(self, n_models: int, config: Dict[str, Any] = None):
        self.n_models = n_models
        self.config = config or self._default_config()
        
        # RL components
        self.actor_critic = ActorCriticNetwork(n_models, self.config)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Experience replay
        self.replay_buffer = deque(maxlen=self.config['buffer_size'])
        
        # Training state
        self.episode_rewards = []
        self.ensemble_weights_history = []
        self.drift_detector = ConceptDriftDetector()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for RL ensemble aggregator."""
        return {
            'learning_rate': 0.001,
            'gamma': 0.95,  # Discount factor
            'buffer_size': 10000,
            'batch_size': 32,
            'update_frequency': 10,
            'exploration_rate': 0.1,
            'exploration_decay': 0.995,
            'hidden_size': 128,
            'drift_sensitivity': 0.05
        }
    
    def train_episode(self, model_predictions: np.ndarray, 
                     true_values: np.ndarray, 
                     context_features: np.ndarray = None) -> Dict[str, float]:
        """
        Train the RL agent through one episode of ensemble aggregation.
        
        Args:
            model_predictions: Shape (time_steps, n_models)
            true_values: Shape (time_steps,)
            context_features: Optional context information
        """
        
        episode_rewards = []
        episode_actions = []
        episode_states = []
        
        # Initialize state
        current_state = self._initialize_state(model_predictions, context_features)
        
        for t in range(len(true_values)):
            # Get current model predictions
            current_predictions = model_predictions[t]
            
            # Choose action (ensemble weights) using actor-critic
            action, action_log_prob = self._select_action(current_state)
            
            # Apply ensemble weights to get final prediction
            ensemble_prediction = np.dot(current_predictions, action)
            
            # Calculate reward based on prediction accuracy
            reward = self._calculate_reward(ensemble_prediction, true_values[t])
            
            # Store experience
            next_state = self._update_state(current_state, current_predictions, 
                                          true_values[t], t)
            
            experience = {
                'state': current_state.copy(),
                'action': action,
                'reward': reward,
                'next_state': next_state.copy(),
                'action_log_prob': action_log_prob,
                'done': t == len(true_values) - 1
            }
            
            self.replay_buffer.append(experience)
            
            # Update metrics
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_states.append(current_state)
            
            current_state = next_state
            
            # Train network periodically
            if len(self.replay_buffer) >= self.config['batch_size'] and \
               t % self.config['update_frequency'] == 0:
                self._update_network()
        
        # Store episode results
        self.episode_rewards.append(np.mean(episode_rewards))
        self.ensemble_weights_history.extend(episode_actions)
        
        # Check for concept drift
        drift_detected = self.drift_detector.detect_drift(
            np.array(episode_rewards), self.config['drift_sensitivity']
        )
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'total_reward': np.sum(episode_rewards),
            'final_weights': episode_actions[-1] if episode_actions else None,
            'drift_detected': drift_detected
        }
    
    def predict_weights(self, model_predictions: np.ndarray,
                       context_features: np.ndarray = None) -> np.ndarray:
        """
        Predict optimal ensemble weights for given model predictions.
        """
        
        self.actor_critic.eval()
        
        # Prepare state
        state = self._prepare_prediction_state(model_predictions, context_features)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.actor_critic(state_tensor)
            weights = torch.softmax(action_probs, dim=1).numpy().flatten()
        
        return weights
    
    def _initialize_state(self, model_predictions: np.ndarray,
                         context_features: np.ndarray = None) -> np.ndarray:
        """Initialize the RL agent state."""
        
        # State includes:
        # 1. Model prediction statistics
        # 2. Historical performance metrics
        # 3. Context features (if available)
        
        pred_stats = self._calculate_prediction_statistics(model_predictions)
        
        if context_features is not None:
            state = np.concatenate([pred_stats, context_features.flatten()])
        else:
            state = pred_stats
        
        return state
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate statistical features from model predictions."""
        
        stats = []
        
        # Basic statistics for each model
        for i in range(self.n_models):
            model_preds = predictions[:, i]
            stats.extend([
                np.mean(model_preds),
                np.std(model_preds),
                np.min(model_preds),
                np.max(model_preds)
            ])
        
        # Cross-model statistics
        stats.extend([
            np.mean(np.std(predictions, axis=1)),  # Average disagreement
            np.mean(predictions),                   # Overall mean
            np.std(predictions.flatten()),          # Overall variance
        ])
        
        return np.array(stats)
    
    def _select_action(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Select ensemble weights using actor-critic policy."""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state_tensor)
        
        # Apply softmax to get valid probability distribution
        action_dist = torch.softmax(action_probs, dim=1)
        
        # Add exploration noise
        if random.random() < self.config['exploration_rate']:
            # Exploration: sample from uniform distribution
            action = np.random.dirichlet(np.ones(self.n_models))
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            action_log_prob = torch.log(action_tensor + 1e-8).sum()
        else:
            # Exploitation: sample from learned policy
            action_tensor = torch.multinomial(action_dist, 1).float()
            action = action_tensor.numpy().flatten()
            
            # Convert to probability distribution
            action = np.zeros(self.n_models)
            action[int(action_tensor.item())] = 1.0
            
            action_log_prob = torch.log(action_dist.gather(1, action_tensor.long())).sum()
        
        return action, action_log_prob
    
    def _calculate_reward(self, prediction: float, true_value: float) -> float:
        """Calculate reward based on prediction accuracy."""
        
        # Primary reward: negative prediction error
        error = abs(prediction - true_value)
        accuracy_reward = -error
        
        # Bonus reward for consistent performance
        consistency_bonus = 0.0
        if len(self.episode_rewards) > 5:
            recent_rewards = self.episode_rewards[-5:]
            consistency_bonus = -np.std(recent_rewards) * 0.1
        
        return accuracy_reward + consistency_bonus
    
    def _update_network(self) -> None:
        """Update actor-critic network using experience replay."""
        
        if len(self.replay_buffer) < self.config['batch_size']:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.config['batch_size'])
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        action_log_probs = torch.stack([exp['action_log_prob'] for exp in batch])
        
        # Current Q-values and value estimates
        action_probs, values = self.actor_critic(states)
        _, next_values = self.actor_critic(next_states)
        
        # Calculate advantages
        targets = rewards + self.config['gamma'] * next_values.squeeze() * (~dones)
        advantages = targets - values.squeeze()
        
        # Actor loss (policy gradient)
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = nn.MSELoss()(values.squeeze(), targets.detach())
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        # Decay exploration rate
        self.config['exploration_rate'] *= self.config['exploration_decay']
    
    def _update_state(self, current_state: np.ndarray, predictions: np.ndarray,
                     true_value: float, time_step: int) -> np.ndarray:
        """Update state based on new observations."""
        
        # Add recent performance information to state
        prediction_errors = np.abs(predictions - true_value)
        
        # Update state with new information
        updated_state = current_state.copy()
        
        # Simple state update: append recent error information
        if len(updated_state) + len(prediction_errors) <= 100:  # Prevent state explosion
            updated_state = np.concatenate([updated_state, prediction_errors])
        else:
            # Replace oldest information with new information
            updated_state[:-len(prediction_errors)] = updated_state[len(prediction_errors):]
            updated_state[-len(prediction_errors):] = prediction_errors
        
        return updated_state
    
    def _prepare_prediction_state(self, model_predictions: np.ndarray,
                                context_features: np.ndarray = None) -> np.ndarray:
        """Prepare state for prediction (without training context)."""
        
        # Use only current model predictions and available context
        pred_stats = self._calculate_prediction_statistics(model_predictions)
        
        if context_features is not None:
            state = np.concatenate([pred_stats, context_features.flatten()])
        else:
            state = pred_stats
        
        return state


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for ensemble weight learning."""
    
    def __init__(self, n_models: int, config: Dict[str, Any]):
        super(ActorCriticNetwork, self).__init__()
        
        self.n_models = n_models
        input_size = self._calculate_input_size(n_models, config)
        hidden_size = config['hidden_size']
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_head = nn.Linear(hidden_size, n_models)
        
        # Critic head (value network)
        self.critic_head = nn.Linear(hidden_size, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic network."""
        
        shared_features = self.shared_layers(state)
        
        # Actor output (action probabilities)
        action_logits = self.actor_head(shared_features)
        
        # Critic output (state value)
        value = self.critic_head(shared_features)
        
        return action_logits, value
    
    def _calculate_input_size(self, n_models: int, config: Dict[str, Any]) -> int:
        """Calculate input size based on state representation."""
        
        # Basic prediction statistics: 4 stats per model + 3 cross-model stats
        base_size = n_models * 4 + 3
        
        # Add space for context features if needed
        context_size = config.get('context_feature_size', 0)
        
        return base_size + context_size


class ConceptDriftDetector:
    """Detect concept drift in time series forecasting performance."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        
    def detect_drift(self, new_performance: np.ndarray, sensitivity: float = 0.05) -> bool:
        """
        Detect concept drift using statistical change point detection.
        
        Args:
            new_performance: Recent performance metrics
            sensitivity: Sensitivity threshold for drift detection
            
        Returns:
            True if drift is detected, False otherwise
        """
        
        # Add new performance to history
        self.performance_history.extend(new_performance)
        
        if len(self.performance_history) < self.window_size:
            return False
        
        # Split history into two halves
        history_array = np.array(self.performance_history)
        split_point = len(history_array) // 2
        
        first_half = history_array[:split_point]
        second_half = history_array[split_point:]
        
        # Perform statistical test for distribution change
        from scipy.stats import ks_2samp
        
        statistic, p_value = ks_2samp(first_half, second_half)
        
        # Drift detected if p-value is below sensitivity threshold
        return p_value < sensitivity
