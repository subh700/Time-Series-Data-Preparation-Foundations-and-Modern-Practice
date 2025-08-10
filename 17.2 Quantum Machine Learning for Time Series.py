import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import warnings

# Note: This is a conceptual implementation
# Real quantum implementations would require quantum hardware/simulators
# like Qiskit, PennyLane, or similar frameworks

@dataclass
class QuantumTSConfig:
    """Configuration for quantum time series models."""
    
    # Quantum circuit parameters
    num_qubits: int = 8
    num_layers: int = 4
    entanglement_type: str = "circular"  # circular, full, linear
    
    # Variational parameters
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 32
    
    # Time series specific
    sequence_length: int = 50
    prediction_horizon: int = 1
    num_features: int = 1
    
    # Quantum-specific
    measurement_shots: int = 1024
    noise_model: Optional[str] = None
    quantum_device: str = "simulator"  # simulator, ibm_quantum, etc.


class QuantumTimeSeriesModel(ABC):
    """Abstract base class for quantum time series models."""
    
    def __init__(self, config: QuantumTSConfig):
        self.config = config
        self.quantum_circuit = None
        self.classical_preprocessing = None
        
    @abstractmethod
    def encode_data(self, time_series: np.ndarray) -> np.ndarray:
        """Encode classical time series data into quantum states."""
        pass
    
    @abstractmethod
    def build_quantum_circuit(self) -> None:
        """Build the parameterized quantum circuit."""
        pass
    
    @abstractmethod
    def measure_expectation_values(self) -> np.ndarray:
        """Measure expectation values from quantum states."""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train the quantum model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained quantum model."""
        pass


class QuantumLSTM(QuantumTimeSeriesModel):
    """
    Quantum-enhanced LSTM for time series forecasting.
    Combines classical LSTM with quantum variational circuits.
    """
    
    def __init__(self, config: QuantumTSConfig):
        super().__init__(config)
        
        # Classical preprocessing layer
        self.classical_lstm = self._build_classical_lstm()
        
        # Quantum variational circuit
        self.build_quantum_circuit()
        
        # Output processing
        self.output_layer = nn.Linear(config.num_qubits, config.prediction_horizon)
        
        # Training parameters
        self.quantum_params = self._initialize_quantum_parameters()
        self.optimizer = torch.optim.Adam([self.quantum_params], lr=config.learning_rate)
    
    def _build_classical_lstm(self) -> nn.Module:
        """Build classical LSTM for initial processing."""
        
        return nn.Sequential(
            nn.LSTM(
                input_size=self.config.num_features,
                hidden_size=self.config.num_qubits,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )[0],  # Take only the LSTM, not the tuple
            nn.Tanh()  # Normalize outputs to [-1, 1] for quantum encoding
        )
    
    def _initialize_quantum_parameters(self) -> torch.Tensor:
        """Initialize variational parameters for quantum circuit."""
        
        # Parameters for rotation gates in each layer
        num_params = self.config.num_qubits * self.config.num_layers * 3  # 3 rotation angles per qubit per layer
        return torch.randn(num_params, requires_grad=True) * 0.1
    
    def build_quantum_circuit(self) -> None:
        """Build parameterized quantum circuit for time series processing."""
        
        # This is a conceptual implementation
        # In practice, you'd use a quantum computing framework
        
        self.quantum_circuit = {
            'num_qubits': self.config.num_qubits,
            'num_layers': self.config.num_layers,
            'entanglement': self.config.entanglement_type,
            'gate_set': ['RX', 'RY', 'RZ', 'CNOT']
        }
    
    def encode_data(self, time_series_features: torch.Tensor) -> torch.Tensor:
        """
        Encode classical time series features into quantum amplitudes.
        
        Args:
            time_series_features: Features from classical LSTM [batch, num_qubits]
            
        Returns:
            Quantum-encoded features
        """
        
        # Amplitude encoding: normalize features to represent quantum amplitudes
        normalized_features = torch.tanh(time_series_features)
        
        # Ensure the features sum to 1 for valid quantum amplitudes
        amplitude_encoded = torch.softmax(normalized_features, dim=-1)
        
        return amplitude_encoded
    
    def quantum_layer(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum variational circuit.
        
        Args:
            encoded_features: Quantum-encoded features
            
        Returns:
            Processed quantum features
        """
        
        batch_size = encoded_features.size(0)
        output_features = torch.zeros_like(encoded_features)
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Initialize quantum state with encoded features
            quantum_state = encoded_features[i]
            
            # Apply variational quantum circuit layers
            param_idx = 0
            for layer in range(self.config.num_layers):
                # Apply parameterized rotation gates
                for qubit in range(self.config.num_qubits):
                    # RX rotation
                    rx_angle = self.quantum_params[param_idx]
                    quantum_state[qubit] = self._apply_rx_rotation(
                        quantum_state[qubit], rx_angle
                    )
                    param_idx += 1
                    
                    # RY rotation
                    ry_angle = self.quantum_params[param_idx]
                    quantum_state[qubit] = self._apply_ry_rotation(
                        quantum_state[qubit], ry_angle
                    )
                    param_idx += 1
                    
                    # RZ rotation
                    rz_angle = self.quantum_params[param_idx]
                    quantum_state[qubit] = self._apply_rz_rotation(
                        quantum_state[qubit], rz_angle
                    )
                    param_idx += 1
                
                # Apply entangling gates
                quantum_state = self._apply_entangling_layer(quantum_state)
            
            output_features[i] = quantum_state
        
        return output_features
    
    def _apply_rx_rotation(self, amplitude: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Apply RX rotation gate (simplified classical simulation)."""
        return amplitude * torch.cos(angle / 2) + 1j * amplitude * torch.sin(angle / 2)
    
    def _apply_ry_rotation(self, amplitude: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Apply RY rotation gate (simplified classical simulation)."""
        return amplitude * torch.cos(angle / 2) + amplitude * torch.sin(angle / 2)
    
    def _apply_rz_rotation(self, amplitude: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Apply RZ rotation gate (simplified classical simulation)."""
        return amplitude * torch.exp(1j * angle / 2)
    
    def _apply_entangling_layer(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Apply entangling gates between qubits."""
        
        if self.config.entanglement_type == "circular":
            # Circular entanglement
            for i in range(self.config.num_qubits):
                j = (i + 1) % self.config.num_qubits
                quantum_state = self._apply_cnot(quantum_state, i, j)
        
        elif self.config.entanglement_type == "linear":
            # Linear entanglement
            for i in range(self.config.num_qubits - 1):
                quantum_state = self._apply_cnot(quantum_state, i, i + 1)
        
        return quantum_state
    
    def _apply_cnot(self, quantum_state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate (simplified classical simulation)."""
        
        # Simplified CNOT implementation
        # In reality, this would involve proper quantum state manipulation
        quantum_state[target] = quantum_state[target] * quantum_state[control]
        
        return quantum_state
    
    def measure_expectation_values(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Measure expectation values from quantum states."""
        
        # Measure in Pauli-Z basis (simplified)
        if torch.is_complex(quantum_state):
            expectation_values = torch.real(quantum_state * torch.conj(quantum_state))
        else:
            expectation_values = quantum_state ** 2
        
        return expectation_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum LSTM.
        
        Args:
            x: Input time series [batch, seq_len, features]
            
        Returns:
            Predictions [batch, prediction_horizon]
        """
        
        # Classical LSTM processing
        lstm_output, _ = self.classical_lstm(x)
        
        # Take the last timestep output
        last_hidden = lstm_output[:, -1, :]  # [batch, num_qubits]
        
        # Quantum processing
        encoded_features = self.encode_data(last_hidden)
        quantum_processed = self.quantum_layer(encoded_features)
        
        # Measure expectation values
        measured_values = self.measure_expectation_values(quantum_processed)
        
        # Final prediction
        predictions = self.output_layer(measured_values)
        
        return predictions
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train the quantum LSTM model."""
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        training_losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.forward(batch_x)
                
                # Calculate loss
                loss = nn.MSELoss()(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return {'training_losses': training_losses}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained quantum model."""
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()


class QuantumReservoirComputing(QuantumTimeSeriesModel):
    """
    Quantum Reservoir Computing for time series forecasting.
    Uses quantum dynamics as a computational reservoir.
    """
    
    def __init__(self, config: QuantumTSConfig):
        super().__init__(config)
        
        # Quantum reservoir parameters
        self.reservoir_size = config.num_qubits
        self.reservoir_params = self._initialize_reservoir()
        
        # Readout layer (classical)
        self.readout_layer = nn.Linear(
            config.num_qubits, 
            config.prediction_horizon
        )
        
        # Training state
        self.reservoir_states = []
        
    def _initialize_reservoir(self) -> Dict[str, torch.Tensor]:
        """Initialize quantum reservoir parameters."""
        
        return {
            'coupling_strengths': torch.randn(self.config.num_qubits) * 0.1,
            'local_fields': torch.randn(self.config.num_qubits) * 0.1,
            'interaction_matrix': torch.randn(
                self.config.num_qubits, 
                self.config.num_qubits
            ) * 0.05
        }
    
    def build_quantum_circuit(self) -> None:
        """Build quantum reservoir circuit."""
        
        # Reservoir circuit implements time evolution
        self.quantum_circuit = {
            'reservoir_type': 'quantum_spin_chain',
            'evolution_time': 1.0,
            'num_time_steps': 10
        }
    
    def encode_data(self, time_series: np.ndarray) -> torch.Tensor:
        """Encode input data into quantum reservoir."""
        
        # Convert to tensor
        if isinstance(time_series, np.ndarray):
            time_series = torch.FloatTensor(time_series)
        
        # Feature encoding: map time series values to rotation angles
        encoded_angles = torch.atan(time_series)  # Map to [-π/2, π/2]
        
        return encoded_angles
    
    def evolve_reservoir(self, 
                        input_data: torch.Tensor, 
                        time_steps: int = 10) -> torch.Tensor:
        """Evolve quantum reservoir with input data."""
        
        batch_size, seq_len, n_features = input_data.shape
        reservoir_states = torch.zeros(batch_size, seq_len, self.reservoir_size)
        
        for batch_idx in range(batch_size):
            # Initialize reservoir state
            current_state = torch.randn(self.reservoir_size) * 0.1
            
            for t in range(seq_len):
                # Encode current input
                input_encoded = self.encode_data(input_data[batch_idx, t, :])
                
                # Apply input to reservoir (drive the system)
                driven_state = current_state + 0.1 * input_encoded[:self.reservoir_size]
                
                # Evolve reservoir dynamics
                for step in range(time_steps):
                    driven_state = self._quantum_evolution_step(driven_state)
                
                # Store reservoir state
                reservoir_states[batch_idx, t, :] = driven_state
                current_state = driven_state
        
        return reservoir_states
    
    def _quantum_evolution_step(self, state: torch.Tensor) -> torch.Tensor:
        """Single step of quantum reservoir evolution."""
        
        # Simplified quantum evolution (Hamiltonian simulation)
        dt = 0.1  # Time step
        
        # Local field terms
        local_evolution = state * torch.cos(
            self.reservoir_params['local_fields'] * dt
        )
        
        # Interaction terms (simplified)
        interaction_term = torch.matmul(
            self.reservoir_params['interaction_matrix'],
            torch.sin(state * dt)
        )
        
        # Combined evolution
        evolved_state = local_evolution + 0.1 * interaction_term
        
        # Add some nonlinearity and normalization
        evolved_state = torch.tanh(evolved_state)
        
        return evolved_state
    
    def measure_expectation_values(self) -> np.ndarray:
        """Measure expectation values from reservoir states."""
        
        if not self.reservoir_states:
            raise ValueError("No reservoir states to measure")
        
        # Measure in computational basis
        measurements = []
        for state_sequence in self.reservoir_states:
            # Take expectation values as |amplitude|^2
            expectation_values = torch.abs(state_sequence) ** 2
            measurements.append(expectation_values)
        
        return torch.stack(measurements)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train quantum reservoir computing model."""
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Evolve reservoir with all training data
        print("Evolving quantum reservoir...")
        reservoir_states = self.evolve_reservoir(X_tensor)
        
        # Use final reservoir states for training readout
        final_states = reservoir_states[:, -1, :]  # [batch, reservoir_size]
        
        # Train linear readout layer
        optimizer = torch.optim.Adam(self.readout_layer.parameters(), lr=0.01)
        
        training_losses = []
        
        for epoch in range(100):  # Train readout for 100 epochs
            optimizer.zero_grad()
            
            predictions = self.readout_layer(final_states)
            loss = nn.MSELoss()(predictions, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Readout training epoch {epoch}, Loss: {loss.item():.6f}")
        
        return {'training_losses': training_losses}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using quantum reservoir."""
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            # Evolve reservoir with test data
            reservoir_states = self.evolve_reservoir(X_tensor)
            final_states = reservoir_states[:, -1, :]
            
            # Make predictions with trained readout
            predictions = self.readout_layer(final_states)
        
        return predictions.numpy()


class QuantumNeuralNetwork(QuantumTimeSeriesModel):
    """
    Variational Quantum Neural Network for time series forecasting.
    """
    
    def __init__(self, config: QuantumTSConfig):
        super().__init__(config)
        
        # Classical preprocessing
        self.input_layer = nn.Sequential(
            nn.Linear(config.num_features, config.num_qubits),
            nn.Tanh()
        )
        
        # Quantum processing
        self.build_quantum_circuit()
        
        # Output processing
        self.output_layer = nn.Sequential(
            nn.Linear(config.num_qubits, config.num_qubits // 2),
            nn.ReLU(),
            nn.Linear(config.num_qubits // 2, config.prediction_horizon)
        )
        
        # Variational parameters
        self.theta = nn.Parameter(
            torch.randn(config.num_layers * config.num_qubits * 3) * 0.1
        )
    
    def build_quantum_circuit(self) -> None:
        """Build variational quantum neural network circuit."""
        
        self.quantum_circuit = {
            'ansatz_type': 'hardware_efficient',
            'num_parameters': self.config.num_layers * self.config.num_qubits * 3,
            'measurement_basis': 'pauli_z'
        }
    
    def encode_data(self, classical_features: torch.Tensor) -> torch.Tensor:
        """Encode classical features into quantum circuit."""
        
        # Angle encoding
        encoded = torch.arctan(classical_features)
        return encoded
    
    def quantum_neural_network_layer(self, 
                                   encoded_data: torch.Tensor) -> torch.Tensor:
        """Apply variational quantum neural network."""
        
        batch_size = encoded_data.size(0)
        quantum_outputs = torch.zeros_like(encoded_data)
        
        param_idx = 0
        
        for batch_idx in range(batch_size):
            # Initialize quantum state with encoded data
            quantum_state = encoded_data[batch_idx]
            
            # Apply variational layers
            for layer in range(self.config.num_layers):
                # Parameterized single-qubit rotations
                for qubit in range(self.config.num_qubits):
                    # Three rotation parameters per qubit
                    rx_param = self.theta[param_idx]
                    ry_param = self.theta[param_idx + 1]
                    rz_param = self.theta[param_idx + 2]
                    
                    # Apply rotations (simplified)
                    quantum_state[qubit] = quantum_state[qubit] * torch.cos(rx_param) + \
                                         torch.sin(ry_param) * torch.cos(rz_param)
                    
                    param_idx += 3
                
                # Reset parameter index for next batch (parameters are shared)
                param_idx = param_idx % len(self.theta)
                
                # Apply entangling layer
                quantum_state = self._apply_entangling_gates(quantum_state)
            
            quantum_outputs[batch_idx] = quantum_state
        
        return quantum_outputs
    
    def _apply_entangling_gates(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Apply entangling gates in the quantum circuit."""
        
        # Simplified entanglement (correlation between adjacent qubits)
        for i in range(self.config.num_qubits - 1):
            correlation = torch.cos(quantum_state[i] - quantum_state[i + 1])
            quantum_state[i] = quantum_state[i] * correlation
            quantum_state[i + 1] = quantum_state[i + 1] * correlation
        
        return quantum_state
    
    def measure_expectation_values(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Measure expectation values from quantum state."""
        
        # Pauli-Z measurement (simplified)
        expectation_values = torch.tanh(quantum_state)
        
        return expectation_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum neural network."""
        
        # Classical preprocessing
        processed_input = self.input_layer(x[:, -1, :])  # Use last timestep
        
        # Quantum processing
        encoded_data = self.encode_data(processed_input)
        quantum_output = self.quantum_neural_network_layer(encoded_data)
        measured_values = self.measure_expectation_values(quantum_output)
        
        # Classical postprocessing
        predictions = self.output_layer(measured_values)
        
        return predictions
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train the variational quantum neural network."""
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        training_losses = []
        
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            
            predictions = self.forward(X_tensor)
            loss = nn.MSELoss()(predictions, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        return {'training_losses': training_losses}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained quantum neural network."""
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()


# Hybrid Classical-Quantum Model
class HybridQuantumClassicalForecaster:
    """
    Hybrid model combining classical and quantum components
    for enhanced time series forecasting.
    """
    
    def __init__(self, config: QuantumTSConfig):
        self.config = config
        
        # Classical components
        self.classical_encoder = nn.LSTM(
            input_size=config.num_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.classical_decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, config.prediction_horizon)
        )
        
        # Quantum component
        self.quantum_processor = QuantumNeuralNetwork(config)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.prediction_horizon * 2, config.prediction_horizon),
            nn.Tanh()
        )
        
        # Training configuration
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def parameters(self):
        """Get all trainable parameters."""
        params = []
        params.extend(self.classical_encoder.parameters())
        params.extend(self.classical_decoder.parameters())
        params.extend(self.quantum_processor.parameters())
        params.extend(self.fusion_layer.parameters())
        return params
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid model."""
        
        # Classical processing
        lstm_out, _ = self.classical_encoder(x)
        classical_pred = self.classical_decoder(lstm_out[:, -1, :])
        
        # Quantum processing
        quantum_pred = self.quantum_processor(x)
        
        # Fusion
        combined = torch.cat([classical_pred, quantum_pred], dim=-1)
        final_pred = self.fusion_layer(combined)
        
        return {
            'final_prediction': final_pred,
            'classical_prediction': classical_pred,
            'quantum_prediction': quantum_pred
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train the hybrid quantum-classical model."""
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        training_losses = []
        classical_losses = []
        quantum_losses = []
        fusion_losses = []
        
        for epoch in range(self.config.num_epochs):
            self.optimizer.zero_grad()
            
            outputs = self.forward(X_tensor)
            
            # Multi-component loss
            final_loss = nn.MSELoss()(outputs['final_prediction'], y_tensor)
            classical_loss = nn.MSELoss()(outputs['classical_prediction'], y_tensor)
            quantum_loss = nn.MSELoss()(outputs['quantum_prediction'], y_tensor)
            
            # Combined loss with weighting
            total_loss = final_loss + 0.3 * classical_loss + 0.3 * quantum_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # Record losses
            training_losses.append(total_loss.item())
            classical_losses.append(classical_loss.item())
            quantum_losses.append(quantum_loss.item())
            fusion_losses.append(final_loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}")
                print(f"  Total Loss: {total_loss.item():.6f}")
                print(f"  Classical: {classical_loss.item():.6f}")
                print(f"  Quantum: {quantum_loss.item():.6f}")
                print(f"  Fusion: {final_loss.item():.6f}")
        
        return {
            'training_losses': training_losses,
            'classical_losses': classical_losses,
            'quantum_losses': quantum_losses,
            'fusion_losses': fusion_losses
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions with component breakdown."""
        
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
        
        return {
            'final_prediction': outputs['final_prediction'].numpy(),
            'classical_prediction': outputs['classical_prediction'].numpy(),
            'quantum_prediction': outputs['quantum_prediction'].numpy()
        }


# Example usage and comparison
if __name__ == "__main__":
    # Configuration
    config = QuantumTSConfig(
        num_qubits=8,
        num_layers=3,
        sequence_length=50,
        prediction_horizon=1,
        num_features=1,
        num_epochs=100,
        learning_rate=0.01
    )
    
    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    seq_len = config.sequence_length
    
    # Create synthetic time series with trend and seasonality
    time_steps = np.arange(n_samples + seq_len)
    trend = 0.001 * time_steps
    seasonal = 2 * np.sin(2 * np.pi * time_steps / 50)
    noise = 0.1 * np.random.randn(len(time_steps))
    time_series = trend + seasonal + noise
    
    # Prepare training data
    X, y = [], []
    for i in range(n_samples):
        X.append(time_series[i:i+seq_len].reshape(-1, 1))
        y.append([time_series[i+seq_len]])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Training data shape:", X_train.shape, y_train.shape)
    
    # Test different quantum models
    models = {
        'Quantum LSTM': QuantumLSTM(config),
        'Quantum Reservoir': QuantumReservoirComputing(config),
        'Quantum Neural Network': QuantumNeuralNetwork(config),
        'Hybrid Model': HybridQuantumClassicalForecaster(config)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Train model
        training_history = model.train(X_train, y_train)
        
        # Make predictions
        if isinstance(model, HybridQuantumClassicalForecaster):
            predictions_dict = model.predict(X_test)
            predictions = predictions_dict['final_prediction']
        else:
            predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions.flatten() - y_test.flatten()))
        mse = np.mean((predictions.flatten() - y_test.flatten()) ** 2)
        
        results[model_name] = {
            'mae': mae,
            'mse': mse,
            'training_history': training_history
        }
        
        print(f"Test MAE: {mae:.6f}")
        print(f"Test MSE: {mse:.6f}")
    
    # Print comparison
    print("\n=== Model Comparison ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: MAE={metrics['mae']:.6f}, MSE={metrics['mse']:.6f}")
