import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch

class GraphTimeSeriesTransformer:
    """
    Graph Transformer for time series forecasting that leverages 
    relational information between time series entities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Model components
        self.graph_transformer = None
        self.forecasting_head = None
        self.entity_embeddings = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Graph construction
        self.entity_graph = None
        self.temporal_samplers = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Graph Transformer."""
        return {
            'entity_embedding_dim': 64,
            'temporal_embedding_dim': 32,
            'transformer_dim': 128,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'sequence_length': 96,
            'prediction_horizon': 24,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'edge_threshold': 0.5,
            'graph_update_frequency': 10
        }
    
    def build_model(self, n_entities: int, n_features: int,
                   entity_features: Optional[Dict[str, Any]] = None) -> None:
        """Build Graph Transformer architecture."""
        
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(
            n_entities, self.config['entity_embedding_dim']
        )
        
        # Graph Transformer
        self.graph_transformer = GraphTransformerModel(
            input_dim=n_features + self.config['entity_embedding_dim'],
            hidden_dim=self.config['transformer_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        # Forecasting head
        self.forecasting_head = ForecastingHead(
            input_dim=self.config['transformer_dim'],
            sequence_length=self.config['sequence_length'],
            prediction_horizon=self.config['prediction_horizon'],
            output_dim=1  # Single target variable
        )
        
        # Move to device
        self.entity_embeddings = self.entity_embeddings.to(self.device)
        self.graph_transformer = self.graph_transformer.to(self.device)
        self.forecasting_head = self.forecasting_head.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.entity_embeddings.parameters()) +
            list(self.graph_transformer.parameters()) +
            list(self.forecasting_head.parameters()),
            lr=self.config['learning_rate']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
    
    def construct_entity_graph(self, time_series_data: Dict[str, pd.DataFrame],
                              entity_metadata: Optional[pd.DataFrame] = None) -> Data:
        """
        Construct graph connecting related time series entities.
        
        Args:
            time_series_data: Dictionary mapping entity_id to time series DataFrame
            entity_metadata: Optional metadata about entities (features, categories, etc.)
        """
        
        entity_ids = list(time_series_data.keys())
        n_entities = len(entity_ids)
        
        # Create entity index mapping
        entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
        
        # Calculate pairwise similarities for edge construction
        similarity_matrix = self._calculate_entity_similarities(time_series_data)
        
        # Construct edges based on similarity threshold
        edge_list = []
        edge_weights = []
        
        for i in range(n_entities):
            for j in range(i + 1, n_entities):
                similarity = similarity_matrix[i, j]
                
                if similarity > self.config['edge_threshold']:
                    edge_list.extend([[i, j], [j, i]])  # Undirected edges
                    edge_weights.extend([similarity, similarity])
        
        # Create node features
        node_features = self._create_node_features(time_series_data, entity_metadata)
        
        # Create PyTorch Geometric data object
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            entity_ids=entity_ids,
            entity_to_idx=entity_to_idx
        )
        
        self.entity_graph = graph_data
        
        return graph_data
    
    def fit(self, time_series_data: Dict[str, pd.DataFrame],
           target_column: str,
           entity_metadata: Optional[pd.DataFrame] = None,
           validation_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, List[float]]:
        """
        Train Graph Transformer on time series data.
        
        Args:
            time_series_data: Dictionary mapping entity_id to time series DataFrame
            target_column: Name of target column to forecast
            entity_metadata: Optional entity metadata
            validation_data: Optional validation data
        """
        
        # Construct entity graph
        if self.entity_graph is None:
            self.construct_entity_graph(time_series_data, entity_metadata)
        
        # Prepare training data
        train_loader = self._create_data_loader(
            time_series_data, target_column, batch_size=self.config['batch_size']
        )
        
        val_loader = None
        if validation_data:
            val_loader = self._create_data_loader(
                validation_data, target_column, batch_size=self.config['batch_size']
            )
        
        # Training loop
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config['max_epochs']):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                training_history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)
            
            # Update entity graph periodically
            if epoch % self.config['graph_update_frequency'] == 0:
                self._update_entity_graph(time_series_data, entity_metadata)
            
            # Early stopping check
            if self._should_early_stop(training_history, patience=20):
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}" +
                      (f", Val Loss = {val_loss:.4f}" if val_loader else ""))
        
        return training_history
    
    def predict(self, time_series_data: Dict[str, pd.DataFrame],
               target_column: str,
               entity_ids: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Generate forecasts for specified entities.
        
        Args:
            time_series_data: Dictionary mapping entity_id to time series DataFrame
            target_column: Name of target column to forecast
            entity_ids: List of entity IDs to forecast (all if None)
        """
        
        if entity_ids is None:
            entity_ids = list(time_series_data.keys())
        
        # Set models to evaluation mode
        self.entity_embeddings.eval()
        self.graph_transformer.eval()
        self.forecasting_head.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for entity_id in entity_ids:
                if entity_id not in time_series_data:
                    continue
                
                # Prepare input data for entity
                entity_data = self._prepare_entity_data(
                    time_series_data[entity_id], target_column, entity_id
                )
                
                # Get entity subgraph
                subgraph = self._sample_temporal_subgraph(entity_id)
                
                # Forward pass
                entity_prediction = self._predict_entity(entity_data, subgraph)
                predictions[entity_id] = entity_prediction.cpu().numpy()
        
        return predictions
    
    def _calculate_entity_similarities(self, time_series_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate pairwise similarities between entities."""
        
        entity_ids = list(time_series_data.keys())
        n_entities = len(entity_ids)
        similarity_matrix = np.zeros((n_entities, n_entities))
        
        # Extract features for similarity calculation
        entity_features = {}
        for entity_id in entity_ids:
            ts_data = time_series_data[entity_id]
            
            # Calculate basic statistical features
            features = []
            for col in ts_data.select_dtypes(include=[np.number]).columns:
                series = ts_data[col].dropna()
                if len(series) > 0:
                    features.extend([
                        series.mean(),
                        series.std(),
                        series.skew() if len(series) > 2 else 0,
                        series.kurt() if len(series) > 3 else 0
                    ])
            
            entity_features[entity_id] = np.array(features)
        
        # Calculate pairwise cosine similarities
        for i, entity_i in enumerate(entity_ids):
            for j, entity_j in enumerate(entity_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    feat_i = entity_features[entity_i]
                    feat_j = entity_features[entity_j]
                    
                    # Ensure same length
                    min_len = min(len(feat_i), len(feat_j))
                    if min_len > 0:
                        feat_i = feat_i[:min_len]
                        feat_j = feat_j[:min_len]
                        
                        # Cosine similarity
                        norm_i = np.linalg.norm(feat_i)
                        norm_j = np.linalg.norm(feat_j)
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = np.dot(feat_i, feat_j) / (norm_i * norm_j)
                            similarity_matrix[i, j] = max(0, similarity)  # Keep positive similarities
        
        return similarity_matrix
    
    def _create_node_features(self, time_series_data: Dict[str, pd.DataFrame],
                            entity_metadata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Create node features for graph construction."""
        
        entity_ids = list(time_series_data.keys())
        node_features = []
        
        for entity_id in entity_ids:
            features = []
            
            # Time series statistical features
            ts_data = time_series_data[entity_id]
            numeric_cols = ts_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                series = ts_data[col].dropna()
                if len(series) > 0:
                    features.extend([
                        series.mean(),
                        series.std(),
                        series.min(),
                        series.max()
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            # Entity metadata features (if available)
            if entity_metadata is not None and entity_id in entity_metadata.index:
                metadata_features = entity_metadata.loc[entity_id].values
                features.extend(metadata_features)
            
            node_features.append(features)
        
        # Pad features to same length
        max_len = max(len(f) for f in node_features)
        padded_features = []
        
        for features in node_features:
            padded = features + [0] * (max_len - len(features))
            padded_features.append(padded)
        
        return np.array(padded_features, dtype=np.float32)
    
    def _train_epoch(self, data_loader) -> float:
        """Train for one epoch."""
        
        self.entity_embeddings.train()
        self.graph_transformer.train()
        self.forecasting_head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            loss = self._compute_batch_loss(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.entity_embeddings.parameters()) +
                list(self.graph_transformer.parameters()) +
                list(self.forecasting_head.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, data_loader) -> float:
        """Validate for one epoch."""
        
        self.entity_embeddings.eval()
        self.graph_transformer.eval()
        self.forecasting_head.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0


class GraphTransformerModel(nn.Module):
    """Graph Transformer model for processing entity relationships."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int,
                 num_layers: int, dropout: float = 0.1):
        super(GraphTransformerModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False
            ) for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
               batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Graph Transformer."""
        
        # Input projection
        h = self.input_projection(x)
        
        # Apply transformer layers with residual connections
        for transformer_layer, norm_layer in zip(self.transformer_layers, self.norm_layers):
            h_new = transformer_layer(h, edge_index)
            h = norm_layer(h + self.dropout(h_new))  # Residual connection
        
        return h


class ForecastingHead(nn.Module):
    """Forecasting head for generating predictions."""
    
    def __init__(self, input_dim: int, sequence_length: int,
                 prediction_horizon: int, output_dim: int):
        super(ForecastingHead, self).__init__()
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Forecasting layers
        self.forecasting_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, prediction_horizon * output_dim)
        )
    
    def forward(self, node_embeddings: torch.Tensor,
               temporal_sequences: torch.Tensor) -> torch.Tensor:
        """Generate forecasts using node embeddings and temporal sequences."""
        
        # Apply temporal attention to sequences
        attended_sequences, _ = self.temporal_attention(
            temporal_sequences, temporal_sequences, temporal_sequences
        )
        
        # Combine with node embeddings
        # Assuming node_embeddings shape: (batch_size, input_dim)
        # and attended_sequences shape: (batch_size, seq_len, input_dim)
        
        # Pool temporal information
        pooled_temporal = torch.mean(attended_sequences, dim=1)  # (batch_size, input_dim)
        
        # Combine node and temporal information
        combined_features = node_embeddings + pooled_temporal
        
        # Generate predictions
        predictions = self.forecasting_layers(combined_features)
        
        # Reshape to (batch_size, prediction_horizon, output_dim)
        batch_size = predictions.shape[0]
        predictions = predictions.view(batch_size, self.prediction_horizon, -1)
        
        return predictions
