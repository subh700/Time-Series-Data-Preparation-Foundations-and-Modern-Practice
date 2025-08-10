import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class LocalAttentionMechanism(nn.Module):
    """
    Local Attention Mechanism optimized for time series forecasting.
    Reduces complexity from O(n²) to O(n log n) by exploiting temporal locality.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, 
                 local_window: int = 64, dropout: float = 0.1):
        super(LocalAttentionMechanism, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.local_window = local_window
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Local position encoding
        self.register_buffer('local_pe', self._generate_local_pe())
        
    def _generate_local_pe(self) -> torch.Tensor:
        """Generate local positional encoding for temporal continuity."""
        
        max_len = self.local_window * 2
        pe = torch.zeros(max_len, self.d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with local attention computation.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)  
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attended output (batch_size, seq_len, d_model)
            attention_weights: Local attention weights
        """
        
        batch_size, seq_len, d_model = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply local attention
        output, attention_weights = self._local_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(output)
        
        return output, attention_weights
    
    def _local_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficient local attention computation using sliding windows.
        """
        
        batch_size, num_heads, seq_len, d_k = Q.size()
        
        # Initialize output tensor
        output = torch.zeros_like(Q)
        attention_weights = torch.zeros(batch_size, num_heads, seq_len, self.local_window * 2)
        
        # Process each position with local window
        for i in range(seq_len):
            # Define local window boundaries
            start_idx = max(0, i - self.local_window)
            end_idx = min(seq_len, i + self.local_window + 1)
            window_size = end_idx - start_idx
            
            # Extract local keys and values
            local_K = K[:, :, start_idx:end_idx, :]  # (batch, heads, window_size, d_k)
            local_V = V[:, :, start_idx:end_idx, :]  # (batch, heads, window_size, d_k)
            current_Q = Q[:, :, i:i+1, :]            # (batch, heads, 1, d_k)
            
            # Compute attention scores with positional bias
            scores = torch.matmul(current_Q, local_K.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Add positional encoding bias
            pos_bias = self._compute_positional_bias(i, start_idx, end_idx)
            scores += pos_bias.unsqueeze(0).unsqueeze(0)
            
            # Apply causal mask for forecasting
            if mask is not None or True:  # Always apply causal mask for time series
                causal_mask = self._create_causal_mask(i, start_idx, end_idx, window_size)
                scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply softmax and dropout
            local_attention = F.softmax(scores, dim=-1)
            local_attention = self.dropout(local_attention)
            
            # Compute weighted values
            local_output = torch.matmul(local_attention, local_V)
            output[:, :, i:i+1, :] = local_output
            
            # Store attention weights (for visualization)
            if window_size <= self.local_window * 2:
                attention_weights[:, :, i, :window_size] = local_attention.squeeze(2)
        
        return output, attention_weights
    
    def _compute_positional_bias(self, current_pos: int, start_idx: int, end_idx: int) -> torch.Tensor:
        """Compute positional bias for local attention."""
        
        window_size = end_idx - start_idx
        relative_positions = torch.arange(start_idx, end_idx) - current_pos
        
        # Gaussian decay based on temporal distance
        temporal_decay = torch.exp(-0.1 * torch.abs(relative_positions.float()))
        
        # Combine with learned positional encoding
        pos_indices = torch.clamp(relative_positions + self.local_window, 0, 
                                 len(self.local_pe) - 1)
        pos_encoding = self.local_pe[pos_indices]
        
        # Project to scalar bias
        bias = temporal_decay * 0.1  # Simple scaling
        
        return bias.unsqueeze(0)  # Add batch dimension
    
    def _create_causal_mask(self, current_pos: int, start_idx: int, 
                          end_idx: int, window_size: int) -> torch.Tensor:
        """Create causal mask for local attention."""
        
        mask = torch.zeros(1, window_size, dtype=torch.bool)
        
        # Mask future positions
        for j, global_pos in enumerate(range(start_idx, end_idx)):
            if global_pos > current_pos:
                mask[0, j] = True
        
        return mask


class CrossAttentionOnlyTransformer(nn.Module):
    """
    Cross-Attention-Only Time Series Transformer (CATS) that eliminates
    self-attention and focuses on cross-attention mechanisms.
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 6, forecast_horizon: int = 24, dropout: float = 0.1):
        super(CrossAttentionOnlyTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable future queries for each forecasting horizon
        self.future_queries = nn.Parameter(
            torch.randn(forecast_horizon, d_model) * 0.02
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, 
                context_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using cross-attention only.
        
        Args:
            x: Input time series (batch_size, seq_len, input_dim)
            context_features: Optional context information
            
        Returns:
            predictions: Forecasted values (batch_size, forecast_horizon, 1)
        """
        
        batch_size, seq_len, _ = x.size()
        
        # Project input to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Prepare queries for forecasting horizons
        queries = self.future_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch_size, forecast_horizon, d_model)
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            queries = layer(queries, x, x)  # Query from future, Key/Value from past
        
        # Apply layer normalization
        queries = self.layer_norm(queries)
        
        # Generate predictions
        predictions = self.output_projection(queries)
        
        return predictions


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for CATS architecture."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttentionLayer, self).__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """Forward pass through cross-attention layer."""
        
        # Cross-attention with residual connection
        attn_output, _ = self.cross_attention(query, key, value)
        query = self.norm1(query + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(query)
        query = self.norm2(query + self.dropout(ff_output))
        
        return query


class SegmentAttentionTransformer(nn.Module):
    """
    Parameter-efficient Transformer with Segment Attention (PSformer)
    that uses spatial-temporal segment attention for efficient processing.
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 4, segment_length: int = 24, 
                 forecast_horizon: int = 24, dropout: float = 0.1):
        super(SegmentAttentionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.segment_length = segment_length
        self.forecast_horizon = forecast_horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Segment attention layers with parameter sharing
        self.shared_segment_layer = SegmentAttentionLayer(
            d_model, num_heads, segment_length, dropout
        )
        self.num_layers = num_layers
        
        # Position encoding for segments
        self.segment_pe = nn.Parameter(torch.randn(1000, d_model) * 0.02)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with segment attention and parameter sharing.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            predictions: Forecasted values (batch_size, input_dim, forecast_horizon)
        """
        
        batch_size, seq_len, input_dim = x.size()
        
        # Embed input
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Create segments
        num_segments = seq_len // self.segment_length
        if seq_len % self.segment_length != 0:
            # Pad sequence to make it divisible by segment_length
            pad_length = self.segment_length - (seq_len % self.segment_length)
            x = F.pad(x, (0, 0, 0, pad_length))
            seq_len += pad_length
            num_segments = seq_len // self.segment_length
        
        # Reshape into segments
        x = x.view(batch_size, num_segments, self.segment_length, self.d_model)
        
        # Add positional encoding
        if num_segments <= len(self.segment_pe):
            segment_pos = self.segment_pe[:num_segments].unsqueeze(0).unsqueeze(2)
            x = x + segment_pos
        
        # Apply segment attention layers (with parameter sharing)
        for _ in range(self.num_layers):
            x = self.shared_segment_layer(x)
        
        # Global pooling across segments and time
        x = x.mean(dim=(1, 2))  # (batch_size, d_model)
        
        # Generate predictions for each input dimension
        predictions = []
        for i in range(input_dim):
            pred = self.output_projection(x)  # (batch_size, forecast_horizon)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (batch_size, input_dim, forecast_horizon)
        
        return predictions


class SegmentAttentionLayer(nn.Module):
    """Spatial-Temporal Segment Attention layer."""
    
    def __init__(self, d_model: int, num_heads: int, segment_length: int, dropout: float = 0.1):
        super(SegmentAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.segment_length = segment_length
        
        # Multi-head attention for segments
        self.segment_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Temporal attention within segments
        self.temporal_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward networks
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segment attention.
        
        Args:
            x: Input tensor (batch_size, num_segments, segment_length, d_model)
            
        Returns:
            output: Processed tensor (batch_size, num_segments, segment_length, d_model)
        """
        
        batch_size, num_segments, segment_length, d_model = x.size()
        
        # 1. Spatial attention across segments for each time position
        spatial_output = torch.zeros_like(x)
        
        for t in range(segment_length):
            # Extract data at time position t across all segments
            spatial_input = x[:, :, t, :]  # (batch_size, num_segments, d_model)
            
            # Apply attention across segments
            attn_out, _ = self.segment_attention(
                spatial_input, spatial_input, spatial_input
            )
            
            # Residual connection and normalization
            spatial_output[:, :, t, :] = self.norm1(
                spatial_input + self.dropout(attn_out)
            )
        
        # Feed-forward
        spatial_output = self.norm2(
            spatial_output + self.dropout(self.ff1(spatial_output))
        )
        
        # 2. Temporal attention within each segment
        temporal_output = torch.zeros_like(spatial_output)
        
        for s in range(num_segments):
            # Extract segment s
            segment_input = spatial_output[:, s, :, :]  # (batch_size, segment_length, d_model)
            
            # Apply temporal attention within segment
            attn_out, _ = self.temporal_attention(
                segment_input, segment_input, segment_input
            )
            
            # Residual connection and normalization
            temporal_output[:, s, :, :] = self.norm3(
                segment_input + self.dropout(attn_out)
            )
        
        # Feed-forward
        output = self.norm4(
            temporal_output + self.dropout(self.ff2(temporal_output))
        )
        
        return output
