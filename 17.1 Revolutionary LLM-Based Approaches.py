import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel, GPT2Model, T5Model
from dataclasses import dataclass
import math
from abc import ABC, abstractmethod
import json
import logging

@dataclass
class TSFoundationModelConfig:
    """Configuration for Time Series Foundation Models."""
    
    # Model architecture
    model_type: str = "time_llm"  # time_llm, chronos, timesfm, moirai
    backbone_model: str = "gpt2-medium"
    patch_size: int = 16
    max_seq_length: int = 512
    prediction_horizon: int = 96
    
    # Multimodal settings
    enable_text_conditioning: bool = True
    text_embedding_dim: int = 768
    cross_modal_fusion: str = "attention"  # attention, concat, gating
    
    # Training configuration
    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_gradient_norm: float = 1.0
    
    # Tokenization settings
    num_bins: int = 1024
    scaling_method: str = "reversible_instance_norm"
    
    # Foundation model specific
    pretrained_path: Optional[str] = None
    zero_shot_capable: bool = True
    few_shot_examples: int = 5


class TimeSeriesFoundationModel(nn.Module, ABC):
    """Abstract base class for time series foundation models."""
    
    def __init__(self, config: TSFoundationModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def forward(self, 
                time_series: torch.Tensor,
                text_context: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for the foundation model."""
        pass
    
    @abstractmethod
    def zero_shot_forecast(self,
                          input_series: torch.Tensor,
                          horizon: int,
                          context: Optional[str] = None) -> torch.Tensor:
        """Generate zero-shot forecasts."""
        pass


class TimeLLM(TimeSeriesFoundationModel):
    """
    Time-LLM: Reprogramming Large Language Models for Time Series Forecasting.
    Based on the approach from Jin et al. (2023).
    """
    
    def __init__(self, config: TSFoundationModelConfig):
        super().__init__(config)
        
        # Initialize backbone LLM
        self.backbone = self._initialize_backbone()
        
        # Time series processing components
        self.patch_embedding = PatchEmbedding(
            patch_size=config.patch_size,
            d_model=self.backbone.config.hidden_size
        )
        
        # Reprogramming components
        self.text_prototype_layer = TextPrototypeLayer(
            vocab_size=self.backbone.config.vocab_size,
            d_model=self.backbone.config.hidden_size
        )
        
        # Prompt-as-Prefix (PaP) mechanism
        self.pap_layer = PromptAsPrefixLayer(
            d_model=self.backbone.config.hidden_size,
            num_prompts=config.max_seq_length // 4
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            self.backbone.config.hidden_size,
            config.prediction_horizon
        )
        
        # Text encoder for multimodal capability
        if config.enable_text_conditioning:
            self.text_encoder = self._initialize_text_encoder()
            self.cross_modal_fusion = self._build_fusion_module()
    
    def _initialize_backbone(self) -> nn.Module:
        """Initialize the backbone LLM."""
        
        if self.config.backbone_model.startswith("gpt2"):
            backbone = GPT2Model.from_pretrained(self.config.backbone_model)
        elif self.config.backbone_model.startswith("t5"):
            backbone = T5Model.from_pretrained(self.config.backbone_model)
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone_model}")
        
        # Freeze backbone if specified
        if self.config.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        return backbone
    
    def _initialize_text_encoder(self) -> nn.Module:
        """Initialize text encoder for multimodal processing."""
        
        return AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def _build_fusion_module(self) -> nn.Module:
        """Build cross-modal fusion module."""
        
        if self.config.cross_modal_fusion == "attention":
            return CrossModalAttention(
                ts_dim=self.backbone.config.hidden_size,
                text_dim=self.config.text_embedding_dim
            )
        elif self.config.cross_modal_fusion == "gating":
            return GatedFusion(
                ts_dim=self.backbone.config.hidden_size,
                text_dim=self.config.text_embedding_dim
            )
        else:
            return ConcatenateFusion()
    
    def forward(self, 
                time_series: torch.Tensor,
                text_context: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Time-LLM.
        
        Args:
            time_series: Input time series [batch, seq_len, features]
            text_context: Optional text context [batch, text_len]
            attention_mask: Attention mask for sequences
            
        Returns:
            Dictionary containing forecasts and intermediate representations
        """
        
        batch_size, seq_len, n_features = time_series.shape
        
        # Step 1: Patch embedding
        patches = self.patch_embedding(time_series)  # [batch, num_patches, d_model]
        
        # Step 2: Text prototype reprogramming
        reprogrammed_patches = self.text_prototype_layer(patches)
        
        # Step 3: Prompt-as-Prefix enhancement
        enhanced_patches = self.pap_layer(
            reprogrammed_patches, 
            context_info=self._extract_context_info(time_series)
        )
        
        # Step 4: Process text context if available
        if text_context is not None and self.config.enable_text_conditioning:
            text_embeddings = self.text_encoder(**text_context).last_hidden_state
            enhanced_patches = self.cross_modal_fusion(enhanced_patches, text_embeddings)
        
        # Step 5: Pass through frozen LLM backbone
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, enhanced_patches.size(1),
                device=enhanced_patches.device
            )
        
        backbone_outputs = self.backbone(
            inputs_embeds=enhanced_patches,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Step 6: Output projection
        hidden_states = backbone_outputs.last_hidden_state
        
        # Aggregate patches for final prediction
        aggregated_representation = self._aggregate_patch_representations(hidden_states)
        forecasts = self.output_projection(aggregated_representation)
        
        return {
            'forecasts': forecasts,
            'hidden_states': hidden_states,
            'patch_embeddings': patches,
            'reprogrammed_patches': reprogrammed_patches
        }
    
    def zero_shot_forecast(self,
                          input_series: torch.Tensor,
                          horizon: int,
                          context: Optional[str] = None) -> torch.Tensor:
        """Generate zero-shot forecasts without fine-tuning."""
        
        self.eval()
        with torch.no_grad():
            # Prepare text context if provided
            text_context = None
            if context is not None:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                text_context = tokenizer(
                    context, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
            
            # Generate forecast
            outputs = self.forward(
                time_series=input_series.unsqueeze(0),
                text_context=text_context
            )
            
            return outputs['forecasts'][:, :horizon]
    
    def _extract_context_info(self, time_series: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract contextual information from time series for PaP layer."""
        
        # Calculate basic statistics
        mean_val = torch.mean(time_series, dim=1, keepdim=True)
        std_val = torch.std(time_series, dim=1, keepdim=True)
        trend = time_series[:, -1:] - time_series[:, :1]
        
        return {
            'mean': mean_val,
            'std': std_val,
            'trend': trend
        }
    
    def _aggregate_patch_representations(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Aggregate patch representations for final prediction."""
        
        # Use attention-based aggregation
        attention_weights = torch.softmax(
            torch.mean(hidden_states, dim=-1), dim=-1
        ).unsqueeze(-1)
        
        aggregated = torch.sum(hidden_states * attention_weights, dim=1)
        return aggregated


class PatchEmbedding(nn.Module):
    """Convert time series to patches and embed them."""
    
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1000, d_model) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patches.
        
        Args:
            x: Time series [batch, seq_len, features]
            
        Returns:
            Patch embeddings [batch, num_patches, d_model]
        """
        
        batch_size, seq_len, n_features = x.shape
        
        # Flatten features into sequence dimension
        x = x.reshape(batch_size, seq_len * n_features)
        
        # Create patches
        num_patches = (seq_len * n_features) // self.patch_size
        if (seq_len * n_features) % self.patch_size != 0:
            # Pad to make divisible by patch_size
            pad_len = self.patch_size - ((seq_len * n_features) % self.patch_size)
            x = torch.cat([x, torch.zeros(batch_size, pad_len, device=x.device)], dim=1)
            num_patches += 1
        
        # Reshape into patches
        patches = x.reshape(batch_size, num_patches, self.patch_size)
        
        # Project patches
        patch_embeddings = self.projection(patches)
        
        # Add position embeddings
        pos_embeddings = self.position_embedding[:num_patches].unsqueeze(0)
        patch_embeddings += pos_embeddings
        
        return patch_embeddings


class TextPrototypeLayer(nn.Module):
    """Reprogram time series patches using text prototypes."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Text prototypes (learnable)
        self.text_prototypes = nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.02
        )
        
        # Reprogramming transformation
        self.reprogramming_layer = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reprogram patches using text prototypes.
        
        Args:
            patch_embeddings: Patch embeddings [batch, num_patches, d_model]
            
        Returns:
            Reprogrammed embeddings [batch, num_patches, d_model]
        """
        
        # Compute similarity between patches and text prototypes
        similarities = torch.matmul(
            patch_embeddings, 
            self.text_prototypes.transpose(0, 1)
        )  # [batch, num_patches, vocab_size]
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(similarities, dim=-1)
        
        # Weighted combination of text prototypes
        reprogrammed = torch.matmul(
            attention_weights, self.text_prototypes
        )  # [batch, num_patches, d_model]
        
        # Apply transformation and residual connection
        transformed = self.reprogramming_layer(reprogrammed)
        output = self.layer_norm(patch_embeddings + transformed)
        
        return output


class PromptAsPrefixLayer(nn.Module):
    """Prompt-as-Prefix mechanism for enhanced context understanding."""
    
    def __init__(self, d_model: int, num_prompts: int):
        super().__init__()
        self.d_model = d_model
        self.num_prompts = num_prompts
        
        # Learnable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, d_model) * 0.02
        )
        
        # Context transformation layers
        self.context_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_prompts * d_model)
        )
        
        self.gating_layer = nn.Sequential(
            nn.Linear(d_model, num_prompts),
            nn.Sigmoid()
        )
    
    def forward(self, 
                patch_embeddings: torch.Tensor,
                context_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply Prompt-as-Prefix mechanism.
        
        Args:
            patch_embeddings: Input patch embeddings
            context_info: Contextual information from time series
            
        Returns:
            Enhanced embeddings with prompt prefix
        """
        
        batch_size = patch_embeddings.size(0)
        
        # Create context vector from statistical information
        context_vector = torch.cat([
            context_info['mean'].squeeze(-1),
            context_info['std'].squeeze(-1),
            context_info['trend'].squeeze(-1)
        ], dim=-1)
        
        # Transform context to prompt space
        context_prompts = self.context_transform(context_vector)
        context_prompts = context_prompts.reshape(
            batch_size, self.num_prompts, self.d_model
        )
        
        # Generate gating weights
        gates = self.gating_layer(context_vector).unsqueeze(-1)
        
        # Combine learnable prompts with context-specific prompts
        prompts = self.prompt_embeddings.unsqueeze(0) + gates * context_prompts
        
        # Concatenate prompts as prefix
        enhanced_embeddings = torch.cat([prompts, patch_embeddings], dim=1)
        
        return enhanced_embeddings


class CrossModalAttention(nn.Module):
    """Cross-modal attention for time series and text fusion."""
    
    def __init__(self, ts_dim: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.ts_dim = ts_dim
        self.text_dim = text_dim
        self.num_heads = num_heads
        
        # Project text to time series dimension
        self.text_projection = nn.Linear(text_dim, ts_dim)
        
        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=ts_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(ts_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                ts_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            ts_embeddings: Time series embeddings [batch, seq_len, ts_dim]
            text_embeddings: Text embeddings [batch, text_len, text_dim]
            
        Returns:
            Fused embeddings [batch, seq_len, ts_dim]
        """
        
        # Project text embeddings
        text_projected = self.text_projection(text_embeddings)
        
        # Cross attention: time series attends to text
        attended_ts, attention_weights = self.cross_attention(
            query=ts_embeddings,
            key=text_projected,
            value=text_projected
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(ts_embeddings + self.dropout(attended_ts))
        
        return output


class GatedFusion(nn.Module):
    """Gated fusion mechanism for multimodal integration."""
    
    def __init__(self, ts_dim: int, text_dim: int):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, ts_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(ts_dim + ts_dim, ts_dim),
            nn.Sigmoid()
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(ts_dim + ts_dim, ts_dim),
            nn.ReLU(),
            nn.Linear(ts_dim, ts_dim)
        )
    
    def forward(self, 
                ts_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply gated fusion."""
        
        # Project text and aggregate
        text_projected = self.text_projection(text_embeddings)
        text_aggregated = torch.mean(text_projected, dim=1, keepdim=True)
        text_expanded = text_aggregated.expand_as(ts_embeddings)
        
        # Compute gates
        combined = torch.cat([ts_embeddings, text_expanded], dim=-1)
        gates = self.gate_network(combined)
        
        # Fused representation
        fused = self.fusion_network(combined)
        
        # Gated combination
        output = gates * fused + (1 - gates) * ts_embeddings
        
        return output


class ConcatenateFusion(nn.Module):
    """Simple concatenation-based fusion."""
    
    def forward(self, 
                ts_embeddings: torch.Tensor,
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply concatenate fusion."""
        return ts_embeddings  # Placeholder for simplicity


class ChronosFoundationModel(TimeSeriesFoundationModel):
    """
    Chronos: Learning the Language of Time Series.
    Implementation based on Amazon's Chronos model.
    """
    
    def __init__(self, config: TSFoundationModelConfig):
        super().__init__(config)
        
        self.tokenizer = TimeSeriesTokenizer(
            num_bins=config.num_bins,
            scaling_method=config.scaling_method
        )
        
        # Use T5 as backbone for encoder-decoder architecture
        self.backbone = T5Model.from_pretrained("t5-small")
        
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Output head for time series generation
        self.lm_head = nn.Linear(
            self.backbone.config.d_model,
            config.num_bins
        )
    
    def forward(self, 
                time_series: torch.Tensor,
                text_context: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for Chronos model."""
        
        # Tokenize time series
        input_tokens = self.tokenizer.encode(time_series)
        
        # Create decoder input (shifted tokens for autoregressive generation)
        decoder_input_ids = torch.cat([
            torch.zeros(input_tokens.size(0), 1, dtype=torch.long, device=input_tokens.device),
            input_tokens[:, :-1]
        ], dim=1)
        
        # Forward through T5
        outputs = self.backbone(
            input_ids=input_tokens,
            decoder_input_ids=decoder_input_ids
        )
        
        # Generate logits for next token prediction
        logits = self.lm_head(outputs.last_hidden_state)
        
        return {
            'logits': logits,
            'tokens': input_tokens,
            'decoder_outputs': outputs
        }
    
    def zero_shot_forecast(self,
                          input_series: torch.Tensor,
                          horizon: int,
                          context: Optional[str] = None) -> torch.Tensor:
        """Generate zero-shot forecasts using autoregressive generation."""
        
        self.eval()
        with torch.no_grad():
            # Tokenize input
            input_tokens = self.tokenizer.encode(input_series)
            
            # Autoregressive generation
            generated_tokens = self._generate_autoregressive(
                input_tokens, 
                max_length=horizon
            )
            
            # Decode back to time series
            forecasts = self.tokenizer.decode(generated_tokens)
            
            return forecasts
    
    def _generate_autoregressive(self, 
                               input_tokens: torch.Tensor,
                               max_length: int,
                               temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive generation of time series tokens."""
        
        batch_size = input_tokens.size(0)
        device = input_tokens.device
        
        # Initialize with input tokens
        generated = input_tokens.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.backbone(
                input_ids=input_tokens,
                decoder_input_ids=generated
            )
            
            # Get logits for next token
            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])
            
            # Sample next token
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 
                    num_samples=1
                )
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class TimeSeriesTokenizer:
    """Tokenizer for converting time series to discrete tokens."""
    
    def __init__(self, num_bins: int = 1024, scaling_method: str = "reversible_instance_norm"):
        self.num_bins = num_bins
        self.scaling_method = scaling_method
        
    def encode(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to discrete tokens.
        
        Args:
            time_series: Input time series [batch, seq_len, features]
            
        Returns:
            Token sequence [batch, seq_len * features]
        """
        
        batch_size, seq_len, n_features = time_series.shape
        
        # Flatten features
        flattened = time_series.reshape(batch_size, -1)
        
        # Apply scaling
        if self.scaling_method == "reversible_instance_norm":
            # Scale by mean absolute value
            scale_factor = torch.mean(torch.abs(flattened), dim=1, keepdim=True)
            scale_factor = torch.clamp(scale_factor, min=1e-8)
            scaled = flattened / scale_factor
        else:
            scaled = flattened
        
        # Quantize to bins
        # Map to [0, num_bins-1]
        min_val = torch.min(scaled, dim=1, keepdim=True)[0]
        max_val = torch.max(scaled, dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = torch.clamp(range_val, min=1e-8)
        
        normalized = (scaled - min_val) / range_val
        tokens = torch.clamp(
            (normalized * (self.num_bins - 1)).long(),
            0, self.num_bins - 1
        )
        
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens back to time series.
        
        Args:
            tokens: Token sequence [batch, seq_len]
            
        Returns:
            Time series [batch, seq_len, 1]
        """
        
        # Convert tokens to normalized values
        normalized = tokens.float() / (self.num_bins - 1)
        
        # Note: In practice, you'd need to store scaling parameters
        # during encoding to properly reverse the transformation
        # This is a simplified version
        
        return normalized.unsqueeze(-1)


class MultimodalTimeSeriesFoundationModel(TimeSeriesFoundationModel):
    """
    Advanced multimodal foundation model integrating multiple data modalities.
    Based on ChronoSteer and other multimodal approaches.
    """
    
    def __init__(self, config: TSFoundationModelConfig):
        super().__init__(config)
        
        # Time series foundation model
        self.ts_foundation_model = ChronosFoundationModel(config)
        
        # Text processing components
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_projection = nn.Linear(384, config.text_embedding_dim)
        
        # Multimodal fusion
        self.fusion_layer = MultimodalFusionLayer(
            ts_dim=self.ts_foundation_model.backbone.config.d_model,
            text_dim=config.text_embedding_dim,
            fusion_dim=512
        )
        
        # Steering mechanism
        self.steering_network = SteeringNetwork(
            input_dim=512,
            output_dim=config.prediction_horizon
        )
    
    def forward(self, 
                time_series: torch.Tensor,
                text_context: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal foundation model."""
        
        # Get time series representations
        ts_outputs = self.ts_foundation_model(time_series)
        ts_representations = ts_outputs['decoder_outputs'].last_hidden_state
        
        # Process text context if available
        if text_context is not None:
            text_embeddings = self.text_encoder(**text_context).last_hidden_state
            text_features = self.text_projection(text_embeddings)
            
            # Multimodal fusion
            fused_representations = self.fusion_layer(ts_representations, text_features)
        else:
            fused_representations = ts_representations
        
        # Generate steered forecasts
        steered_forecasts = self.steering_network(fused_representations)
        
        return {
            'forecasts': steered_forecasts,
            'ts_representations': ts_representations,
            'fused_representations': fused_representations,
            'base_outputs': ts_outputs
        }
    
    def zero_shot_forecast(self,
                          input_series: torch.Tensor,
                          horizon: int,
                          context: Optional[str] = None) -> torch.Tensor:
        """Generate zero-shot forecasts with text steering."""
        
        self.eval()
        with torch.no_grad():
            # Prepare text context
            text_context = None
            if context is not None:
                tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                text_context = tokenizer(
                    context,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
            
            # Generate forecast
            outputs = self.forward(
                time_series=input_series.unsqueeze(0),
                text_context=text_context
            )
            
            return outputs['forecasts'][:, :horizon]


class MultimodalFusionLayer(nn.Module):
    """Advanced multimodal fusion for time series and text."""
    
    def __init__(self, ts_dim: int, text_dim: int, fusion_dim: int):
        super().__init__()
        
        self.ts_projection = nn.Linear(ts_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        
        # Cross-attention layers
        self.ts_to_text_attention = nn.MultiheadAttention(
            fusion_dim, num_heads=8, batch_first=True
        )
        self.text_to_ts_attention = nn.MultiheadAttention(
            fusion_dim, num_heads=8, batch_first=True
        )
        
        # Fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
    
    def forward(self, 
                ts_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Apply multimodal fusion."""
        
        # Project to common space
        ts_proj = self.ts_projection(ts_features)
        text_proj = self.text_projection(text_features)
        
        # Cross-attention
        ts_attended, _ = self.ts_to_text_attention(ts_proj, text_proj, text_proj)
        text_attended, _ = self.text_to_ts_attention(text_proj, ts_proj, ts_proj)
        
        # Aggregate text features
        text_aggregated = torch.mean(text_attended, dim=1, keepdim=True)
        text_expanded = text_aggregated.expand_as(ts_attended)
        
        # Combine features
        combined = torch.cat([ts_attended, text_expanded], dim=-1)
        fused = self.fusion_network(combined)
        
        return fused


class SteeringNetwork(nn.Module):
    """Network for steering forecasts based on multimodal inputs."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.steering_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim)
        )
        
        # Attention for temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            input_dim, num_heads=8, batch_first=True
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Generate steered forecasts."""
        
        # Temporal attention for sequence aggregation
        attended_features, _ = self.temporal_attention(
            fused_features, fused_features, fused_features
        )
        
        # Aggregate temporal dimension
        aggregated = torch.mean(attended_features, dim=1)
        
        # Generate forecasts
        forecasts = self.steering_layers(aggregated)
        
        return forecasts


# Example usage and training
class FoundationModelTrainer:
    """Trainer for time series foundation models."""
    
    def __init__(self, model: TimeSeriesFoundationModel, config: TSFoundationModelConfig):
        self.model = model
        self.config = config
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = self.model(**batch)
            
            # Calculate loss based on model type
            if 'logits' in outputs:
                # For Chronos-style models
                loss = nn.CrossEntropyLoss()(
                    outputs['logits'].reshape(-1, outputs['logits'].size(-1)),
                    batch['target_tokens'].reshape(-1)
                )
                
            else:
                # For direct regression models
                loss = nn.MSELoss()(outputs['forecasts'], batch['targets'])
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_gradient_norm
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return {'loss': loss.item()}
    
    def evaluate_zero_shot(self, test_datasets: List[Dict]) -> Dict[str, float]:
        """Evaluate zero-shot performance across multiple datasets."""
        
        self.model.eval()
        results = {}
        
        for dataset_name, dataset in test_datasets:
            dataset_results = []
            
            for batch in dataset:
                with torch.no_grad():
                    forecasts = self.model.zero_shot_forecast(
                        input_series=batch['input_series'],
                        horizon=batch['horizon'],
                        context=batch.get('context')
                    )
                    
                    # Calculate metrics
                    mae = torch.mean(torch.abs(forecasts - batch['targets']))
                    mse = torch.mean((forecasts - batch['targets']) ** 2)
                    
                    dataset_results.append({
                        'mae': mae.item(),
                        'mse': mse.item()
                    })
            
            # Average results for this dataset
            results[dataset_name] = {
                'mae': np.mean([r['mae'] for r in dataset_results]),
                'mse': np.mean([r['mse'] for r in dataset_results])
            }
        
        return results


# Example usage
if __name__ == "__main__":
    # Configuration
    config = TSFoundationModelConfig(
        model_type="time_llm",
        backbone_model="gpt2-medium",
        patch_size=16,
        max_seq_length=512,
        prediction_horizon=96,
        enable_text_conditioning=True
    )
    
    # Initialize model
    model = TimeLLM(config)
    
    # Example input
    batch_size, seq_len, n_features = 32, 336, 7
    time_series = torch.randn(batch_size, seq_len, n_features)
    
    # Text context example
    text_context = {
        'input_ids': torch.randint(0, 1000, (batch_size, 50)),
        'attention_mask': torch.ones(batch_size, 50)
    }
    
    # Forward pass
    outputs = model(time_series, text_context)
    print(f"Forecast shape: {outputs['forecasts'].shape}")
    
    # Zero-shot inference
    zero_shot_forecast = model.zero_shot_forecast(
        input_series=time_series[:1],
        horizon=24,
        context="Economic indicators suggest moderate growth in the next quarter."
    )
    print(f"Zero-shot forecast shape: {zero_shot_forecast.shape}")
