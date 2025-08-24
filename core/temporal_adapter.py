import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TemporalAdapter(nn.Module):
    """Temporal adapter for models without explicit temporal awareness"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 max_frames: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_frames = max_frames
        
        # Temporal position embeddings
        self.temporal_embed = nn.Embedding(max_frames, embedding_dim)
        
        # Two-layer MLP for temporal feature transformation
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, frame_embeddings: torch.Tensor, frame_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frame_embeddings: [batch_size, num_frames, embedding_dim]
            frame_indices: [batch_size, num_frames] or None (will use sequential indices)
        
        Returns:
            Enhanced embeddings with temporal information
        """
        batch_size, num_frames, embedding_dim = frame_embeddings.shape
        
        # Generate frame indices if not provided
        if frame_indices is None:
            frame_indices = torch.arange(num_frames, device=frame_embeddings.device)
            frame_indices = frame_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Get temporal embeddings
        temporal_embeddings = self.temporal_embed(frame_indices)
        
        # Transform temporal embeddings
        temporal_features = self.temporal_mlp(temporal_embeddings)
        
        # Add temporal information with learned scaling
        enhanced_embeddings = frame_embeddings + self.scale * temporal_features
        
        return enhanced_embeddings

class PositionalTemporalAdapter(nn.Module):
    """Lightweight temporal adapter using positional embeddings"""
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        
        # Learned linear transformation for normalized positions
        self.position_projection = nn.Linear(1, embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.position_projection.weight)
        nn.init.zeros_(self.position_projection.bias)
    
    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_embeddings: [batch_size, num_frames, embedding_dim]
        
        Returns:
            Enhanced embeddings with positional temporal information
        """
        batch_size, num_frames, embedding_dim = frame_embeddings.shape
        
        # Generate normalized positions
        positions = torch.linspace(0, 1, num_frames, device=frame_embeddings.device)
        positions = positions.unsqueeze(0).unsqueeze(-1)  # [1, num_frames, 1]
        positions = positions.expand(batch_size, -1, -1)  # [batch_size, num_frames, 1]
        
        # Project positions to embedding space
        temporal_features = self.position_projection(positions)  # [batch_size, num_frames, embedding_dim]
        
        # Add temporal information
        enhanced_embeddings = frame_embeddings + temporal_features
        
        return enhanced_embeddings