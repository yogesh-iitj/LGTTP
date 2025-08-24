import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from .temporal_cue_extractor import TemporalCueExtractor
from .temporal_weight_generator import TemporalWeightGenerator

class LGTTPruner:
    """Main LGTTP framework for language-guided temporal token pruning"""
    
    def __init__(self, 
                 alpha: float = 0.65,  # Overall pruning rate
                 min_token_ratio: float = 0.1,  # Minimum tokens to retain per frame
                 a: float = 1.0,  # Cosine similarity weight
                 b: float = 0.0):  # Cosine similarity bias
        
        self.alpha = alpha
        self.min_token_ratio = min_token_ratio
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        
        # Initialize components
        self.temporal_cue_extractor = TemporalCueExtractor()
        self.temporal_weight_generator = TemporalWeightGenerator()
    
    def compute_base_relevance(self, 
                             frame_embeddings: torch.Tensor, 
                             query_embedding: torch.Tensor) -> torch.Tensor:
        """Compute base relevance scores using cosine similarity"""
        # Normalize embeddings
        frame_embeddings_norm = F.normalize(frame_embeddings, dim=-1)
        query_embedding_norm = F.normalize(query_embedding, dim=-1)
        
        # Compute cosine similarity
        cos_sim = torch.matmul(frame_embeddings_norm, query_embedding_norm.T)
        
        # Apply learned transformation
        relevance = self.a * cos_sim + self.b
        
        return relevance.squeeze(-1)  # [batch_size, num_frames]
    
    def compute_temporal_relevance(self, 
                                 base_relevance: torch.Tensor, 
                                 temporal_weights: torch.Tensor) -> torch.Tensor:
        """Apply temporal weights to base relevance scores"""
        return base_relevance * temporal_weights
    
    def compute_pruning_rates(self, temporal_relevance: torch.Tensor) -> torch.Tensor:
        """Convert temporal relevance scores to frame-specific pruning rates"""
        batch_size, num_frames = temporal_relevance.shape
        
        # Apply softmax to get probability distribution
        relevance_probs = F.softmax(temporal_relevance, dim=-1)
        
        # Scale by alpha and number of frames to get pruning rates
        pruning_rates = self.alpha * num_frames * relevance_probs
        
        return pruning_rates
    
    def apply_soft_pruning(self, 
                          pruning_rates: torch.Tensor, 
                          total_tokens_per_frame: int) -> torch.Tensor:
        """Apply soft selection to determine tokens to retain per frame"""
        # Compute tokens to retain (1 - pruning_rate)
        retention_rates = 1.0 - pruning_rates
        tokens_to_retain = retention_rates * total_tokens_per_frame
        
        # Apply minimum token threshold
        min_tokens = int(self.min_token_ratio * total_tokens_per_frame)
        tokens_to_retain = torch.clamp(tokens_to_retain, min=min_tokens)
        
        # Round to integers
        tokens_to_retain = torch.ceil(tokens_to_retain).long()
        
        return tokens_to_retain
    
    def prune_tokens(self, 
                    query: str,
                    frame_embeddings: torch.Tensor,
                    query_embedding: torch.Tensor,
                    frame_tokens: torch.Tensor,
                    total_tokens_per_frame: int) -> Tuple[torch.Tensor, Dict]:
        """
        Main pruning method
        
        Args:
            query: Natural language query
            frame_embeddings: [batch_size, num_frames, embedding_dim]
            query_embedding: [batch_size, embedding_dim]  
            frame_tokens: [batch_size, num_frames, tokens_per_frame, token_dim]
            total_tokens_per_frame: Number of tokens per frame before pruning
            
        Returns:
            pruned_tokens: Pruned token tensor
            pruning_info: Dictionary with pruning statistics
        """
        batch_size, num_frames, embedding_dim = frame_embeddings.shape
        
        # Step 1: Extract temporal cues
        temporal_cues = self.temporal_cue_extractor.extract_temporal_cues(query)
        
        # Step 2: Generate temporal weights
        temporal_weights = self.temporal_weight_generator.generate_temporal_weights(
            temporal_cues, num_frames)
        temporal_weights = temporal_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Step 3: Compute base relevance
        base_relevance = self.compute_base_relevance(frame_embeddings, query_embedding)
        
        # Step 4: Apply temporal weighting
        temporal_relevance = self.compute_temporal_relevance(base_relevance, temporal_weights)
        
        # Step 5: Compute pruning rates
        pruning_rates = self.compute_pruning_rates(temporal_relevance)
        
        # Step 6: Apply soft pruning
        tokens_to_retain = self.apply_soft_pruning(pruning_rates, total_tokens_per_frame)
        
        # Step 7: Actually prune tokens based on importance scores
        pruned_tokens_list = []
        
        for batch_idx in range(batch_size):
            batch_pruned_tokens = []
            for frame_idx in range(num_frames):
                frame_token_embeddings = frame_tokens[batch_idx, frame_idx]  # [tokens_per_frame, token_dim]
                num_tokens_to_keep = tokens_to_retain[batch_idx, frame_idx].item()
                
                # Compute token importance (using attention or similarity)
                query_emb = query_embedding[batch_idx].unsqueeze(0)  # [1, embedding_dim]
                token_similarities = F.cosine_similarity(
                    frame_token_embeddings, query_emb, dim=-1)  # [tokens_per_frame]
                
                # Select top-k tokens
                _, top_indices = torch.topk(token_similarities, num_tokens_to_keep)
                selected_tokens = frame_token_embeddings[top_indices]
                
                batch_pruned_tokens.append(selected_tokens)
            
            pruned_tokens_list.append(batch_pruned_tokens)
        
        # Package pruning information
        pruning_info = {
            "temporal_cues": temporal_cues,
            "temporal_weights": temporal_weights,
            "pruning_rates": pruning_rates,
            "tokens_retained": tokens_to_retain,
            "total_tokens_before": batch_size * num_frames * total_tokens_per_frame,
            "total_tokens_after": tokens_to_retain.sum().item(),
            "compression_ratio": 1.0 - (tokens_to_retain.sum().item() / 
                                      (batch_size * num_frames * total_tokens_per_frame))
        }
        
        return pruned_tokens_list, pruning_info