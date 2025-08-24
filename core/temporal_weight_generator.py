import torch
import numpy as np
from typing import List, Dict
from .temporal_cue_extractor import TemporalRelation

class TemporalWeightGenerator:
    """Generates temporal weights based on extracted temporal cues"""
    
    def __init__(self, lambda_cooccurrence: float = 2.0):
        self.lambda_cooccurrence = lambda_cooccurrence
    
    def generate_precedence_weights(self, num_frames: int) -> torch.Tensor:
        """Generate weights emphasizing earlier frames"""
        weights = []
        for i in range(num_frames):
            w_i = 1.5 - (i / (num_frames - 1))
            weights.append(w_i)
        return torch.tensor(weights, dtype=torch.float32)
    
    def generate_subsequence_weights(self, num_frames: int) -> torch.Tensor:
        """Generate weights emphasizing later frames"""
        weights = []
        for i in range(num_frames):
            w_i = 0.5 + (i / (num_frames - 1))
            weights.append(w_i)
        return torch.tensor(weights, dtype=torch.float32)
    
    def generate_cooccurrence_weights(self, num_frames: int) -> torch.Tensor:
        """Generate weights emphasizing central frames"""
        weights = []
        for i in range(num_frames):
            normalized_pos = i / (num_frames - 1)
            w_i = torch.exp(-self.lambda_cooccurrence * abs(normalized_pos - 0.5))
            weights.append(w_i)
        return torch.tensor(weights, dtype=torch.float32)
    
    def generate_uniform_weights(self, num_frames: int) -> torch.Tensor:
        """Generate uniform weights for queries without temporal markers"""
        return torch.ones(num_frames, dtype=torch.float32)
    
    def generate_temporal_weights(self, temporal_cues: Dict, num_frames: int) -> torch.Tensor:
        """Generate temporal weights based on extracted cues"""
        primary_relation = temporal_cues["primary_relation"]
        
        if primary_relation == TemporalRelation.PRECEDENCE:
            weights = self.generate_precedence_weights(num_frames)
        elif primary_relation == TemporalRelation.SUBSEQUENCE:
            weights = self.generate_subsequence_weights(num_frames)
        elif primary_relation == TemporalRelation.CO_OCCURRENCE:
            weights = self.generate_cooccurrence_weights(num_frames)
        else:  # TemporalRelation.NONE
            weights = self.generate_uniform_weights(num_frames)
        
        # Handle multiple temporal relationships
        detected_markers = temporal_cues.get("detected_markers", [])
        if len(detected_markers) > 1:
            combined_weights = weights
            for marker in detected_markers[1:]:  # Skip first as it's already processed
                if marker == TemporalRelation.PRECEDENCE:
                    additional_weights = self.generate_precedence_weights(num_frames)
                elif marker == TemporalRelation.SUBSEQUENCE:
                    additional_weights = self.generate_subsequence_weights(num_frames)
                elif marker == TemporalRelation.CO_OCCURRENCE:
                    additional_weights = self.generate_cooccurrence_weights(num_frames)
                else:
                    continue
                
                # Element-wise multiplication and normalization
                combined_weights = combined_weights * additional_weights
                combined_weights = combined_weights / combined_weights.mean()
            
            weights = combined_weights
        
        return weights