import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional
from enum import Enum

class TemporalRelation(Enum):
    PRECEDENCE = "precedence"      # before, prior to
    SUBSEQUENCE = "subsequence"    # after, following
    CO_OCCURRENCE = "co_occurrence"  # during, while
    NONE = "none"

class TemporalCueExtractor:
    """Extracts temporal cues from natural language queries"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters
        for param in self.bert_model.parameters():
            param.requires_grad = False
            
        # Temporal marker classifier (2-layer MLP)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(TemporalRelation)),
            nn.Softmax(dim=-1)
        )
        
        # Temporal marker patterns
        self.temporal_patterns = {
            TemporalRelation.PRECEDENCE: [
                r'\bbefore\b', r'\bprior to\b', r'\bearlier\b', 
                r'\bpreviously\b', r'\bfirst\b', r'\binitially\b'
            ],
            TemporalRelation.SUBSEQUENCE: [
                r'\bafter\b', r'\bfollowing\b', r'\bthen\b',
                r'\blater\b', r'\bnext\b', r'\bsubsequently\b'
            ],
            TemporalRelation.CO_OCCURRENCE: [
                r'\bduring\b', r'\bwhile\b', r'\bwhen\b',
                r'\bas\b', r'\bmeanwhile\b', r'\bsimultaneously\b'
            ]
        }
    
    def extract_temporal_markers(self, query: str) -> List[TemporalRelation]:
        """Extract temporal markers using pattern matching"""
        markers = []
        query_lower = query.lower()
        
        for relation, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    markers.append(relation)
                    break
        
        return markers if markers else [TemporalRelation.NONE]
    
    def classify_temporal_relation(self, query: str) -> TemporalRelation:
        """Classify temporal relation using learned classifier"""
        # Tokenize and encode query
        inputs = self.tokenizer(query, return_tensors="pt", 
                               padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            bert_output = self.bert_model(**inputs)
            cls_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Classify
        logits = self.classifier(cls_embedding)
        predicted_idx = torch.argmax(logits, dim=-1).item()
        
        return list(TemporalRelation)[predicted_idx]
    
    def extract_temporal_cues(self, query: str) -> Dict:
        """Main method to extract all temporal cues from query"""
        # Pattern-based extraction
        pattern_markers = self.extract_temporal_markers(query)
        
        # Classifier-based extraction
        classifier_relation = self.classify_temporal_relation(query)
        
        # Combine results (prefer classifier if confident, fallback to patterns)
        primary_relation = classifier_relation if classifier_relation != TemporalRelation.NONE else pattern_markers[0]
        
        return {
            "primary_relation": primary_relation,
            "detected_markers": pattern_markers,
            "query": query,
            "has_temporal_cues": primary_relation != TemporalRelation.NONE
        }