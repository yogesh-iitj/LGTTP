import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import argparse

from lgttp.core.lgttp_pruner import LGTTPruner
from lgttp.core.temporal_adapter import TemporalAdapter
from lgttp.utils.metrics import compute_metrics
from lgttp.models.timechat_integration import TimeChat_LGTTP
from lgttp.models.llava_video_integration import LLaVAVideo_LGTTP

class LGTTPTrainer:
    def __init__(self, config_path="configs/default_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model based on config
        self.model_type = self.config['model']['type']
        if self.model_type == 'timechat':
            self.model = TimeChat_LGTTP(
                alpha=self.config['lgttp']['alpha'],
                min_token_ratio=self.config['lgttp']['min_token_ratio']
            )
        elif self.model_type == 'llava_video':
            self.model = LLaVAVideo_LGTTP(
                alpha=self.config['lgttp']['alpha'],
                min_token_ratio=self.config['lgttp']['min_token_ratio']
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Initialized LGTTP trainer with {self.model_type} on {self.device}")
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (customize based on your data format)
            queries, videos, frame_embeddings, query_embeddings, targets = batch
            
            # Move to device
            frame_embeddings = frame_embeddings.to(self.device)
            query_embeddings = query_embeddings.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            
            batch_size = frame_embeddings.size(0)
            batch_loss = 0
            
            for i in range(batch_size):
                query = queries[i]
                frame_emb = frame_embeddings[i:i+1]
                query_emb = query_embeddings[i:i+1]
                target = targets[i] if isinstance(targets, torch.Tensor) else targets[i]
                
                # Forward pass through LGTTP
                output, pruning_info = self.model.forward_with_lgttp(
                    query=query,
                    frame_embeddings=frame_emb,
                    query_embedding=query_emb
                )
                
                # Compute task-specific loss (customize based on your task)
                loss = self.compute_task_loss(output, target, pruning_info)
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / batch_size
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        return total_loss / len(dataloader)
    
    def compute_task_loss(self, output, target, pruning_info):
        """Compute task-specific loss - customize based on your task"""
        # Example: For video QA or temporal grounding
        if isinstance(target, dict):
            # Multi-task loss
            total_loss = 0
            if 'qa_loss' in target:
                qa_loss = nn.CrossEntropyLoss()(output['qa_logits'], target['qa_labels'])
                total_loss += qa_loss
            
            if 'temporal_loss' in target:
                temporal_loss = nn.MSELoss()(output['temporal_pred'], target['temporal_gt'])
                total_loss += temporal_loss
            
            # Add efficiency regularization
            compression_ratio = pruning_info.get('compression_ratio', 0)
            efficiency_bonus = self.config['training']['efficiency_weight'] * compression_ratio
            total_loss -= efficiency_bonus
            
            return total_loss
        else:
            # Simple classification loss
            return nn.CrossEntropyLoss()(output, target)
    
    def validate(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                queries, videos, frame_embeddings, query_embeddings, targets = batch
                
                frame_embeddings = frame_embeddings.to(self.device)
                query_embeddings = query_embeddings.to(self.device)
                
                batch_size = frame_embeddings.size(0)
                batch_loss = 0
                
                for i in range(batch_size):
                    query = queries[i]
                    frame_emb = frame_embeddings[i:i+1]
                    query_emb = query_embeddings[i:i+1]
                    target = targets[i] if isinstance(targets, torch.Tensor) else targets[i]
                    
                    output, pruning_info = self.model.forward_with_lgttp(
                        query=query,
                        frame_embeddings=frame_emb,
                        query_embedding=query_emb
                    )
                    
                    loss = self.compute_task_loss(output, target, pruning_info)
                    batch_loss += loss
                    
                    # Collect predictions for metrics
                    all_predictions.append(output.cpu())
                    all_targets.append(target)
                
                total_val_loss += batch_loss.item() / batch_size
        
        # Compute validation metrics
        val_metrics = compute_metrics(all_predictions, all_targets, task_type=self.config['task']['type'])
        
        return total_val_loss / len(val_dataloader), val_metrics
    
    def train(self, train_dataloader, val_dataloader=None):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_metrics = self.validate(val_dataloader)
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Metrics: {val_metrics}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model(f"checkpoints/best_model_epoch_{epoch+1}.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['training']['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_model(f"checkpoints/checkpoint_epoch_{epoch+1}.pth")
        
        print("Training completed!")
    
    def save_model(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Train LGTTP model')
    parser.add_argument('--config', default='configs/default_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--data_dir', required=True, 
                       help='Path to training data directory')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = LGTTPTrainer(args.config)
    
    # Load your data here - customize based on your dataset
    # train_dataloader = create_dataloader(args.data_dir, split='train')
    # val_dataloader = create_dataloader(args.data_dir, split='val')
    
    # Start training
    # trainer.train(train_dataloader, val_dataloader)
    

if __name__ == "__main__":
    main()