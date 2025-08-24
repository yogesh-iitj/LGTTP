import torch
import yaml
import argparse
import json
from tqdm import tqdm
import os

from lgttp.core.lgttp_pruner import LGTTPruner
from lgttp.models.timechat_integration import TimeChat_LGTTP
from lgttp.models.llava_video_integration import LLaVAVideo_LGTTP
from lgttp.utils.metrics import compute_metrics

class LGTTPInference:
    def __init__(self, model_path, config_path="configs/default_config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model_type = checkpoint['config']['model']['type']
        
        if self.model_type == 'timechat':
            self.model = TimeChat_LGTTP(
                alpha=checkpoint['config']['lgttp']['alpha'],
                min_token_ratio=checkpoint['config']['lgttp']['min_token_ratio']
            )
        elif self.model_type == 'llava_video':
            self.model = LLaVAVideo_LGTTP(
                alpha=checkpoint['config']['lgttp']['alpha'],
                min_token_ratio=checkpoint['config']['lgttp']['min_token_ratio']
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded LGTTP model ({self.model_type}) on {self.device}")
    
    def process_single_query(self, query, video_path=None, frame_embeddings=None, query_embedding=None):
        """Process a single query-video pair"""
        with torch.no_grad():
            # If video_path is provided, process video to get embeddings
            if video_path is not None and frame_embeddings is None:
                frame_embeddings, query_embedding = self.process_video(video_path, query)
            
            # Move to device
            if isinstance(frame_embeddings, torch.Tensor):
                frame_embeddings = frame_embeddings.to(self.device)
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.to(self.device)
            
            # Run LGTTP inference
            output, pruning_info = self.model.forward_with_lgttp(
                query=query,
                frame_embeddings=frame_embeddings,
                query_embedding=query_embedding
            )
            
            return output, pruning_info
    
    def process_video(self, video_path, query):
        """Process video file to extract embeddings - customize based on your preprocessing"""
        print(f"Processing video: {video_path} with query: {query}")
        
        # Example: Load and process video frames
        # frames = load_video_frames(video_path)
        # frame_embeddings = self.model.encode_frames(frames)
        # query_embedding = self.model.encode_query(query)
        
        # For now, return dummy embeddings
        num_frames = 16
        embedding_dim = 768
        frame_embeddings = torch.randn(1, num_frames, embedding_dim)
        query_embedding = torch.randn(1, embedding_dim)
        
        return frame_embeddings, query_embedding
    
    def evaluate_dataset(self, test_dataloader, output_path="results.json"):
        """Evaluate on test dataset"""
        all_results = []
        all_metrics = {'total_samples': 0, 'avg_compression': 0, 'task_metrics': {}}
        
        print("Running evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
                queries, video_paths, frame_embeddings, query_embeddings, targets = batch
                
                batch_results = []
                batch_size = len(queries)
                
                for i in range(batch_size):
                    query = queries[i]
                    video_path = video_paths[i] if video_paths else None
                    frame_emb = frame_embeddings[i:i+1] if frame_embeddings is not None else None
                    query_emb = query_embeddings[i:i+1] if query_embeddings is not None else None
                    target = targets[i] if targets else None
                    
                    # Run inference
                    try:
                        output, pruning_info = self.process_single_query(
                            query=query,
                            video_path=video_path,
                            frame_embeddings=frame_emb,
                            query_embedding=query_emb
                        )
                        
                        result = {
                            'query': query,
                            'video_path': video_path,
                            'output': output.cpu().tolist() if isinstance(output, torch.Tensor) else output,
                            'target': target,
                            'pruning_info': {
                                'temporal_relation': pruning_info['temporal_cues']['primary_relation'].value,
                                'compression_ratio': pruning_info['compression_ratio'],
                                'tokens_retained': pruning_info['tokens_retained'].cpu().tolist()
                            }
                        }
                        
                        batch_results.append(result)
                        all_metrics['avg_compression'] += pruning_info['compression_ratio']
                        all_metrics['total_samples'] += 1
                        
                    except Exception as e:
                        print(f"Error processing sample {batch_idx}_{i}: {str(e)}")
                        continue
                
                all_results.extend(batch_results)
        
        # Compute final metrics
        if all_metrics['total_samples'] > 0:
            all_metrics['avg_compression'] /= all_metrics['total_samples']
            
            # Compute task-specific metrics
            predictions = [r['output'] for r in all_results]
            ground_truths = [r['target'] for r in all_results]
            task_metrics = compute_metrics(predictions, ground_truths, 
                                         task_type=self.config.get('task', {}).get('type', 'classification'))
            all_metrics['task_metrics'] = task_metrics
        
        # Save results
        final_results = {
            'config': self.config,
            'metrics': all_metrics,
            'detailed_results': all_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {output_path}")
        print(f"Average compression ratio: {all_metrics['avg_compression']:.2%}")
        print(f"Task metrics: {all_metrics['task_metrics']}")
        
        return final_results
    
    def interactive_demo(self):
        """Interactive demo for testing queries"""
        print("LGTTP Interactive Demo")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() == 'quit':
                break
            
            video_path = input("Enter video path (or press Enter for dummy data): ").strip()
            if not video_path:
                video_path = None
            
            try:
                output, pruning_info = self.process_single_query(
                    query=query,
                    video_path=video_path
                )
                
                print(f"\nResults:")
                print(f"Temporal relation: {pruning_info['temporal_cues']['primary_relation'].value}")
                print(f"Compression ratio: {pruning_info['compression_ratio']:.2%}")
                print(f"Output: {output}")
                
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='LGTTP Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config', default='configs/default_config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['eval', 'demo'], default='demo', 
                       help='Inference mode: eval or demo')
    parser.add_argument('--test_data', help='Path to test data (for eval mode)')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = LGTTPInference(args.model_path, args.config)
    
    if args.mode == 'demo':
        inference.interactive_demo()
    elif args.mode == 'eval':
        if args.test_data is None:
            print("Error: --test_data required for eval mode")
            return
        
        # Load test dataloader - customize based on your data format
        # test_dataloader = create_test_dataloader(args.test_data)
        # inference.evaluate_dataset(test_dataloader, args.output)

if __name__ == "__main__":
    main()