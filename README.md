# Language-Guided Temporal Token Pruning (LGTTP)


## Installation

```bash
git clone https://github.com/yogesh-iitj/LGTTP.git
cd LGTTP
pip install -r requirements.txt

## Repository Structure

LGTTP/
├── README.md
├── requirements.txt
├── train.py                 # Training script
├── inference.py             # Inference/testing script
├── lgttp/
│   ├── core/               # Core LGTTP components
│   │   ├── temporal_cue_extractor.py
│   │   ├── temporal_weight_generator.py
│   │   ├── temporal_adapter.py
│   │   └── lgttp_pruner.py
│   ├── models/             # Model integrations
│   │   ├── timechat_integration.py
│   │   └── llava_video_integration.py
│   └── utils/
│       └── metrics.py
├── configs/
│   └── default_config.yaml
└── examples/
    ├── demo.py
    └── integration_example.py

## Training
# Train with TimeChat integration
python train.py --config configs/default_config.yaml --data_dir /path/to/data

# Customize training parameters
python train.py --config configs/custom_config.yaml --data_dir /path/to/data

## Inference

# Interactive demo
python inference.py --model_path checkpoints/best_model.pth --mode demo

# Evaluate on test dataset  
python inference.py --model_path checkpoints/best_model.pth --mode eval --test_data /path/to/test

# Batch inference
python inference.py --model_path checkpoints/best_model.pth --mode eval --output results.json


