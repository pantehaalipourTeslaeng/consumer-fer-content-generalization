# Facial Emotion Recognition for Content Generalizability

This repository implements a system for analyzing facial emotions and addressing the challenge of generalizing deep learning models across diverse content types. The project integrates multiple data sources, including controlled experimental data, social media data, and synthetic data generated using Generative Adversarial Networks (GANs), to create models that perform reliably across diverse and representative content.

## Project Overview

The project implements an Xception-based architecture for facial emotion recognition with the following key features:
- Multi-source data integration
- Synthetic data generation using StyleGAN2
- Content-based generalizability testing
- Advanced data augmentation techniques

## Data Sources

The project uses three types of data sources:

1. **NeuroBioSense Dataset** (controlled experimental data)
   - Due to privacy and licensing considerations, we provide scripts to process raw videos which are available in ([NeuroBioSense Dataset](https://data.mendeley.com/datasets/7md7yty9sk/2))
   - Sample data and preprocessing pipeline included for demonstration

2. **YouTube Data**
   - Scripts for collecting and processing public YouTube content
   - CSV file with video references (links) provided
   - Data collection and preprocessing pipeline included

3. **Synthetic Data (StyleGAN2)**
   - Implementation of StyleGAN2-based synthetic data generation
   - Scripts to generate diverse facial expressions
   - Sample synthetic images included for demonstration

## Setup and Installation

### System Requirements
- CUDA 11.0 or higher
- NVIDIA GPU with at least 8GB VRAM (recommended: NVIDIA RTX 2080 Ti or better)
- Python 3.8 or higher
- At least 50GB of free disk space

1. Clone the repository:
```bash
git clone https://github.com/pantehaalipourTeslaeng/consumer-fer-content-generalization.git
cd fer-content-generalization
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure data paths:
```bash
cp configs/paths.yaml.example configs/paths.yaml
# Edit configs/paths.yaml with your local paths
```

## Data Preparation

### NeuroBioSense Data Processing
1. Place your raw videos in the appropriate directory (see `configs/paths.yaml`)
2. Run the frame extraction script:
```bash
python scripts/video_processing/frame_extraction.py
```

### YouTube Data Collection
1. Set up YouTube API credentials (see `docs/youtube_setup.md`)
2. Run the video collection script:
```bash
python scripts/youtube_scraping/video_downloader.py
```

### Synthetic Data Generation

You can generate synthetic data using either of these two approaches:

#### Option 1: Using the simplified generator script
1. Set up StyleGAN2 (see `docs/stylegan2_setup.md`)
2. Generate synthetic images:

```bash
python scripts/synthetic_data/stylegan2_generator.py
```

#### Option 2: Using StyleGAN2-ADA (recommended for more control)
1. Install StyleGAN2 dependencies:
```bash
pip install ninja tensorflow-gan
```

2. Clone and install StyleGAN2-ADA:
```bash
git clone https://github.com/NVlabs/stylegan2-ada
cd stylegan2-ada
pip install -e .
```

3. Set up pretrained models:
```bash
mkdir -p models/stylegan2/pretrained
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl -O models/stylegan2/pretrained/stylegan2-ffhq-256x256.pkl
```

4. Generate synthetic images:
```bash
python scripts/synthetic_data/generate_synthetic_dataset.py --config configs/stylegan_config.yaml
```

## Model Training

1. Prepare your configuration:
```bash
cp configs/model_config.yaml.example configs/model_config.yaml
# Edit configs/model_config.yaml as needed
```

2. Train the model:
```bash
python src/model/train.py --config configs/model_config.yaml
```

## Results and Evaluation

The model achieves the following performance metrics:
- Precision-Recall AUC: 0.94 (Baseline + StyleGAN2)
- ROC-AUC: 0.94 (Baseline + StyleGAN2)

## Citation

If you use this code in your research, please cite:
```bibtex
@article{alipour2024leveraging,
  title={Leveraging Generative AI Synthetic and Social Media Data for Content Generalizability to Overcome Data Constraints in Vision Deep Learning},
  author={Alipour, Panteha and Gallegos, Erika},
  journal={[Artificial Intelligence Review]},
  year={2024}
}
```

## License

This project is licensed under the [appropriate license] - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Cloud Research Team for providing computational resources
- StyleGAN2 developers for their groundbreaking work
- The research community for their valuable contributions

## Contact

For any questions or concerns, please open an issue or contact the authors:
- Panteha Alipour
- Erika Gallegos 

## Model Architecture

### Xception-based FER Model
Our model is based on the Xception architecture with the following modifications:

```
Input (224x224x3)
│
├── Entry Flow
│   ├── Conv2D (32 filters) + BatchNorm + ReLU
│   ├── Conv2D (64 filters) + BatchNorm + ReLU
│   └── 3x Separable Convolution Blocks
│
├── Middle Flow
│   └── 8x Residual Blocks with Separable Convolutions
│
├── Exit Flow
│   ├── Separable Convolution Block
│   ├── Global Average Pooling
│   └── Dense Layer (7 emotions)
│
└── Output (Softmax)
```

Key Features:
- Separable convolutions for efficiency
- Residual connections to prevent vanishing gradients
- Batch normalization for training stability
- Global average pooling to reduce parameters

## Evaluation Methodology

### Metrics
- **Accuracy**: Overall classification accuracy across 7 emotions
- **Precision-Recall AUC**: Area under the precision-recall curve for each emotion
- **ROC-AUC**: Area under the ROC curve for binary classification per emotion
- **Confusion Matrix**: To analyze per-class performance

### Cross-Dataset Evaluation
We evaluate model generalizability across three scenarios:
1. **Within-Domain**: Testing on held-out data from the same source
2. **Cross-Domain**: Testing on different data sources
3. **Cross-Content**: Testing on different content types (controlled vs. social media)

### Evaluation Protocol
1. Train on combined dataset (80%)
2. Validate on mixed-source validation set (10%)
3. Test separately on:
   - NeuroBioSense test set
   - YouTube test set
   - Synthetic test set

## Troubleshooting Guide

### Common Issues

#### Data Processing
1. **Video Frame Extraction Fails**
   - Check video codec compatibility
   - Ensure sufficient disk space
   - Verify video file integrity
   ```bash
   ffmpeg -v error -i input.mp4 -f null - 2>error.log
   ```

2. **Memory Issues During Training**
   - Reduce batch size in config
   - Enable gradient accumulation
   - Use mixed precision training
   ```yaml
   training:
     batch_size: 16
     mixed_precision: true
     gradient_accumulation_steps: 2
   ```

3. **CUDA Out of Memory**
   - Monitor GPU usage: `nvidia-smi`
   - Clear cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Error Messages and Solutions

| Error | Possible Cause | Solution |
|-------|---------------|----------|
| `CUDA out of memory` | Batch size too large | Reduce batch size in config |
| `File not found` | Incorrect data paths | Check paths.yaml configuration |
| `ValueError: dim mismatch` | Input size mismatch | Verify image preprocessing |

### Performance Issues
- If training is slow: Enable mixed precision training
- If inference is slow: Use model quantization
- If preprocessing is slow: Increase number of workers
