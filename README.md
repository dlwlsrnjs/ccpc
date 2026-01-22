# CPCC: Cross-Frequency Phase Connectivity Classification

This repository contains the core code for task classification using cross-frequency phase connectivity (CPCC) features from EEG data.

## Overview

This project implements a CNN-based classification model that takes connectivity matrix images as input and classifies cognitive tasks. The model supports:

- **Multiview Learning**: Combining multiple frequency bands (delta, theta, alpha, low_beta, high_beta, gamma) as input channels
- **CPCC Types**: Support for both absolute CPCC (absCPCC) and imaginary CPCC (imCPCC)
- **Subject-Independent Splitting**: Train/validation/test splits based on subjects to avoid data leakage
- **XAI Support**: Grad-CAM visualization for model interpretability

## Project Structure

```
.
├── src/
│   └── models/
│       ├── task_classification_from_images_xai.py  # Main model and dataset classes
│       ├── train_task_classification_images_xai.py  # Training script
│       └── evaluate_test_only.py                    # Evaluation script
├── generate_connectivity_images.py                  # Connectivity image generation
├── requirements.txt                                 # Dependencies
└── README.md                                        # This file
```

## Features

### 1. Connectivity Image Generation
- Converts connectivity CSV data to images for CNN training
- Supports multiple frequency bands and CPCC types
- Optimized for batch processing

### 2. Model Architecture
- ResNet-based backbone (ResNet18/34/50)
- Supports multiview input (18 or 36 channels)
- XAI-ready architecture with feature map access

### 3. Training Features
- Subject-independent train/val/test split
- Age-stratified splitting for balanced validation/test sets
- Advanced data augmentation (MixUp, CutMix, etc.)
- Multiple optimizers (AdamW, Lion, Adafactor)
- Learning rate schedulers (Plateau, Cosine Warm Restarts, OneCycle)
- Mixed precision training (AMP)
- Early stopping

### 4. Evaluation
- Test set evaluation with comprehensive metrics
- Per-task accuracy analysis
- Confusion matrix generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dlwlsrnjs/ccpc.git
cd ccpc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Connectivity Images

First, generate connectivity matrix images from CSV data:

```bash
python generate_connectivity_images.py \
    --data_dir /path/to/connectivity/csv/files \
    --output_dir /path/to/output/images
```

### 2. Train Model

Train a model with multiview learning:

```bash
python src/models/train_task_classification_images_xai.py \
    --value_type absCPCC \
    --multiview \
    --use_both_cpcc \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --optimizer_type lion \
    --scheduler_type cosine_warm_restarts \
    --use_amp \
    --save_dir ./checkpoints/my_model
```

### 3. Evaluate Model

Evaluate a trained checkpoint:

```bash
python src/models/evaluate_test_only.py \
    --checkpoint ./checkpoints/my_model/best_model.pt \
    --image_base_dir /path/to/images \
    --device cuda \
    --batch_size 32
```

## Key Parameters

### Model Configuration
- `--value_type`: `absCPCC` or `imCPCC`
- `--multiview`: Enable multiview learning (combines 6 frequency bands)
- `--use_both_cpcc`: Use both absCPCC and imCPCC (36 channels)
- `--backbone`: `resnet18`, `resnet34`, or `resnet50`

### Training Configuration
- `--optimizer_type`: `adam`, `adamw`, `lion`, or `adafactor`
- `--scheduler_type`: `plateau`, `cosine_warm_restarts`, or `onecycle`
- `--use_amp`: Enable mixed precision training
- `--use_data_augmentation`: Enable data augmentation
- `--use_mixup`: Enable MixUp augmentation
- `--use_cutmix`: Enable CutMix augmentation

## Data Format

### Input Images
- Format: PNG images of connectivity matrices
- Naming: `connectivity_matrix_{value_type}_{freq_band}.png`
- Example: `connectivity_matrix_absCPCC_alpha.png`

### Directory Structure
```
images/connectivity/
├── sub-XXX_task1_connectivity/
│   ├── connectivity_matrix_absCPCC_alpha.png
│   ├── connectivity_matrix_absCPCC_delta.png
│   └── ...
└── sub-YYY_task2_connectivity/
    └── ...
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ccpc2024,
  title={Cross-Frequency Phase Connectivity Classification for EEG Task Classification},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
