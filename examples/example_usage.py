#!/usr/bin/env python3
"""
Example usage of CPCC classification model

This script demonstrates how to:
1. Generate connectivity images from CSV data
2. Train a classification model
3. Evaluate the trained model
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def example_generate_images():
    """Example: Generate connectivity images from CSV data"""
    print("=" * 80)
    print("Example 1: Generate Connectivity Images")
    print("=" * 80)
    
    print("""
# Generate connectivity images from CSV files
python generate_connectivity_images.py \\
    --data_dir /path/to/connectivity/csv/files \\
    --output_dir ./images/connectivity \\
    --num_workers 8
    """)
    
    print("This will create PNG images for each frequency band and CPCC type.")


def example_train_model():
    """Example: Train a classification model"""
    print("\n" + "=" * 80)
    print("Example 2: Train Classification Model")
    print("=" * 80)
    
    print("""
# Basic training with absCPCC
python src/models/train_task_classification_images_xai.py \\
    --value_type absCPCC \\
    --multiview \\
    --batch_size 32 \\
    --learning_rate 1e-4 \\
    --num_epochs 50 \\
    --save_dir ./checkpoints/my_model

# Advanced training with both absCPCC and imCPCC
python src/models/train_task_classification_images_xai.py \\
    --value_type absCPCC \\
    --multiview \\
    --use_both_cpcc \\
    --optimizer_type lion \\
    --scheduler_type cosine_warm_restarts \\
    --use_amp \\
    --use_data_augmentation \\
    --aug_strength medium \\
    --save_dir ./checkpoints/my_model_both
    """)


def example_evaluate_model():
    """Example: Evaluate a trained model"""
    print("\n" + "=" * 80)
    print("Example 3: Evaluate Trained Model")
    print("=" * 80)
    
    print("""
# Evaluate on test set
python src/models/evaluate_test_only.py \\
    --checkpoint ./checkpoints/my_model/best_model.pt \\
    --image_base_dir ./images/connectivity \\
    --device cuda \\
    --batch_size 32
    """)


def example_data_structure():
    """Example: Expected data structure"""
    print("\n" + "=" * 80)
    print("Example 4: Expected Data Structure")
    print("=" * 80)
    
    print("""
images/connectivity/
├── sub-XXX_task1_connectivity/
│   ├── connectivity_matrix_absCPCC_alpha.png
│   ├── connectivity_matrix_absCPCC_delta.png
│   ├── connectivity_matrix_absCPCC_gamma.png
│   ├── connectivity_matrix_absCPCC_high_beta.png
│   ├── connectivity_matrix_absCPCC_low_beta.png
│   ├── connectivity_matrix_absCPCC_theta.png
│   ├── connectivity_matrix_imCPCC_alpha.png
│   └── ... (imCPCC for other bands)
└── sub-YYY_task2_connectivity/
    └── ...
    """)


if __name__ == '__main__':
    example_generate_images()
    example_train_model()
    example_evaluate_model()
    example_data_structure()
    
    print("\n" + "=" * 80)
    print("For more details, see README.md")
    print("=" * 80)

