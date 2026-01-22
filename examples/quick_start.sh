#!/bin/bash
# Quick Start Script for CPCC Classification

set -e

echo "=========================================="
echo "CPCC Classification - Quick Start"
echo "=========================================="
echo ""

# Check if required directories exist
if [ ! -d "images/connectivity" ]; then
    echo "⚠️  Warning: images/connectivity directory not found"
    echo "   Please generate connectivity images first using:"
    echo "   python generate_connectivity_images.py"
    echo ""
fi

# Step 1: Generate images (if needed)
echo "Step 1: Generate Connectivity Images"
echo "--------------------------------------"
read -p "Do you want to generate connectivity images? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running image generation..."
    python generate_connectivity_images.py \
        --data_dir ./data/processed/connectivity/multiband/data \
        --output_dir ./images/connectivity
fi

# Step 2: Train model
echo ""
echo "Step 2: Train Classification Model"
echo "--------------------------------------"
read -p "Do you want to train a model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    MODEL_NAME="my_cpcc_model"
    echo "Training model: $MODEL_NAME"
    python src/models/train_task_classification_images_xai.py \
        --value_type absCPCC \
        --multiview \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --num_epochs 50 \
        --optimizer_type lion \
        --scheduler_type cosine_warm_restarts \
        --use_amp \
        --save_dir "./checkpoints/$MODEL_NAME"
fi

# Step 3: Evaluate model
echo ""
echo "Step 3: Evaluate Model"
echo "--------------------------------------"
read -p "Do you want to evaluate a model? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter checkpoint path: " CHECKPOINT_PATH
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Evaluating model..."
        python src/models/evaluate_test_only.py \
            --checkpoint "$CHECKPOINT_PATH" \
            --image_base_dir ./images/connectivity \
            --device cuda \
            --batch_size 32
    else
        echo "⚠️  Checkpoint not found: $CHECKPOINT_PATH"
    fi
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

