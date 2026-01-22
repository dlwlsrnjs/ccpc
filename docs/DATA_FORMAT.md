# Data Format Documentation

## Overview

This document describes the expected data format for CPCC classification.

## Input Data Structure

### Directory Structure

```
images/connectivity/
├── sub-XXX_task1_connectivity/
│   ├── connectivity_matrix_absCPCC_alpha.png
│   ├── connectivity_matrix_absCPCC_delta.png
│   ├── connectivity_matrix_absCPCC_gamma.png
│   ├── connectivity_matrix_absCPCC_high_beta.png
│   ├── connectivity_matrix_absCPCC_low_beta.png
│   ├── connectivity_matrix_absCPCC_theta.png
│   ├── connectivity_matrix_imCPCC_alpha.png
│   ├── connectivity_matrix_imCPCC_delta.png
│   └── ... (imCPCC for other bands)
└── sub-YYY_task2_connectivity/
    └── ...
```

### Naming Convention

- **Directory name format**: `sub-{SUBJECT_ID}_{TASK_NAME}_connectivity`
- **Image name format**: `connectivity_matrix_{VALUE_TYPE}_{FREQ_BAND}.png`
  - `VALUE_TYPE`: `absCPCC` or `imCPCC`
  - `FREQ_BAND`: `alpha`, `delta`, `gamma`, `high_beta`, `low_beta`, `theta`

### Frequency Bands

The model supports 6 frequency bands:
- `delta`: 1-4 Hz
- `theta`: 4-8 Hz
- `alpha`: 8-13 Hz
- `low_beta`: 13-20 Hz
- `high_beta`: 20-30 Hz
- `gamma`: 30-50 Hz

### CPCC Types

- **absCPCC**: Absolute value of Cross-Frequency Phase Connectivity
- **imCPCC**: Imaginary part of Cross-Frequency Phase Connectivity

## Image Format

- **Format**: PNG
- **Content**: Connectivity matrix visualization
- **Size**: Any (will be resized to 224x224 during training)
- **Channels**: RGB (3 channels)

## CSV Input Format (for image generation)

If you have CSV files with connectivity data, use `generate_connectivity_images.py` to convert them to images.

### Required CSV Columns

- `absCPCC`: Absolute CPCC value
- `imCPCC`: Imaginary CPCC value (optional)
- `freq_band`: Frequency band name
- `channel_i_idx`: Source channel index
- `channel_j_idx`: Target channel index
- `num_channels`: Total number of channels
- `window_idx`: Window index (for averaging)

### Example CSV Structure

```csv
absCPCC,imCPCC,freq_band,channel_i_idx,channel_j_idx,num_channels,window_idx
0.85,0.12,alpha,0,1,129,0
0.72,0.08,alpha,0,2,129,0
...
```

## Output Format

### Model Checkpoint

The trained model is saved as:
```
checkpoints/{model_name}/best_model.pt
```

Contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Training epoch
- `val_acc`: Validation accuracy
- `val_loss`: Validation loss
- `num_classes`: Number of task classes
- `backbone`: Backbone architecture name

### Evaluation Results

Evaluation outputs:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- Per-task accuracy breakdown
- Confusion matrix

