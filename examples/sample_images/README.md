# Sample Images

This directory contains example connectivity matrix images.

## Image Types

- `connectivity_matrix_absCPCC_alpha.png`: Absolute CPCC for alpha band
- `connectivity_matrix_absCPCC_theta.png`: Absolute CPCC for theta band
- `connectivity_matrix_imCPCC_alpha.png`: Imaginary CPCC for alpha band

## Image Format

- **Format**: PNG
- **Content**: Connectivity matrix (129x129 channels)
- **Visualization**: Heatmap showing connectivity strength between channels
- **Color scale**: Red-Yellow-Blue (RdYlBu_r colormap)

## Usage

These sample images demonstrate the expected input format for the classification model. Each image represents the connectivity pattern for a specific frequency band and CPCC type.

For multiview learning, the model combines images from all 6 frequency bands (delta, theta, alpha, low_beta, high_beta, gamma) as input channels.

