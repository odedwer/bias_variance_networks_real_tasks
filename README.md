# Bias-Variance Networks on Real Tasks

This repository investigates how bias initialization affects the bias-variance tradeoff in neural networks trained on facial expression recognition tasks.

## Overview

This project studies the relationship between initial bias values and network learning dynamics using the FER2013 facial expression dataset. The research examines whether controlled initialization of biases can influence model variance and generalization.

## Project Structure

### Main Training Scripts

- **[face_recognition_model_comparison.py](face_recognition_model_comparison.py)**: Main training script that implements:
  - `FER2013Dataset`: PyTorch dataset loader for facial expression images
  - `SimpleCNN`: Simple CNN architecture with customizable batch norm and bias initialization
  - Training loop with multiple random seeds and bias configurations
  - Metrics logging to TensorBoard

- **[ResNet.py](ResNet.py)**: ResNet18 architecture implementation from scratch:
  - `BasicBlock`: Residual blocks with optional batch normalization
  - `ResNet`: Full ResNet18 model with customizable batch norm and bias
  - Supports grayscale and color images

### Core Modules

- **[modules.py](modules.py)**: Network utilities and base classes:
  - `DeviceDataLoader`: Automatically moves batches to GPU/CPU
  - `FER2013`: Original FER2013 dataset class with label mapping
  - `BiasVarianceNetwork`: Base class for networks with custom weight/bias initialization
  - `SimpleCNN`: CNN architecture with customizable architecture

- **[utils.py](utils.py)**: Training and logging utilities:
  - `get_device()`: Returns available device (CUDA/CPU)
  - `init_training()`: Initializes model, optimizer, and criterion
  - `train_epoch()`: Single training epoch with loss computation
  - `epoch_validation()`: Validation loop with metrics logging
  - `test_model()`: Final test evaluation
  - `end_to_end_model_train()`: Complete training pipeline
  - `plot_confusion_matrix()`: Logs confusion matrices to TensorBoard

### Saliency Analysis (`saliency_project/`)

The `saliency_project` directory contains tools for analyzing what image regions the model focuses on:

- **[compute_saliency.py](saliency_project/compute_saliency.py)**: 
  - `compute_saliency()`: Gradient-based saliency map computation

- **[metrics.py](saliency_project/metrics.py)**:
  - `saliency_entropy()`: Measure concentration of attention
  - `top_k_concentration()`: Fraction of saliency in top k% pixels
  - `face_part_coverage()`: Coverage of different face regions (eyes, nose, mouth)
  - `saliency_attribution()`: Distribution of saliency across face regions
  - `mdl_cluster_analysis()`: MDL principle-based clustering
  - `dbscan_cluster_analysis()`: DBSCAN-based region clustering
  - `connected_component_analysis()`: Connected component analysis of salient regions

- **[plots.py](saliency_project/plots.py)**:
  - `draw_mask_contour()`: Visualize face region masks
  - `extract_seed_from_name()`: Parse model metadata from filenames
  - `create_metrics_table()`: Tabular visualization of metrics

- **[run_analysis.py](saliency_project/run_analysis.py)**:
  - Main analysis script that processes models and computes saliency metrics
  - Generates visualizations and output CSV files

## Data

The project uses the FER2013 facial expression dataset:
- Located in `data/face-expression/`
- Contains 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise
- This implementation uses binary classification: `fear` vs `angry`
- Train/test split: 80/20 with stratification

## Training

### Running Training

```python
python face_recognition_model_comparison.py
```

This trains multiple models with configurations:
- **Models**: SimpleCNN, ResNet18
- **Batch Norm**: With/Without
- **Bias Initialization**: None (default), 0.0, 0.1, 1.0, 10.0
- **Seeds**: 30 different random seeds for variance estimation

### Output

Training outputs are saved to:
- `runs/[timestamp]_[model]_[classes]/`: TensorBoard logs
- `models/[model_name]/`: Model checkpoints
  - `init.pth`: Initial model weights
  - `epoch-*.pth`: Checkpoints at intervals
  - `epoch-final.pth`: Final trained model
  - `params.csv`: Training hyperparameters

### TensorBoard Visualization

```bash
tensorboard --logdir runs/
```

View training loss, validation accuracy, bias histograms, and confusion matrices.

## Analysis

### Model Analysis

[model_analysis.py](model_analysis.py): Analyzes trained models across epochs:
- CKA (Centered Kernel Alignment) similarity between models
- Feature evolution during training
- Layer-wise analysis of weight and bias distributions

### Saliency Analysis

```python
cd saliency_project
python run_analysis.py
```

Computes:
- Saliency maps for each test image
- Face region attribution and coverage
- Clustering analysis of attention patterns
- Statistical comparison of models

Output: `saliency_metrics_*.csv` with metrics for each model

## Key Configuration Parameters

In [face_recognition_model_comparison.py](face_recognition_model_comparison.py):

```python
bn_list = [False]                    # Batch norm enabled
init_bias_list = [0.0, 10.0, None, 1.0]  # Bias initialization values
seeds = [42, 0, 1, ...]              # 30 random seeds
num_epochs = 70                       # Training epochs
lr = 1e-3                            # Learning rate
```

## Dependencies

- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tensorboard

Install with:
```bash
pip install -r requirements.txt
```

## Project Organization

```
.
├── Main Scripts
│   ├── face_recognition_model_comparison.py
│   ├── ResNet.py
│   ├── modules.py
│   └── utils.py
├── Analysis
│   ├── model_analysis.py
│   └── saliency_project/
├── Data
│   └── data/face-expression/
├── Results
│   ├── models/
│   ├── runs/
│   └── saliency results/
└── Documentation
    └── README.md
```


## Notes

- Models are trained with mixed precision (torch.amp) for efficiency
- Cross-entropy loss with class weighting to handle imbalance
- Validation accuracy logged every epoch to TensorBoard
- All experiments use deterministic algorithms for reproducibility
- Bias variance is controlled via custom initialization in `reinitialize()` methods

## Related Files

- `pyproject.toml`: Project metadata
- `requirements.txt`: Dependencies
- `.gitignore`: Git ignore rules (if applicable)

