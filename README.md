# GMM-TS: Gating Architecture for Multimodal Time Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **GMM-TS: Gating Architecture for Multimodal Time Series Forecasting**.

## Overview

GMM-TS is a novel framework for multimodal time series forecasting that combines numerical time series data with textual information through a learnable gating mechanism. The framework dynamically weighs predictions from multiple uni-modal experts (TSF-N for numerical data and TSF-T for text-based forecasting) using a transformer-based gating network.

### Key Features

- **Learnable Gating Mechanism**: Transformer-based architecture that dynamically computes expert weights at each time step
- **Flexible Expert Integration**: Seamlessly combines multiple TSF-N experts (Informer, DLinear, PatchTST, FiLM, Reformer) and TSF-T experts (BERT, GPT2, LLAMA2, LLAMA3)
- **Dual Training Modes**: 
  - **Online**: End-to-end training with live expert predictions
  - **Offline**: Pre-compute expert latents for efficient experimentation
- **Multimodal Fusion**: Effectively integrates heterogeneous representations from numerical and textual modalities
- **State-of-the-Art Performance**: Achieves superior forecasting accuracy across diverse domains

### Architecture

GMM-TS consists of three main components:

1. **TSF-N Experts**: Numerical time series forecasting models that process sequential numerical data
2. **TSF-T Experts**: Text-enhanced forecasting models that leverage LLMs to incorporate textual information
3. **Gating Network**: Transformer-based module that learns to dynamically weight expert predictions

```
Input Time Series + Text → [TSF-N Expert₁, ..., TSF-N Expertₙ] → Predictions₁, Latents₁
                         ↘ [TSF-T Expert₁, ..., TSF-T Expertₘ] → Predictions₂, Latents₂
                                                                        ↓
                                          Gating Network ← [Latents₁, Latents₂]
                                                  ↓
                                          Final Prediction
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd GMM_TS
```

2. **Install the package**
```bash
pip install -e .
```

3. **Set up data path** (MM-TSFlib is required for datasets)
```bash
# Clone MM-TSFlib for data
git clone https://github.com/AdityaLab/MM-TSFlib.git

# Set environment variable
export MM_TSFLIB_PATH=/path/to/MM-TSFlib
```

### Dependencies

Core dependencies (automatically installed):
- `torch>=2.0.0`
- `transformers>=4.40.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- And others listed in `requirements.txt`

## Supported Datasets

GMM-TS supports forecasting at multiple temporal granularities across diverse domains:

### Daily Datasets
- **Environment**: `NewYork_AQI_Day.csv` (Air Quality Index)
  - Prediction lengths: 48, 96, 192, 336

**Note**: Daily datasets are provided as RAR archives in MM-TSFlib and need to be extracted before use.

### Weekly Datasets
- **Energy**: Weekly energy consumption data
- **Public Health**: Weekly health metrics
  - Prediction lengths: 12, 24, 36, 48

### Monthly Datasets
- **Agriculture**: `US_RetailBroilerComposite_Month.csv`
- **Climate**: `US_precipitation_month.csv`
- **Economy**: `US_TradeBalance_Month.csv`
- **Security**: `US_FEMAGrant_Month.csv`
- **Social Good**: `Unadj_UnemploymentRate_ALL_processed.csv`
- **Traffic**: `US_VMT_Month.csv`
  - Prediction lengths: 6, 8, 10, 12

All datasets are sourced from [MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib) and should be placed in `$MM_TSFLIB_PATH/data/`.

## Quick Start

### Online Training (Recommended for Beginners)

Online training performs end-to-end training where expert predictions are computed on-the-fly:

```bash
# Run online gating for different temporal granularities
bash examples/run_online_gating_daily.sh 0      # Daily datasets
bash examples/run_online_gating_weekly.sh 0     # Weekly datasets
bash examples/run_online_gating_monthly.sh 0    # Monthly datasets
```

Example configurations:

**Daily dataset:**
```bash
python run_online_gating.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $MM_TSFLIB_PATH/data/Environment \
  --data_path NewYork_AQI_Day.csv \
  --model_id Environment_2021_24_48 \
  --model Informer \
  --llm_model GPT2 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001
```

**Monthly dataset:**
```bash
python run_online_gating.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $MM_TSFLIB_PATH/data/Security \
  --data_path US_FEMAGrant_Month.csv \
  --model_id Security_2021_24_6 \
  --model Informer \
  --llm_model GPT2 \
  --data custom \
  --features M \
  --seq_len 8 \
  --label_len 4 \
  --pred_len 6 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.001
```

**Note**: Domain is automatically extracted from `root_path` (e.g., "Security" from `.../data/Security`), and `all_experts_config.csv` is loaded from the default location.

### Offline Training (Recommended for Large-Scale Experiments)

Offline training pre-computes expert latents and predictions, enabling faster experimentation with different gating configurations.

#### Step 1: Generate Expert Latents

**For Daily Datasets:**
```bash
# Generate TSF-N (Numerical) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfns_latents_predictions_daily_datasets.sh 0 4 0

# Generate TSF-T (Text-based) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfts_latents_predictions_daily_datasets.sh 0 2 0
```

**For Weekly Datasets:**
```bash
# Generate TSF-N (Numerical) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfns_latents_predictions_weekly_datasets.sh 0 4 0

# Generate TSF-T (Text-based) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfts_latents_predictions_weekly_datasets.sh 0 2 0
```

**For Monthly Datasets:**
```bash
# Generate TSF-N (Numerical) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfns_latents_predictions_monthly_datasets.sh 0 4 0

# Generate TSF-T (Text-based) Expert Latents
bash scripts/dataset_prep_scripts/save_tsfts_latents_predictions_monthly_datasets.sh 0 2 0
```

This generates latents for all numerical experts (Informer, Reformer, DLinear, PatchTST, FiLM) and LLM-based experts (GPT2, BERT, LLAMA2).

**What gets generated:**
- `train_gating_dataset/` - Training expert latents and predictions
- `val_gating_dataset/` - Validation expert latents and predictions
- `test_gating_dataset/` - Test expert latents and predictions

Each expert's folder contains:
- `latents.npy` - Expert latent representations
- `preds.npy` - Expert predictions
- `trues.npy` - Ground truth values
- `x.npy` - Input sequences

#### Step 2: Generate Expert Configuration File

After generating all expert latents, create the configuration file:

```bash
python scripts/dataset_prep_scripts/prepare_all_expert_config.py \
  --dataset_folder ./train_gating_dataset \
  --output_file all_experts_config.csv
```

This script:
- Scans all generated expert folders
- Extracts latent dimensions automatically
- Creates `all_experts_config.csv` with expert metadata

**Note**: The repository includes a pre-generated `all_experts_config.csv` from our experiments. You only need to regenerate it if you add new experts or datasets.

#### Step 3: Train Gating Network

```bash
# For daily datasets
bash examples/run_offline_gating_daily.sh

# For weekly/monthly datasets
bash examples/run_offline_gating_monthly_weekly.sh
```

Or run directly:
```bash
python run_offline_gating.py \
  --task_name mm_long_term_forecast \
  --pred_len 6 \
  --is_training 1 \
  --agg_type direct \
  --seed 2021 \
  --expert_input_type latent \
  --tsfn_experts Informer,DLinear,PatchTST \
  --tsft_experts GPT2,BERT \
  --domain Security \
  --train_epochs 10
```

## Training Modes Comparison

| Feature | Online Training | Offline Training |
|---------|----------------|------------------|
| **Speed** | Slower (computes expert predictions each epoch) | Faster (uses pre-computed latents) |
| **Memory** | Higher (loads all expert models) | Lower (only gating network) |
| **Flexibility** | End-to-end optimization | Quick gating experiments |
| **Use Case** | Final model training | Hyperparameter tuning, ablations |
| **Storage** | Minimal | Requires disk space for latents |

**When to use Online Training:**
- Final model training for deployment
- When you need end-to-end gradient flow
- Limited disk space

**When to use Offline Training:**
- Running many gating configurations
- Hyperparameter tuning
- Ablation studies
- When you have pre-trained experts

## Expert Configuration

### Configuration Files

GMM-TS uses CSV files to store expert metadata and latent dimensions:

1. **`all_experts_config.csv`** (Required for offline gating):
   - Contains metadata for all expert models across all datasets
   - Includes: expert name, domain, latent dimension, frequency, prediction length, folder path
   - **Generation**: Automatically created by `scripts/dataset_prep_scripts/prepare_all_expert_config.py` after generating expert latents
   - **Usage**: Loaded by `run_offline_gating.py` to locate pre-computed expert latents
   - **Pre-generated**: Included in repository with 290 expert configurations from our experiments
   - **When to regenerate**: Only if you add new experts or datasets

2. **`expert_config.csv`** (Quick-start reference):
   - Simplified configuration with 5 example experts
   - **Purpose**: Documentation and learning
   - **Usage**: Shows expert structure for beginners
   - **Not used**: Online gating specifies experts via command-line args

### CSV File Format

**`all_experts_config.csv`** structure:
```csv
expert,domain,freq,pl,latent_dim,expert_folder
Informer,Security,Monthly,6,5120,long_term_forecast_tsfn_Security_2021_24_6_fullLLM_0_Informer...
DLinear,Security,Monthly,6,512,long_term_forecast_tsfn_Security_2021_24_6_fullLLM_0_DLinear...
GPT2,Security,Monthly,6,96,long_term_forecast_tsft_Security_2021_24_6_fullLLM_0_GPT2...
```

**`expert_config.csv`** structure (simplified):
```csv
expert_name,latent_dim,expert_model_type,root_name,data_name,expert_tag
Informer,5120,tsfn,Security,US_FEMAGrant_Month.csv,Security_2021_24_6_fullLLM_0_Informer
GPT2,96,tsft,Security,US_FEMAGrant_Month.csv,Security_2021_24_6_fullLLM_0_GPT2
```

### How Expert Configuration Works

**Both online and offline training use `all_experts_config.csv`** to look up expert latent dimensions and metadata.

**For Online Training:**
```bash
python run_online_gating.py \
  --model Informer \              # TSF-N expert
  --llm_model GPT2 \              # TSF-T expert
  --root_path $MM_TSFLIB_PATH/data/Security \  # Domain auto-extracted: "Security"
  --pred_len 6
  # all_experts_config.csv loaded automatically from default location
```
- Experts specified via `--model` (TSF-N) and `--llm_model` (TSF-T)
- Domain automatically extracted from last directory in `root_path`
- `all_experts_config.csv` provides latent dimensions for gating architecture
- Predictions computed on-the-fly during training

**For Offline Training:**
```bash
python run_offline_gating.py \
  --tsfn_experts Informer,DLinear \
  --tsft_experts GPT2,BERT \
  --domain Security \
  --pred_len 6 \
  --all_experts_config all_experts_config.csv  # Looks up latent dims + folders
```
- Multiple experts specified via comma-separated lists
- CSV provides latent dimensions AND folder paths to pre-computed latents
- Loads pre-computed predictions from disk

### Adding New Experts

1. Add expert configuration to CSV file
2. Generate latents using modified `run.py`:
```bash
python run.py \
  --model YourModel \
  --save_gating_dataset 1 \
  --prompt_weight 0.0  # For TSF-N
  # or --prompt_weight 1.0 --llm_model YourLLM  # For TSF-T
```

3. Update gating script to include new expert

## Configuration and Parameters

### Default Configuration Files

All default parameters are defined in the main script files:

1. **Gating Network Defaults** (`run_online_gating.py`, `run_offline_gating.py`):
   - Gating architecture parameters
   - Training hyperparameters
   - Aggregation strategies

2. **Expert Training Defaults** (`run.py`):
   - Model architecture parameters (d_model, n_heads, etc.)
   - Expert-specific configurations
   - Data loading parameters

3. **Expert Lists** (CSV files):
   - `expert_config.csv`: Quick-start configuration (5 experts)
   - `all_experts_config.csv`: Complete expert catalog (290 configurations)

### Key Parameters

#### Gating Network (run_online_gating.py / run_offline_gating.py)
- `--gating_d_model`: Gating network hidden dimension (default: 256)
- `--n_heads`: Number of attention heads (default: 8)
- `--e_layers`: Number of encoder layers (default: 2)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--agg_type`: Aggregation strategy (default: 'direct')
- `--expert_input_type`: Expert input type (default: 'latent')

#### Training
- `--train_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.0001 for gating, 0.0001 for experts)
- `--patience`: Early stopping patience (default: 5)
- `--seed`: Random seed (default: 2024)

#### Expert Models (run.py)
- `--d_model`: Model hidden dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--e_layers`: Encoder layers (default: 2)
- `--d_layers`: Decoder layers (default: 1)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--dropout`: Dropout rate (default: 0.1)

#### Task Configuration
- `--seq_len`: Input sequence length (default: 8)
- `--label_len`: Label sequence length (default: 4)
- `--pred_len`: Prediction horizon (default: 6)
- `--features`: Forecasting mode ('M': multivariate, 'S': univariate)

#### Text-based Expert Parameters (run.py)
- `--prompt_weight`: Text influence (0.0: pure numerical, 1.0: pure textual)
- `--llm_model`: LLM backbone (BERT, GPT2, LLAMA2, LLAMA3)
- `--text_len`: Text feature length (default: 4)
- `--pool_type`: Text pooling method (default: 'avg')

### Viewing All Parameters

To see all available parameters with their defaults:

```bash
# For gating network
python run_offline_gating.py --help
python run_online_gating.py --help

# For expert training
python run.py --help
```

## Supported Datasets

GMM-TS has been evaluated on diverse multimodal time series datasets:

### Monthly Datasets
- **Agriculture**: US Retail Broiler Composite Price
- **Climate**: US Precipitation
- **Economy**: US Trade Balance
- **Security**: US FEMA Grants
- **Social Good**: Unemployment Rate
- **Traffic**: US Vehicle Miles Traveled

### Weekly Datasets
- **Energy**: Electricity demand
- **Public Health**: COVID-19 statistics

All datasets include paired numerical time series and textual descriptions.

## Repository Structure

```
GMM_TS/
├── gmm_ts/                          # Main Python package
│   ├── models/                      # TSF models (Informer, DLinear, etc.)
│   ├── layers/                      # Neural network layers
│   ├── gating/                      # Gating mechanism implementation
│   ├── exp/                         # Experiment classes
│   ├── data_provider/               # Data loading utilities
│   └── utils/                       # Helper functions
│
├── scripts/                         # Training scripts
│   ├── dataset_prep_scripts/        # Latent generation scripts
│   │   ├── prepare_all_expert_config.py
│   │   ├── save_tsfns_latents_predictions_daily_datasets.sh
│   │   ├── save_tsfns_latents_predictions_weekly_datasets.sh
│   │   ├── save_tsfns_latents_predictions_monthly_datasets.sh
│   │   ├── save_tsfts_latents_predictions_daily_datasets.sh
│   │   ├── save_tsfts_latents_predictions_weekly_datasets.sh
│   │   └── save_tsfts_latents_predictions_monthly_datasets.sh
│   └── experiments_scripts/         # Gating training scripts
│       └── run_offline_gating_monthly_weekly.sh
│
├── examples/                        # Example usage scripts
│   ├── run_online_gating_daily.sh
│   ├── run_online_gating_weekly.sh
│   ├── run_online_gating_monthly.sh
│   ├── run_offline_gating_daily.sh
│   └── run_offline_gating_monthly_weekly.sh
│
├── run.py                           # Train individual experts
├── run_online_gating.py             # Online gating training
├── run_offline_gating.py            # Offline gating training
│
├── expert_config.csv                # Expert configuration
├── all_experts_config.csv           # Complete expert list
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Advanced Usage

### Custom Expert Combination

```python
# Select specific experts for your task
python run_offline_gating.py \
  --tsfn_experts Informer,PatchTST \
  --tsft_experts GPT2 \
  --expert_input_type latent
```

### Aggregation Strategies

GMM-TS supports three aggregation strategies for combining expert predictions:

#### 1. Direct Aggregation (Default)
```bash
--agg_type direct --expert_input_type latent
```
- Computes time-step-wise weights for each expert
- Weight dimension: `[batch, num_experts, pred_len]`
- Aggregates predictions: `output = Σ(weight_i * prediction_i)` for each time step
- **Best for**: Standard multimodal forecasting tasks

#### 2. Latent Aggregation
```bash
--agg_type latent --expert_input_type latent
```
- Learns expert-specific latent transformations
- Generates: `[batch, num_experts, gating_d_model]`
- Aggregates in latent space before final prediction
- **Best for**: When you want to model expert interactions in latent space

#### 3. Hierarchical Aggregation
```bash
--agg_type hierarchical --expert_input_type latent
```
- Two-level gating mechanism:
  - Level 1: Computes expert weights (like direct)
  - Level 2: Additional refinement layer for final prediction
- Combines weighted expert predictions with learned refinement
- **Best for**: Complex scenarios requiring multi-level decision making

#### Expert Input Type

Additionally, you can control what information from experts is used:

```bash
--expert_input_type latent       # Use expert latent representations (default)
--expert_input_type prediction   # Use expert predictions directly
```

**Example:**
```bash
# Direct aggregation with latent inputs
python run_offline_gating.py \
  --agg_type direct \
  --expert_input_type latent \
  --tsfn_experts Informer,DLinear \
  --tsft_experts GPT2

# Hierarchical aggregation
python run_offline_gating.py \
  --agg_type hierarchical \
  --expert_input_type latent \
  --tsfn_experts Informer,PatchTST \
  --tsft_experts BERT,GPT2
```

### Hyperparameter Tuning

```bash
# Experiment with different gating dimensions
for dim in 128 256 512; do
  python run_offline_gating.py \
    --gating_d_model $dim \
    --des "gating_dim_${dim}"
done
```

## Results and Checkpoints

### Trained Models
- Online gating checkpoints: `./gating_checkpoints/`
- Expert checkpoints: `./checkpoints/`

### Predictions
- Test predictions are saved automatically
- Metrics (MSE, MAE) are logged during training

## Reproducibility

To reproduce paper results:

1. **Generate all expert latents:**
```bash
# TSF-N experts
bash scripts/dataset_prep_scripts/save_tsfns_latents_predictions_monthly_datasets.sh 0 4 0

# TSF-T experts
bash scripts/dataset_prep_scripts/save_tsfts_latents_predictions_monthly_datasets.sh 0 2 0
```

2. **Train gating network:**
```bash
bash scripts/experiments_scripts/run_offline_gating_monthly_weekly.sh
```

3. **Results will be saved in:** `./results/`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon:
- [MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib) for the multimodal time series framework
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) for baseline TSF models
- HuggingFace Transformers for LLM integration

## Contact

For questions and feedback:
- Open an issue on GitHub
- Contact: [Your contact information]

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the existing style and includes appropriate tests.

---


