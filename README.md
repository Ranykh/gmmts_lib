# GMM-TS: Gating-based Multimodal Time Series Forecasting

A gating framework for multimodal time series forecasting that dynamically combines predictions from multiple expert models—time-series foundation models and text/LLM-based models—to improve long-term forecast accuracy.

# Overview

GMM-TS (Gating-based Multimodal Time Series Forecasting) learns a gating network that adaptively weights expert model outputs for each forecast horizon. It supports two training paradigms:

- **Online gating** — trains expert models and the gating network jointly in an end-to-end pipeline.
- **Offline gating** — combines pre-trained time-series foundation network (TSFN) and time-series foundation text (TSFT) experts via a learned gating module.

The project is built on [MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib) and includes a standalone `gmm_ts` package with gating layers, experiment runners, and example scripts for daily, weekly, and monthly forecasting tasks.

# Getting Started

```bash
# Clone the repository
git clone https://github.com/kathyrazNVDA/GMMTS.git
cd GMMTS

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Set path to MM-TSFlib data directory (required for datasets)
export MM_TSFLIB_PATH=/path/to/MM-TSFlib

# Run offline gating (example)
python run_offline_gating.py \
  --task_name mm_long_term_forecast \
  --is_training 1 \
  --pred_len 96 \
  --tsfn_experts Informer \
  --tsft_experts BERT \
  --domain Environment

# Run online gating (example)
python run_online_gating.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model Informer \
  --data custom \
  --root_path $MM_TSFLIB_PATH/data/Environment \
  --data_path NewYork_AQI_Day.csv \
  --pred_len 96 \
  --llm_model BERT

# Verify installation
python -c "from gmm_ts import GatingNet; print('GMM-TS OK')"
```

See the `examples/` directory for batch experiment scripts covering daily, weekly, and monthly forecasting configurations.

# Requirements

- OS/Arch: Linux recommended; macOS supported for development (GPU optional)
- Python: 3.8+
- Runtime: PyTorch 2.0+
- GPU/Drivers (recommended): CUDA-capable GPU with compatible NVIDIA driver
- Data: MM-TSFlib datasets via `MM_TSFLIB_PATH` environment variable

# Usage

```bash
# Offline gating with multiple experts
python run_offline_gating.py \
  --task_name mm_long_term_forecast \
  --is_training 1 \
  --pred_len 96 \
  --agg_type direct \
  --expert_input_type latent \
  --tsfn_experts Informer,Reformer \
  --tsft_experts GPT2,BERT \
  --domain Environment \
  --train_epochs 10 \
  --batch_size 32
```

- More examples/tutorials: see `examples/` for shell scripts (`run_offline_gating_*.sh`, `run_online_gating_*.sh`)
- Built on MM-TSFlib: https://github.com/AdityaLab/MM-TSFlib

# Performance (Optional)

Benchmark results depend on the expert models, domain, and prediction horizon. Use the provided example scripts to reproduce experiments across prediction lengths (48, 96, 192, 336) and domains.

## Releases & Roadmap

- Releases/Changelog: tracked via GitHub releases (see repository tags)

# Contribution Guidelines

- Start here: `CONTRIBUTING.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Development quickstart (build/test):

```bash
git clone https://github.com/kathyrazNVDA/GMMTS.git
cd GMMTS
pip install -r requirements.txt
pip install -e ".[dev]"
./print_env.sh
```

## Governance & Maintainers

- Maintainers: `MAINTAINERS.md`
- Kathy Razmadze (krazmadze@nvidia.com), Yoli Shavit (yolis@nvidia.com)
- Labeling/triage policy: use GitHub issue labels for bug, enhancement, and documentation requests

## Security

- Vulnerability disclosure: `SECURITY.md`
- Do not file public issues for security reports.

## Support

- Level: Experimental
- How to get help: [GitHub Issues](https://github.com/kathyrazNVDA/GMMTS/issues)

# Community

For questions and discussions, open a GitHub issue in this repository.

# References

- [MM-TSFlib: Multimodal Time Series Forecasting Library](https://github.com/AdityaLab/MM-TSFlib)

# License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
