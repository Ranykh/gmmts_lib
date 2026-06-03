# Third-Party Notices

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

## Dependencies

Installing this project (for example, via `pip install -r requirements.txt` or `pip install -e .`) may fetch and install third-party packages listed in `requirements.txt` and `setup.py`, including but not limited to:

- PyTorch
- Hugging Face Transformers and related libraries
- Scientific Python stack (NumPy, pandas, scikit-learn, SciPy, and others)

Pre-trained model weights may be downloaded at runtime when using LLM-based experts (for example, BERT, GPT-2). Those artifacts are subject to the license terms published by their respective providers.

Review each dependency's license before use. License texts are typically available in your Python environment under `site-packages` or from the upstream project repositories.
