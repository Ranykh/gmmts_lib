from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gmm-ts",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gating-based Multimodal Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GMM_TS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "transformers>=4.40.0",
        "tokenizers>=0.19.0",
        "sentencepiece>=0.2.0",
        "sacremoses>=0.1.0",
        "einops>=0.8.0",
        "axial_positional_embedding>=0.3.0",
        "local-attention>=1.11.0",
        "tqdm>=4.67.0",
        "PyYAML>=6.0.0",
        "python-dateutil>=2.9.0",
        "pytz>=2025.1",
        "matplotlib>=3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)

