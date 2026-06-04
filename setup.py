from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gmm-ts",
    version="0.1.0",
    author="Kathy Razmadze and Yoli Shavit",
    author_email="krazmadze@nvidia.com, yolis@nvidia.com",
    description="Gating-based Multimodal Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/gmmts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.12.0",
        "numpy>=1.26.4,<2.0.0",
        "pandas>=2.3.3,<3.0.0",
        "scipy>=1.15.3",
        "scikit-learn>=1.7.2",
        "transformers>=4.57.6,<5.0.0",
        "tokenizers>=0.23.1",
        "sentencepiece>=0.2.1",
        "sacremoses>=0.1.1",
        "einops>=0.8.2",
        "axial_positional_embedding>=0.3.12",
        "local-attention>=1.11.2",
        "reformer-pytorch>=1.4.4",
        "product-key-memory>=0.3.0",
        "sktime>=0.40.1",
        "tqdm>=4.67.3",
        "PyYAML>=6.0.3",
        "python-dateutil>=2.9.0.post0",
        "pytz>=2025.2",
        "matplotlib>=3.10.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)

