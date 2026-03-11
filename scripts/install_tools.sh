#!/bin/bash

# DrugToolAgent Tool Installation Script
# This script installs the necessary dependencies for the Drug Design Agent tools into the current Conda environment.
# Target Environment: drugtoolagent (Python 3.10 recommended)

echo "Starting installation of DrugToolAgent tools..."

# 1. Core Chemistry & Biology Tools (Conda)
echo "Installing Core Chemistry & Biology Tools via Conda..."
conda install -y -c conda-forge -c bioconda \
    python=3.10 \
    rdkit \
    openbabel \
    pdbfixer \
    openmm \
    ambertools \
    biopython \
    fpocket \
    vina \
    smina \
    numpy \
    pandas \
    scipy \
    networkx \
    matplotlib \
    seaborn \
    pymol-open-source

# 2. Deep Learning Framework (PyTorch & Geometric)
# Installing PyTorch 2.1.x (Stable) with CUDA 11.8 support
echo "Installing PyTorch and Geometric Deep Learning stack..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install pytorch-lightning hydra-core wandb

# 3. Specific Tool Dependencies (Pip)
echo "Installing specific tool dependencies..."

# Docking & Analysis
conda install meeko \
    posebusters \
    plip \
    prolif \
    MDAnalysis \
    chemprop

# Generative Model Dependencies (Merged)
# Dependencies for AutoFragDiff, DecompDiff, DiffSBDD, MiDi, GenMol
conda install \
    py3dmol \
    biopandas \
    tensorboard \
    pyyaml \
    easydict \
    python-lmdb \
    mdtraj \
    scikit-learn \
    pot \
    einops \
    datasets \
    transformers \
    safe-mol \
    pytdc

pip install alphaspace2
# 4. Synthesis Tools
echo "Installing Synthesis Tools..."
# Syntheseus is a modern library for retrosynthesis planning (replacing AiZynthFinder).
pip install rdchiral
pip install syntheseus

echo "----------------------------------------------------------------"
echo "Installation Complete."
echo "Note: Some legacy tools (like DiffGui) requiring Python 3.7 might need a separate environment."
echo "Note: AlphaFold2 and ESMFold require separate large-scale setups or API keys."
echo "----------------------------------------------------------------"
