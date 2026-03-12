# drugmind environment

This directory contains installation files generated from the current local drugmind environment. Conda-installed packages and pip-installed packages are separated into different files.

## Files

- conda-requirements.txt: packages installed through Conda in the current environment
- pip-requirements.txt: packages installed through pip in the current environment

## Installation

All commands below are expected to be run from the project root.

### 1. Download large files not uploaded to GitHub

This repository also depends on large assets that are not included in GitHub. Download the archive from the Zenodo record below and extract it directly into the project root:

https://zenodo.org/records/18973366

After extraction, make sure the archived directories are merged into the repository root. At minimum, directories such as the following should exist in the project root:

- model/
- tools/

If the archive contains additional directories, keep the original directory structure and extract them as-is.

Example:

```bash
cd /home/public_data/ytyu/pythonproject/drugmind
unzip your_zenodo_archive.zip -d .
```

Or:

```bash
cd /home/public_data/ytyu/pythonproject/drugmind
tar -xf your_zenodo_archive.tar.gz -C .
```

### 2. Create the Conda environment

The recommended channel order is:

```bash
conda create -n drugmind -c conda-forge -c bioconda -c defaults --file environment/conda-requirements.txt
```

### 3. Activate the environment

```bash
conda activate drugmind
```

### 4. Install pip dependencies

```bash
pip install -r environment/pip-requirements.txt
```

Notes:

- pip-requirements.txt already includes the extra index URL for PyTorch CUDA 12.1 wheels
- if your machine does not use CUDA 12.1, you may need to adjust the PyTorch-related packages manually

### 5. Basic verification

```bash
python -c "import rdkit, streamlit, langchain, torch; print('environment ok')"
```

## Notes

- These files were exported from the local drugmind environment on 2026-03-12
- conda-requirements.txt and pip-requirements.txt are environment lock-style lists intended for reproducibility, not a minimal dependency set
- If the environment changes later, regenerate and replace both files