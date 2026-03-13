# 🧠 DrugMind

### A Collaborative Multi-Agent Framework for Automated Drug Discovery

------

## 📚 Table of Contents

- [🔎 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [📦 Repository Structure](#-repository-structure)
- [⚡ Getting Started](#-getting-started)
  - [🛠 Installation](#-installation)
  - [⬇️ Download Required Resources](#download-required-resources)
  - [🚀 Run the Framework](#-run-the-framework)
- [📊 Data and Resources](#-data-and-resources)
- [⚙️ Environment Configuration](#environment-configuration)
- [🗺 Roadmap](#-roadmap)
- [📖 Citation](#-citation)

------

# 🔎 Overview

**DrugMind** is a collaborative **multi-agent framework** designed for **automated drug discovery**.

The system integrates **large language models (LLMs)**, domain-specific computational tools, and structured memory mechanisms to orchestrate complex workflows in medicinal chemistry.

Traditional LLM-based agent systems typically rely on linear conversation histories to manage tool execution. While effective for simple tasks, this approach becomes unstable for long scientific workflows due to:

- 📉 **Context explosion** caused by large intermediate outputs
- ⚠️ **Limited deterministic workflow control**
- 🔧 **Inefficient coordination between heterogeneous scientific tools**

DrugMind addresses these challenges through a **multi-agent architecture combined with structured memory and protocol-guided orchestration**, enabling automated execution of complex pipelines such as:

- 🧬 Molecular generation
- 🧪 Molecular property evaluation
- ⚗️ Docking analysis
- 🔄 Retrosynthesis planning
- 📄 Automated report generation

The framework aims to provide a **traceable, extensible, and reproducible platform** for AI-assisted scientific research in computational chemistry.

------

# ✨ Key Features

### 🤖 Multi-Agent Collaboration

DrugMind employs multiple specialized agents that collaborate to perform complex drug discovery tasks.

------

### 🧠 Structured Memory Blackboard

Agents communicate through a shared structured memory system instead of long dialogue histories, enabling stable long-horizon reasoning.

------

### 🔗 Knowledge-Guided Tool Selection

DrugMind incorporates **DrugToolKG**, a knowledge graph that organizes relationships between scientific tools and supports intelligent tool selection.

------

### 📜 SOP-Driven Workflows

The framework uses **Standard Operating Procedures (SOPs)** to guide the execution of complex scientific processes.

------

### 🔍 Visual Traceability

All workflow steps and intermediate outputs are logged, making the discovery process **auditable and reproducible**.

------

# 📦 Repository Structure

```plaintext
DrugMind
│
├── agent/                # Core multi-agent logic
├── drugtoolkg/           # Knowledge graph for drug discovery tools
├── sop/                  # Standard operating procedures
│
├── environment/          # Environment configuration files
├── scripts/              # Utility scripts
│
├── streamlit_app.py      # Streamlit user interface
├── run.sh                # One-click startup script
│
├── tools/                # Tool implementations (Zenodo)
├── model/                # Model files (Zenodo)
├── data/                 # Dataset resources (Zenodo)
├── logs/                 # Execution logs (Zenodo)
└── services/             # Backend service modules (Zenodo)
```

### 🧠 Core Modules

**`agent/`**
Core multi-agent orchestration logic responsible for planning, tool selection, execution management, and result aggregation.

**`drugtoolkg/`**
Implements the **DrugToolKG knowledge graph**, modeling relationships between drug discovery tools.

**`sop/`**
Defines **Standard Operating Procedures** used to guide structured agent workflows.

**`environment/`**
Contains environment configuration files required to reproduce the software environment.

**`scripts/`**
Utility scripts for system inspection, knowledge graph processing, and maintenance.

------

# ⚡ Getting Started

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/BeiQI1/drugmind
cd drugmind
```

Create the conda environment:

```bash
conda create -n drugmind python=3.10
conda activate drugmind
```

Install conda dependencies:

```bash
conda install --file environment/conda-requirements.txt
```

Install pip dependencies:

```bash
pip install -r environment/pip-requirements.txt
```

Detailed environment instructions can be found in:

```
environment/README.md
```

------

# 📥 Download Required Resources

Due to repository size limitations, some essential components are hosted on **Zenodo**.

📦 Zenodo page:

https://zenodo.org/records/18973366

Download the following archives:

```
tools.rar
model.rar
data.rar
logs.rar
services.rar
```

Extract them into the project root directory.

Expected structure:

```
drugmind/
├── tools/
├── model/
├── data/
├── logs/
└── services/
```

These resources contain the toolkits, model files, datasets, and service modules required for running the complete system.

------

# 🚀 Run the Framework

DrugMind can be launched in two ways.

## Method 1 — Startup Script

```bash
./run.sh
```

------

## Method 2 — Streamlit Interface

```bash
conda activate drugmind
streamlit run streamlit_app.py
```

Once started, the **Streamlit interface** will open in your browser.

------

# 📊 Data and Resources

DrugMind relies on multiple datasets and computational tools commonly used in computational chemistry.

The Zenodo archive includes:

- 🧬 molecular generation models
- 🧪 evaluation tools
- ⚗️ docking utilities
- 🔧 service orchestration modules
- 📜 experimental logs and example data

These resources are required for running the full workflow.

------

# 🛠 Environment Configuration

All environment configuration files are located in:

```
environment/
```

This directory contains:

```
conda-requirements.txt
pip-requirements.txt
README.md
```

Dependencies are separated into **conda-based packages** and **pip-based packages** to ensure compatibility with scientific libraries such as **RDKit** and **PyTorch**.

------

# 🗺 Roadmap

Current version includes:

- 🤖 OpenAI API-based LLM integration
- 🧠 multi-agent orchestration framework
- 🔗 knowledge graph guided tool selection
- 📜 SOP-based workflow execution

Planned improvements:

- 🖥 support for **local large language models**
- 🔬 expanded **drug discovery tool ecosystem**
- ⚙️ improved **workflow optimization**
- 📊 enhanced **visual workflow tracing**

------

# 📖 Citation

If you use **DrugMind** in your research, please cite the corresponding paper.

```bibtex
@article{drugmind2026,
  title={A Collaborative Multi-Agent Framework for Automated Drug Discovery},
  author={Anonymous},
  year={2026}
}
```

------

## 🔗 Project Page

GitHub Repository

https://github.com/BeiQI1/drugmind

