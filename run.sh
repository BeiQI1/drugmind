#!/usr/bin/env bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate drugtoolagent
streamlit run streamlit_app.py
