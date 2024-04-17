#!/bin/bash

# Stop the script on any error
set -e

# Check for Conda installation and initialize Conda in script
if [ -z "$(which conda)" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda env create -f environment.yml
conda activate three_gen_prompt_generator
conda info --env

python -m textblob.download_corpora

# LLAMA-CPP backends, uncomment one option and comment out the rest
# CUDA support for llama-cpp
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# If you want to use metal api uncomment this
# CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# if you have amd gpu, uncomment this
# CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# if you want to use Vulkan API, uncomment this
# CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
