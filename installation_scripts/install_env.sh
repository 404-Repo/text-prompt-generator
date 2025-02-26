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
conda activate three-gen-prompt-generator
conda info --env

CUDA_HOME=${CONDA_PREFIX}
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install flashinfer-python==0.2.0.post2 -i https://flashinfer.ai/whl/cu121/torch2.5

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > ../generation.config.js
module.exports = {
  apps : [{
    name: 'prompts_generator',
    script: 'run.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."
