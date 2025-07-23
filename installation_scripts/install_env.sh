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
conda env create -f ../environment.yml
conda activate three-gen-prompt-generator
conda info --env

CUDA_HOME=${CONDA_PREFIX}
mkdir tmp
wget -O ./tmp/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl "https://github.com/404-Repo/compiled_libs/releases/download/flash-attn-cu126-torch270/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl"
pip install ./tmp/flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl
rm -rf tmp


# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > ../generation.config.js
module.exports = {
  apps : [{
    name: 'prompts_generator',
    script: '-m prompt_generator.run',
    interpreter: '${CONDA_INTERPRETER_PATH}',
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."
