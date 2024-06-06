if [ "$CONDA_DEFAULT_ENV" != "base" ]; then
  conda deactivate
fi

conda env remove --name three-gen-prompt-generator