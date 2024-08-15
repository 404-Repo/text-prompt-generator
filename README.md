## Prompt-dataset Generator
#### *Questions* can be addressed to Alexander Tereshin

### Requirements:

Packages:
- miniconda or anaconda
- python 3.10
- CUDA 12.1.1 (tested)

Hardware (offline mode):

- 32 GB of RAM or more
- CPU with 8+ Cores (preferable)
- Nvidia GPU with at least 16 GB (quantized model) or 24 GB (non-quantized model) of VRAM or more 
- At least 40 GB of free space (depending on the LLM model)

### Project description
This project allows to run the server for continuous generation of the prompts according to the specified rules 
in the **launching_config.yml** file.

#### Offline LLM requirements
 
- The requested LLM should be stored on Hugging face;
- The requested LLM should have a permission for a commercial use;
- The requested LLM should be compatible with [vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html) input, 
i.e. for quantized models search models on hugging face with [AWQ](https://huggingface.co/models?sort=trending&search=awq) in the name.
- You will need to generate huggingface [API Token](https://huggingface.co/docs/hub/en/security-tokens) and store it in the provided **./configs/pipeline_config.yml**.

<span style="color:orange">**NOTE**: Models that have around 7-8 billion parameters can be run on GPUs with 24 GB VRAM without any problems.
Models that have around 10-25 Billion parameters can be run on GPUs with 48 GB VRAM 
(You might came across some exceptions, e.g. llama3-70B-AWQ can be run on 48 GB VRAM GPU)</span>


#### Groq usage, preparation steps.

- An account should be registered at their [webpage](https://groq.com/). Log in the system of their web-page.
- To get access to their online API you will need to generate [API Key](https://console.groq.com/keys) and store it in the provided **./configs/generator_config.yml**.
- Groq [documentation](https://console.groq.com/docs/quickstart)
- Groq supported LLM [models](https://console.groq.com/docs/models) 
- NOTE: it is free to run Groq, but there are some [limitations](https://console.groq.com/docs/rate-limits) in place.

The default LLM for online mode is [llama3-8B](https://console.groq.com/docs/models).

#### Configuration files:

In **./configs** folder you can find three configuration files:
- *generation_config.yml* - configuration file with parameters for running supported generators backends. Currently they are vLLM, Groq. 
- *pipeline_config.yml* - configuration file with parameters for controlling the prompt generation, e.g. instruction prompt, prompt filtering, etc. 
- *server_config.yml* - configuration file with parameters for setting up remote access to the server that will accept generated prompts.

### Installing packages

For installing Conda environment only:
```commandline
cd installation_scripts
bash install_env.sh
```

For [Runpod](https://www.runpod.io/) platform run the following commands:
```commandline
cd installation_scripts
bash install_runpod_env.sh
bash install_env.sh
```

For cleaning up the conda environment run the following command:
```commandline
cd installation_scripts
bash cleanup_env.sh
```

**install_env.sh** will generate **generation.config.js** in the project root directory that can be used with pm2 process.

### Running tool:
```commandline
python prompt_generator_tool.py --mode 'prompt_generation, vllm'
```
**"mode"** option can be set to the following values:

- 'prompt_generation, groq' - running Groq API
- 'prompt_generation, vllm' - running vLLM API
- 'filter_unique_prompts' - find and return all unique prompts within provided prompts list
- 'filter_prompts' - filter the generated prompts if it has not been done before
- 'preload_llms, vllm'

### Running server:

Initialising the server (locally for testing):
```commandline
python prompt_generator_server.py
```

Initialising the server as a separate process (Runpod and similar):
```commandline
pm2 start generation.config.js
```

Start generation (Port 10006 is default, but you can change it in *generation.config.js* 
and then you will need to amend http://0.0.0.0:10006/ it here accordingly):
```commandline
curl -d "inference_api=vllm" -d "save_locally_only=False" -X POST http://0.0.0.0:10006/generate_prompts/
```
