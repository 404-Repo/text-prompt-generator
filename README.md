## Prompt-dataset Generator
#### *Questions* can be addressed to Alexander Tereshin

### Requirements:

Packages:
- [miniconda](https://docs.anaconda.com/miniconda/) or [anaconda](https://www.anaconda.com/)
- [pm2](https://pm2.keymetrics.io/)

Hardware:

- 32 GB of RAM or more
- CPU with 8+ Cores (preferable)
- Nvidia GPU with at least 48 GB of VRAM or mode (models with 7-14 B of parameters) 
- At least 100 GB of free space (depending on the LLM model, if you are using 1-2 LLMs at the same time)

### Project description:
Prompt generator is based on vLLM library. Prompt generator supplies the subnet with synthetically generated prompts. 
It uses STAR (STate of the ARt) LLMs (Large Language Models) for generating prompts. 

#### LLM requirements:
 
- The requested LLM should be stored on Hugging face;
- The requested LLM should have a permission for a commercial use;
- The requested LLM should be compatible with [vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html) inference engine;
- You might need to generate huggingface [API Token](https://huggingface.co/docs/hub/en/security-tokens) and store it in the provided **./configs/pipeline_config.yml** if 
the model that you are going to use required granting of extra access;

<span style="color:orange">**NOTE**: Models that have around 7-8 billion parameters can be run on GPUs with 24 GB VRAM without any problems.
Models that have around 10-25 Billion parameters can be run on GPUs with 48 GB VRAM 
(You might came across some exceptions, e.g. quantized model llama3-70B-AWQ can be run on 48 GB VRAM GPU)</span>

#### Configuration files:

In **./configs** folder you can find three configuration files:
- *generator_config.yml* - configuration file with parameters for inferencing LLMs with vLLM backend; 
- *pipeline_config.yml* - configuration file with parameters for controlling the prompt generation, e.g. instruction prompt, prompt filtering, etc.; 
- *service_config.yml* - configuration file with parameters for setting up access to "get_prompts" server for further distribution to the subnet;

### Installing packages:

For installing Conda environment only (**without** installing mini-conda or anaconda):
```commandline
cd installation_scripts
bash install_env.sh
```

For [Runpod](https://www.runpod.io/) platform or [Debian](https://www.debian.org/)-like systems run the following commands:
```commandline
cd installation_scripts
bash install_runpod_env.sh
```
It will install miniconda, pm2, nano in your Linux system. Reopen your console and execute the following:
```commandline
bash install_env.sh
```

For uninstalling created conda environment with all packages run the following command:
```commandline
cd installation_scripts
bash cleanup_env.sh
```

**install_env.sh** will generate **generation.config.js** in the project root directory that can be used with pm2 process.

### Running tool:
Go to the **"configs"** folder:
1. Open **"generator_config.yml"**. Add your models to the list "llm_models: ["hf_model1", "hf_model2"]";
2. **[OPTIONAL]** Open **"pipeline_config.yml"**:
   - Edit **"hugging_face_api_key: "** (if needed) and add hugging face token;
   - Edit **"iterations_number: "** (if needed) (-1 infinite generation, any positive integer will define for how long to generate);
   - Edit **"iterations_for_swapping_model: "** (if needed) to set up the minimum amount of iterations after which the model will be changed to another one;
   - Edit **"prompts_cache_file: "** (if needed) to set up a path to the txt file where prompts will be additionally cached;
3. **[OPTIONAL]** Open **"service_config.yml"** (if you run generator for distributing prompts):
   - Edit **"generator_id: "** to set up the id for the current generator;
   - Edit **"get_prompts_service_url: "** to set up URL for the service that will receive sent prompts;
   - Edit **"get_prompts_api_key: "** api-key for getting access to remote server;
   - Edit **"get_prompts_send_max_retries: "** to set up maximum amount of retries to send prompts to the `get-prompts` service;
   - Edit **"get_prompts_send_retry_delay: "** to set up the waiting time in seconds before reattempting to send prompts to the `get-prompts` service;
   - Edit **"discord_webhook_url: "** `[optional]` to set up the discord hook where errors from the server will be posted;
4. Run prompt generator as follows:
```commandline
conda activate three-gen-prompt-generator
pm2 run generation.config.js
```