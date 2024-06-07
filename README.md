## Prompt-dataset Generator
#### *Questions* can be addressed to Alexander Tereshin

### Requirements:

Packages:
- miniconda or anaconda
- python 3.10
- CUDA 12.1.1 (tested)

Hardware (offline mode):

- 16 GB of RAM or more
- CPU with 8+ Cores (preferable)
- Nvidia GPU with at least 16 GB (quantized model) or 24 GB (non-quantized model) of VRAM or more 
- At least 15 GB of free space (depending on the LLM model)

### Project description
This project allows to run the server for continuous generation of the prompts according to the specified rules 
in the **launching_config.yml** file.

#### Offline LLM requirements
 
- The requested LLM should be stored on Hugging face;
- The requested LLM should have a permission for a commercial use;
- The requested LLM should be compatible with [vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html) input, 
i.e. for quantized models search models on hugging face with [AWQ](https://huggingface.co/models?sort=trending&search=awq) in the name.
- You will need to generate huggingface [API Token](https://huggingface.co/docs/hub/en/security-tokens) and store it in the provided **launching_config.yml**.

The default LLM for offline mode is [llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).


#### Groq usage, preparation steps.

- An account should be registered at their [webpage](https://groq.com/). Log in the system of their web-page.
- To get access to their online API you will need to generate [API Key](https://console.groq.com/keys) and store it in the provided **launching_config.yml**.
- Groq [documentation](https://console.groq.com/docs/quickstart)
- Groq supported LLM [models](https://console.groq.com/docs/models) 
- NOTE: on 25/03/2024 it is free to run Groq, but there are some [limitations](https://console.groq.com/docs/rate-limits) in place.

The default LLM for online mode is [llama3-8B](https://console.groq.com/docs/models).

#### Configuration file description (launching_config.yml):
```yaml
# api server parameters:
server:
    # api key to get access to the server
    api_key_prompt_server: ""

    # server address
    api_prompt_server_url: ""

    # maximum amount of retries accessing the server before continue prompt generation
    server_max_retries_number: 5

    # wait time in seconds before reattempting to send the data to the server
    server_retry_delay: 1

# parameters for running Groq API
groq_api:
    # put here token generated at https://console.groq.com/keys
    api_key: ""

    # one of the models supported by groq platform: llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it
    llm_model: "llama3-8b-8192"

    # llm model for checking prompts
    llm_model_prompt_checker: "gemma-7b-it"

    # max tokens for prompt generation
    max_tokens: 256

    # random seed
    seed: -1

# parameters for running vLLM api
vllm_api:
    # the vllm model that will be used for the prompt generation
    # awq - quantization level of the model supported by the vllm
    llm_model: "casperhansen/llama-3-8b-instruct-awq"

    # the llm model that will be used for checking the quality of the prompts
    llm_model_prompt_checker: "TechxGenus/gemma-1.1-7b-it-AWQ"

    # max tokens for prompt generation
    max_tokens: 256
    
transformers_llm_model_prompt_checker: "google/gemma-1.1-7b-it"

# hugging face api token, can be generated within your account on the platform. Will be required
# for downloading gemma LLM.
hugging_face_api_key: ""

# the prompt that will be used for generating the dataset.
# NOTE: member_placeholder and prompts_num are mandatory placeholders.
# prompts_num will be replaced with prompts_num from this config;
# member_placeholder will be replaced with one of the strings stored in obj_categories list.

prompt: "Generate a prompt dataset for generating 3D models.
         Each prompt should define a single 3D object that can be generated as a 3D mesh.
         The prompt should contain one or two distinctive features such as color, shape, or pose of the generating object.  
         Each object should be different and must be strictly picked from the member_placeholder category.
         Each prompt should be unique, on the new line, consists of between three to ten words.
         Generate a numbered list of prompts_num prompts.
        "

# Categories of objects from where the LLM model could sample the data.
obj_categories: ["humanoids", "animals", "monsters", "robots", "buildings", "nature", "vehicles", "weapons and equipments",
                 "food and drinks", "gadgets and electronics", "decorative elements", "furniture", "jewelry"]

# Words that prompts should npt contain. Prompts with these words will be removed from the dataset and filtering stage.
filter_prompts_with_words: ["sky", "skies", "river", "ocean", "sea", "garden", "wind", "field", "terrain", "family", "tow", "city", "accessories",
                            "jungle", "forest", "space", "pool", "pond", "I", "i", "fields", "horizon", "oops", "hillside", "underwater",
                            "floor", "grass", "nature", "mist", "air", "waterfall", "music", "sunset", "sunrise", "beach", "room", "cluster", "accents",
                            "melody", "wind", "winds", "tale", "sure", "prompts", "prompt", "sunbeam", "water", "word", "words", "money", "copy",
                            "vacuum", "outdoor", "to", "us", "miami", "kidding", "time", "sunken", "point", "like", "breathing", "whoops", "labyrinth",
                            "village", "seaside", "cloud", "clouds", "exterior", "no", "unit", "harbor", "window", "grip", "island", "song", "ambiance",
                            "orbit", "hope", "melody", "animate", "vagina"]

# amount of prompts to generate per category.
prompts_num: 30

# specify number of times you want to run the model (total prompt size: prompts_num x len(obj_categories) x iteration_num
# if set to -1, the prompts will be generated infinitely
iteration_num: -1

# file where to output the prompts (.txt file)
prompts_output_file: "prompt_dataset.txt"

```

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
python prompt_generator_tool.py --mode 'prompt_generation, groq'
```
**"mode"** option can be set to the following values:

- 'prompt_generation, groq' - running Groq API
- 'prompt_generation, vllm' - running vLLM API
- 'filter_unique_prompts' - find and return all unique prompts within provided prompts list
- 'filter_prompts' - filter the generated prompts if it has not been done before
- 'semantic_check, groq' - checking & correcting the prompts using groq API
- 'semantic_check, vllm' - checking & correcting the prompts using vLLM API

### Running server:

Initialising the server (locally for testing):
```commandline
python prompt_generator_server.py
```

Start generation (locally):
```commandline
curl POST http://0.0.0.0:10006/generate_prompts/
```

Initialising the server as a separate process (Runpod and similar):
```commandline
pm2 start generation.config.js
```

Start generation on Runpod:
```commandline
curl POST http://0.0.0.0:8888/generate_prompts/
```

