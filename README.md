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
- Nvidia GPU with at least 8 GB of VRAM or more
- At least 15 GB of free space (depending on the LLM model)

### Project description
This projects consists of a set of methods for generating and post-processing the output of the LLM: 

**Tool to run:**

**prompt_generation_tool.py** - this tool allows the used to generate prompts either by using online [Groq](https://groq.com/) API
or by using offline API that relies on [llama-cpp](https://github.com/abetlen/llama-cpp-python).

#### Offline LLM requirements
 
- The requested LLM should be stored on Hugging face;
- The requested LLM should have a permission for a commercial use;
- The requested LLM should be compatible with llama-cpp input, 
i.e. stored or converted to *.gguf* file format or transformers.

The default LLM for offline mode is [mixtral-8x7b-instruct](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF).


#### Groq usage, preparation steps.

- An account should be registered at their [webpage](https://groq.com/). Log in the system of their web-page.
- To get access to their online API you will need to generate [API key](https://console.groq.com/keys). Copy this key
and paste it in the corresponding field "groq_api_key" in the launching_config.yml.
- Groq [documentation](https://console.groq.com/docs/quickstart)
- Groq supported LLM [models](https://console.groq.com/docs/models) 
- NOTE: on 25/03/2024 it is free to run Groq, but there are some [limitations](https://console.groq.com/docs/rate-limits) in place.


#### Configuration file description (launching_config.yml):
```shell
# relative path to the repository on the hugging face portal:
# model for prompt generation
llamacpp_hugging_face_repo: "TheBloke/Mixtral-8x7B-Instruct-v0.2-GGUF"
# model for prompt checking and correction
llamacpp_hugging_face_repo_prompt_checker: "google/gemma-1.1-7b-it-GGUF"


# the file with model that will be downloaded from the hugging face (note on file format: llama-cpp support .gguf only)
llamacpp_model_file_name: "mixtral-8x7b-instruct-v0.1.Q2_K.gguf"
llamacpp_model_file_name_prompt_checker: "7b_it_v1p1.gguf"

# one of the models supported by groq platform: llama2-70b-4096, mixtral-8x7b-32768, gemma-7b-it
groq_llm_model: "mixtral-8x7b-32768"

# transformers model that will be used for generating prompt dataset in offline mode
transformers_llm_model: "mistralai/Mistral-7B-Instruct-v0.2"

# the llm model that will be used for checking the quality of the prompts
transformers_llm_model_prompt_checker: "google/gemma-1.1-7b-it"

# put here token generated at https://console.groq.com/keys
groq_api_key: ""

# hugging face api token, can be generated within your account on the platform. Will be required
# for downloading gemma LLM.
hugging_face_api_key: ""

# path to where download the LLM model
cache_folder: "./model"

# the prompt that will be used for generating the dataset.
# NOTE: member_placeholder and prompts_num are mandatory placeholders.
# prompts_num will be replaced with prompts_num from this config;
# member_placeholder will be replaced with one of the strings stored in obj_categories list.
prompt: "Generate a prompt dataset for generating 3D models. 
         Each prompt should define a single 3D object that can be generated as a 3D mesh. 
         The prompt should contain one or two distinctive features such as color, shape, or pose of the generating object.  
         Each object should be different and must be strictly picked from the member_placeholder category. 
         Remove these words from prompts: clouds, river, sky, ocean, sea, wind, fields, jungles, forest, garden, water, sun, moon. 
         Generate a single unique finished prompt on the new line with no more than five or six words.
         Prompt examples: a red gorilla with green eyes, a purple parrot with orange eyes; a chair in a modern style; a laptop made from aluminium. 
         Generate prompts_num prompts. "


# Categories of objects from where the LLM model could sample the data.
obj_categories: ["animals", "furniture", "vehicles", "fantastic creatures", "weapons",
                 "buildings", "trees", "plants", "jewelry", "rocks", "gadgets", "sea creatures",
                 "lego", "instruments", "accessory", "food", "architecture"]

# Words that prompts should npt contain. Prompts with these words will be removed from the dataset and filtering stage.
filter_prompts_with_words: ["sky", "skies", "river", "ocean", "sea", "garden", "wind", "field", "terrain", "family", "tow", "city", "accessories",
                            "jungle", "forest", "space", "pool", "pond", "I", "i", "fields", "horizon", "oops", "hillside", "underwater",
                            "floor", "grass", "nature", "mist", "air", "waterfall", "music", "sunset", "sunrise", "beach", "room", "cluster", "accents",
                            "melody", "wind", "winds", "tale", "sure", "prompts", "prompt", "sunbeam", "water", "word", "words", "money", "cave", "copy",
                            "vacuum", "outdoor", "to", "us", "miami", "kidding", "time", "sunken", "point", "like", "breathing", "whoops", "labyrinth",
                            "village", "seaside", "cloud", "clouds", "exterior", "no", "unit", "harbor", "window", "grip", "island", "song", "ambiance",
                            "orbit", "hope", "melody", "animate"
                            ]

# amount of prompts to generate per category.
prompts_num: 30

# specify number of times you want to run the model (total prompt size: prompts_num x len(obj_categories) x iteration_num
iteration_num: 1

# file where to output the prompts (.txt file)
prompts_output_file: "prompts_dataset_test.txt"

# parameters for the llama-cpp loader
llm_model:
    # RNG seed, -1 for random [llama-cpp & Groq]
    seed: -1

    # text context, should correspond to little_endian (2048) and big_endian (4096) [llama-cpp]
    n_ctx: 2048

    # enable/disable extra output from llama-cpp [llama-cpp]
    verbose: false

    # Maximum number of tokens to keep in the last_n_tokens deque [llama-cpp]
    last_n_tokens_size: 128

    #  Number of threads to use for generation [llama-cpp]
    n_threads: 16

    # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded. [llama-cpp]
    n_gpu_layers: -1

    # The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
    # [llama-cpp & Groq]
    max_tokens: 1024
```

### Installing packages
```commandline
sh install_env.sh
```

### Running tool:
```commandline
python prompt_generator_tool.py --mode 'prompt_generation, groq'
```
**"mode"** option can be set to the following values:

- 'prompt_generation, groq' - running Groq API
- 'prompt_generation, transformers' - running transformers API
- 'prompt_generation, llamacpp' - running llama-cpp API
- 'grammar' - check the grammar of the generated prompts if it has not been done before
- 'filter_unique_prompts' - find and return all unique prompts within provided prompts list
- 'filter_prompts' - filter the generated prompts if it has not been done before
- 'semantic_check, groq' - checking & correcting the prompts using groq API
- 'semantic_check, transformers' - checking & correcting the prompts using transformers API
- 'semantic_check, llamacpp' - checking & correcting the prompts using llamacpp API

To preload LLM you can run the following command:
```commandline
python prompt_generator_tool.py --preload_llm 'prompt_generation, transformers'
```
**"preload_llm"** option can be set to the following values:

- 'prompt_generation, transformers' - pre-loading LLM using transformers API for prompt generation
- 'prompt_checking, transformers' - pre-loading LLM using transformers API for prompt checking
- 'prompt_generation, llamacpp' - pre-loading LLM using llamacpp API for prompt generation
- 'prompt_checking, llamacpp' - pre-loading LLM using llamacpp API for prompt checking **(Not quite working yet)**

To quantize llamacpp LLM you can run the following command:
```commandline
python prompt_generator_tool.py --quantize_llamacpp 'prompt_generation, 1'
```
**"quantize_llamacpp"** option can be set to the following values:

- 'prompt_generation, digit' - quantizing llamacpp LLM for prompt generation, digit should be 1, 2, or 3
- 'prompt_checking, digit' - quantizing llamacpp LLM for prompt checking, digit should be 1, 2, or 3
