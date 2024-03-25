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
- The requested LLM should be compatible with llama-cpp input, i.e. stored or converted to *.gguf* file format.

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
# relative path to the repository on the hugging face portal
hugging_face_repo: "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"

# the file with model that will be downloaded from the hugging face (note on file format: llama-cpp support .gguf only)
llm_model_file_name: "mixtral-8x7b-instruct-v0.1.Q2_K.gguf"

# one of the models supported by groq platform: llama2-70b-4096, mixtral-8x7b-32768
groq_llm_model: "mixtral-8x7b-32768"

# put here token generated at https://console.groq.com/keys
groq_api_key: "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"

# path to where download the LLM model
cache_folder: "./model"

# the prompt that will be used for generating the dataset.
# NOTE: member_placeholder and prompts_num are mandatory placeholders.
# prompts_num will be replaced with prompts_num from this config;
# member_placeholder will be replaced with one of the strings stored in obj_categories list.
prompt: "Generate a prompt dataset for generating 3D models. 
         Each prompt should be meaningful and define a single 3D object that can be generated in 3D with no more than five words. 
         The object description should contain one or two distinctive features such as color, shape, or pose of the generating object. 
         Make all prompts simple, e.g. a purple parrot with orange eyes; a chair in a modern style; a laptop made from aluminium.  
         Remove these words from prompts: clouds, river, sky, ocean, sea, wind, fields, jungles, forest, garden.
         Every prompt must be unique and remove repetitions.
         Each object should be different and must be strictly picked up from the member_placeholder category. 
         Generate prompts_num prompts. "

prompt_groq: "Generate a prompt dataset for generating 3D models. 
              Each prompt should define a single 3D object that can be generated as a 3D mesh. 
              The prompt should contain one or two distinctive features such as color, shape, or pose of the generating object.  
              Each object should be different and must be strictly picked from the member_placeholder category. 
              Remove these words from prompts: clouds, river, sky, ocean, sea, wind, fields, jungles, forest, garden, water, sun, moon. 
              Generate a single unique prompt on the new line with no more than five words.
              Prompt examples: a red gorilla with green eyes, a purple parrot with orange eyes; a chair in a modern style; a laptop made from aluminium. 
              Generate prompts_num prompts. "


# Categories of objects from where the LLM model could sample the data.
obj_categories: ["animals", "furniture", "cars", "fantastic creatures", "weapons",
                 "buildings", "trees", "plants", "jewelry", "rocks", "gadgets", "sea creatures",
                 "lego", "instruments"]

# Words that prompts should npt contain. Prompts with these words will be removed from the dataset and filtering stage.
filter_prompts_with_words: ["sky", "skies", "river", "ocean", "sea", "garden", "wind", "field", "terrain", "family", "tow", "city",
                            "jungle", "forest", "space", "pool", "pond", "I", "fields", "horizon", "oops", "hillside", "underwater",
                            "floor", "grass", "nature", "mist", "air", "waterfall", "music", "sunset", "sunrise", "beach", "room",
                            "melody", "wind", "winds", "tale", "sure", "prompts", "prompt", "sunbeam", "water", "word", "words", "money",
                            "vacuum"]

# prompts with colours which will be filtered out if the prompt is as follows: "green", "black" etc.
filter_colors: ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey"]

# amount of prompts to generate per category.
prompts_num: 30

# specify number of times you want to run the model (total prompt size: prompts_num x len(obj_categories) x iteration_num
iteration_num: 1

# file where to output the prompts (.txt file)
prompts_output_file: "prompts_dataset_mixtral_test.txt"

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
    n_gpu_layers: 35

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
python prompt_generator_tool.py --mode online
```
"mode" option can be set to the following values:

- 'online'  - running Groq API
- 'offline' - running llama-cpp API
- 'grammar' - check the grammar of the generated prompts if it has not been done before
- 'filter'  - filter the generated prompts if it has not been done before
