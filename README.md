## Prompt-dataset Generator
#### *Questions* can be addressed to Alexander Tereshin

### Requirements:

Packages:
- miniconda or anaconda
- python 3.10
- CUDA 12.1.1 (tested)

Hardware:

- 16 GB of RAM or more
- CPU with 8+ Cores (preferable)
- Nvidia GPU with at least 8 GB of VRAM or more
- At least 15 GB of free space (depending on the LLM model)

### Project description
This projects consists of two tools: 

- **prompt_generation_tool.py** - this tool allows to download and use LLM model compatible with [llama-cpp](https://github.com/abetlen/llama-cpp-python). 
The downloaded LLM model will be used for generating prompts datasets according to the providded instructions.
The default model is [mixtral-8x7b-instruct](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF).
- **filter_prompt_dataset_tool.py** - this tool allows to loop through the generated dataset and filter out all duplicates as well as
text prompts with specified words that should be give in the **launching_config.yml** file.

#### Configuration file description (launching_config.yml):
```shell
# relative path to the repository on the hugging face portal
hugging_face_repo: "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"

# the file with model that will be downloaded from the hugging face (note on file format: llama-cpp support .gguf only)
llm_model_file_name: "mixtral-8x7b-instruct-v0.1.Q2_K.gguf"

# path to where download the LLM model
cache_folder: "./model"

# the prompt that will be used for generating the dataset.
# NOTE: member_placeholder and prompts_num are mandatory placeholders.
# prompts_num will be replaced with prompts_num from this config;
# member_placeholder will be replaced with one of the strings stored in obj_categories list.
prompt: "Generate a prompt dataset for generating 3D models. 
         Each prompt should be meaningful and define a single 3D object that can be generated in 3D with no more than 5 words. 
         The object description should contain one or two distinctive features such as color, shape, or pose of the generating object. 
         Make all prompts simple, e.g. a purple parrot with orange eyes; a chair in a modern style; a laptop made from aluminium.  
         Remove these words from prompts: clouds, river, sky, ocean, sea, wind, fields, jungles, forest, garden.
         Every prompt must be unique and remove repetitions.
         Each object should be different and must be strictly picked up from the member_placeholder category. 
         Generate prompts_num prompts. "

# Categories of objects from where the LLM model could sample the data.
obj_categories: ["animals", "furniture", "cars", "fantasy creatures", "weapons", "buildings", "trees", "plants", "jewelry", "rocks", "gadgets"]

# Words that prompts should npt contain. Prompts with these words will be removed from the dataset and filtering stage.
filter_prompts_with_words: ["sky", "river", "ocean", "sea", "garden", "wind", "field", "jungle", "forest", "space", "pool", "pond", "I", "fields", "floor", "grass"]

# prompts with colours which will be filtered out if the prompt is as follows: "green", "black" etc.
filter_colors: ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey"]

# amount of prompts to generate per category.
prompts_num: 10

# specify number of times you want to run the model (total prompt size: prompts_num x len(obj_categories) x iteration_num
iteration_num: 1

# file where to output the prompts (.txt file)
prompts_output_file: "prompts_dataset.txt"

# parameters for the llama-cpp loader
llm_model:
    # RNG seed, -1 for random
    seed: -1

    # text context, should correspond to little_endian (2048) and big_endian (4096)
    n_ctx: 2048

    # enable/disable extra output from llama-cpp
    verbose: false

    # Maximum number of tokens to keep in the last_n_tokens deque
    last_n_tokens_size: 128

    #  Number of threads to use for generation
    n_threads: 16

    # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
    n_gpu_layers: 35

    # The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
    max_tokens: 1024
```

### Running tools:
```commandline
python prompt_generator_tool.py
python filter_prompt_dataset_tool.py
```
