import llama_cpp
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from time import time
import os
import tqdm
import colorama
import re
import yaml
import random
import sys


if __name__ == '__main__':
    colorama.init()

    print("*"*40)
    print("[INFO] *** Prompt Dataset Generator ***")
    print("*"*40, "\n")

    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)
        # print(config_data)

    # model to pick up from the hugging face (should have .gguf extension to run with llama)
    hf_model_repo = config_data["hugging_face_repo"]
    print(f"[INFO] Hugging Face repository: {colorama.Fore.GREEN}{hf_model_repo}{colorama.Style.RESET_ALL}")

    # the name of the file to be downloaded
    model_file_name = config_data["llm_model_file_name"]
    print(f"[INFO] LLM model to load: {colorama.Fore.GREEN}{model_file_name}{colorama.Style.RESET_ALL}")

    # cache folder where you want to store the downloaded model
    cache_folder = config_data["cache_folder"]
    os.makedirs(cache_folder, exist_ok=True)
    print(f"[INFO] LLM model will be stored here: {colorama.Fore.GREEN}{cache_folder}{colorama.Style.RESET_ALL}")

    model_path = hf_hub_download(repo_id=hf_model_repo, filename=model_file_name, cache_dir=cache_folder, local_files_only=True)
    print(f"[INFO] Downloaded model stored in: {colorama.Fore.GREEN}{model_path}{colorama.Style.RESET_ALL} \n")

    # init the llm model using Llama pipeline
    print("[INFO] Preparing model.")
    t1 = time()

    if config_data['llm_model']['seed'] < 0:
        seed = random.randint(0, sys.maxsize)
    else:
        seed = config_data['llm_model']['seed']

    llm_model = Llama(model_path=model_path,
                      seed=seed,
                      n_ctx=config_data['llm_model']['n_ctx'],
                      last_n_tokens_size=config_data['llm_model']['last_n_tokens_size'],
                      n_threads=config_data['llm_model']['n_threads'],
                      n_gpu_layers=config_data['llm_model']['n_gpu_layers'],
                      verbose=config_data['llm_model']['verbose'])
    t2 = time()
    duration = (t2-t1)/60.0
    print(f"f[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
    print("[INFO] Done.\n")

    # prompt for dataset generation
    prompt = config_data["prompt"]
    prompt = prompt.replace("prompts_num", str(config_data["prompts_num"]))

    print("[INFO] Input prompt: ")
    regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
    prompt_printing = regex.sub(r'\1\n', prompt)
    print(f"{colorama.Fore.GREEN}{prompt_printing}{colorama.Style.RESET_ALL}")

    # categories of the objects from where the objects will be piked up by the llm model
    object_categories = config_data['obj_categories']
    print(f"[INFO] Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")

    # defining the grammar for the LLM model -> forcing to output strings according to specified rules
    grammar = llama_cpp.LlamaGrammar.from_string(r'''root ::= items
                                                     items ::= item ("," ws* item)*
                                                     item ::= string
                                                     string  ::= "\"" word (ws+ word)* "\"" ws*
                                                     word ::= [a-zA-Z]+
                                                     ws ::= " "
                                                  ''', verbose=config_data['llm_model']['verbose'])

    # generate prompts using the provided object categories
    print("[INFO] Started prompt generation.")
    t3 = time()
    output_list = []
    for i in range(config_data["iteration_num"]):
        print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(0.45, 0.65)
            print(f"[INFO] Current temperature: {colorama.Fore.GREEN}{temperature}{colorama.Style.RESET_ALL}")
            # find 'member' in the input string and replace it with category
            prompt = prompt.replace("member_placeholder", category)
            output = llm_model.create_completion(prompt=prompt,
                                                 max_tokens=config_data['llm_model']['max_tokens'],
                                                 seed=seed,
                                                 echo=False,
                                                 grammar=grammar,
                                                 temperature=temperature)
            prompt = prompt.replace(category, "member_placeholder")

            # extracting the response of the llm model: generated prompts
            output_list.append(output['choices'][0]['text'])

    t4 = time()
    duration = (t4-t3)/60.0
    print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
    print("[INFO] Done.")

    # cleaning & reformating of the generated prompts
    prompt_list = []
    for l in output_list:
        split_list = l.split(",")
        prompt_list = prompt_list + split_list

    result_prompts = [l.replace("\"", "") for l in prompt_list]

    # printing the generated prompts
    print("[INFO] Generated prompts: ")

    # writing generated prompts to file
    with open(config_data["prompts_output_file"], "a") as file:
        for prompt in result_prompts:
            print(prompt)
            file.write("%s\n" % prompt)
