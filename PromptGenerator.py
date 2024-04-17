import sys
import re
import os
import copy
import yaml
import colorama
import random
import tqdm
import logging

import llama_cpp
import groq
import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from time import time


def load_config_file():
    """ Function for loading parameters for running the LLM
    return loaded dictionary with data from the configuration file"""
    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def load_file_with_prompts(file_name: str):
    """ Function for loading the prompts dataset for processing.
    return list with loaded prompts
    """
    with open(file_name, "r") as file:
        prompts = [line.rstrip() for line in file]
    return prompts


def save_prompts(file_name: str, prompts_list: list, mode: str = "a"):
    """ Function for saving the prompts stored in the prompts list
    :param file_name: a string with the name of the file that will be loaded
    :param prompts_list: a list with strings (generated prompts)
    :param mode: mode for writing the file: 'a', 'w'
    """
    with open(file_name, mode) as file:
        for p in prompts_list:
            file.write("%s" % p)


class PromptGenerator:
    """ Class that implements a text prompt generator for creating prompts for generating 3D models.
    It provides access to two different LLM APIs:
    1) Groq (online) - platform that provides access to three LLM models with quick inference
    2) Offline LLM - slower than Groq but any LLM model can be plugged in that is compatible with llama-cpp
    """
    def __init__(self, config_file_data: dict):
        colorama.init()
        self._init_logger()
        self.__logger = logging.getLogger("app1")

        self.__config_data = config_file_data

        transformers.logging.set_verbosity_error()

        if self.__config_data["groq_api_key"] == "":
            self.__logger.setLevel(logging.WARNING)
            self.__logger.warning(f"{colorama.Fore.RED} Groq Api Access Token was not specified. "
                                  f"You will not be able to use Groq API without it.{colorama.Style.RESET_ALL}")
            self.__logger.setLevel(logging.INFO)

        # login to hugging face platform using api token
        if self.__config_data["hugging_face_api_key"] == "":
            self.__logger.setLevel(logging.WARNING)
            self.__logger.warning(f"{colorama.Fore.RED} Hugging Face Api Access Token was not specified. "
                                  f"You will not be able to download Gemma model.{colorama.Style.RESET_ALL}")
            self.__logger.setLevel(logging.INFO)
        else:
            login(token=self.__config_data["hugging_face_api_key"])

        self.__pipeline = None
        self.__llamacpp_model_path = ""

    """ Initializing custom logger """
    @staticmethod
    def _init_logger():
        logger = logging.getLogger("app1")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    """ Function that calls Groq api for generating requested output. All supported by Groq models are supported. """
    def groq_generator(self):
        self.__logger.info(f"\n")
        self.__logger.info("*" * 40)
        self.__logger.info(" *** Prompt Dataset Generator ***")
        self.__logger.info("*" * 40)
        self.__logger.info(f"\n")

        prompt = self._load_input_prompt()
        object_categories = self.__config_data['obj_categories']
        self.__logger.info(f" Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")

        self.__logger.info(" Started prompt generation.")

        t1 = time()
        client = groq.Groq(api_key=self.__config_data["groq_api_key"])

        output_prompts = []
        for i in range(self.__config_data["iteration_num"]):
            self.__logger.info(f"\n")
            self.__logger.info(f" Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL}")
            self.__logger.info(f"\n")

            prompt_in = copy.copy(prompt)

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.5)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt_in.replace("member_placeholder", category)

                if self.__config_data['llm_model']['seed'] < 0:
                    seed = random.randint(0, sys.maxsize)
                else:
                    seed = self.__config_data['llm_model']['seed']

                output = client.chat.completions.create(messages=[
                                                                    {
                                                                        "role": "user",
                                                                        "content": prompt_in
                                                                    }
                                                                 ],
                                                        model=self.__config_data["groq_llm_model"],
                                                        temperature=temperature,
                                                        seed=seed,
                                                        top_p=1,
                                                        max_tokens=self.__config_data["llm_model"]["max_tokens"])

                prompt_in = prompt_in.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_prompt = output.choices[0].message.content
                output_list.append(output_prompt)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

            self.__logger.info(f" Done.")
            self.__logger.info(f"\n")

        t2 = time()
        duration = (t2 - t1) / 60.0
        self.__logger.info(f" It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        return output_prompts

    """ llama-cpp loader for LLM models. LLM models should be stored in .gguf file format. """
    def llamacpp_generator(self):
        self.__logger.info(f"\n")
        self.__logger.info("*" * 40)
        self.__logger.info(" *** Prompt Dataset Generator ***")
        self.__logger.info("*" * 40)
        self.__logger.info(f"\n")

        # init the llm model using Llama pipeline
        self.__logger.info(" Preparing model.")

        if self.__config_data['llm_model']['seed'] < 0:
            seed = random.randint(0, sys.maxsize)
        else:
            seed = self.__config_data['llm_model']['seed']

        llm_model = llama_cpp.Llama(model_path=self.__llamacpp_model_path,
                                    seed=seed,
                                    n_ctx=self.__config_data['llm_model']['n_ctx'],
                                    last_n_tokens_size=self.__config_data['llm_model']['last_n_tokens_size'],
                                    n_threads=self.__config_data['llm_model']['n_threads'],
                                    n_gpu_layers=self.__config_data['llm_model']['n_gpu_layers'],
                                    verbose=self.__config_data['llm_model']['verbose'])
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        prompt = self._load_input_prompt()
        object_categories = self.__config_data['obj_categories']
        self.__logger.info(f" Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")

        # defining the grammar for the LLM model -> forcing to output strings according to specified rules
        grammar = llama_cpp.LlamaGrammar.from_string(r'''root ::= items
                                                             items ::= item ("," ws* item)*
                                                             item ::= string
                                                             string  ::= "\"" word (ws+ word)* "\"" ws*
                                                             word ::= [a-zA-Z]+
                                                             ws ::= " "
                                                          ''', verbose=self.__config_data['llm_model']['verbose'])

        # generate prompts using the provided object categories
        self.__logger.info(" Started prompt generation.")
        t1 = time()
        output_prompts = []
        for i in range(self.__config_data["iteration_num"]):
            self.__logger.info(f"\n")
            self.__logger.info(f" Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL}")
            self.__logger.info(f"\n")

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.5)

                # find 'member' in the input string and replace it with category
                prompt = prompt.replace("member_placeholder", category)
                output = llm_model.create_completion(prompt=prompt,
                                                     max_tokens=self.__config_data['llm_model']['max_tokens'],
                                                     seed=seed,
                                                     echo=False,
                                                     grammar=grammar,
                                                     temperature=temperature)
                prompt = prompt.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_prompt = output['choices'][0]['text']
                output_list.append(output_prompt)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

        t2 = time()
        duration = (t2 - t1) / 60.0
        self.__logger.info(f" It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        return output_prompts

    """ Transformers version of the pipeline for generating prompt dataset """
    def transformers_generator(self):
        model_name = self.__config_data["transformers_llm_model"]

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, load_in_8bit=False, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     trust_remote_code=True)

        self.__pipeline = transformers.pipeline("text-generation",
                                                model=model,
                                                tokenizer=tokenizer,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto")

        prompt = self._load_input_prompt()
        object_categories = self.__config_data['obj_categories']
        self.__logger.info(f" Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")

        # generate prompts using the provided object categories
        self.__logger.info(" Started prompt generation.")
        t1 = time()

        output_prompts = []
        for i in range(self.__config_data["iteration_num"]):
            self.__logger.info(f"\n")
            self.__logger.info(f" Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL}")
            self.__logger.info(f"\n")

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.5)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt.replace("member_placeholder", category)
                outputs = self.__pipeline(prompt_in,
                                          max_new_tokens=self.__config_data['llm_model']['max_tokens'],
                                          do_sample=True,
                                          temperature=temperature,
                                          top_k=1)

                prompt = prompt.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_prompt = outputs[0]['generated_text']
                output_list.append(output_prompt)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

        t2 = time()
        duration = (t2 - t1) / 60.0
        self.__logger.info(f" It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        return output_prompts

    """ Function for loading (including downloading from hugging face) the requested LLM for offline generations. """
    def llamacpp_load_checkpoint(self, local_files_only: bool = False):
        # model to pick up from the hugging face (should have .gguf extension to run with llama)
        hf_model_repo = self.__config_data["llamacpp_hugging_face_repo"]
        self.__logger.info(f" Hugging Face repository: {colorama.Fore.GREEN}{hf_model_repo}{colorama.Style.RESET_ALL}")

        # the name of the file to be downloaded
        model_file_name = self.__config_data["llamacpp_model_file_name"]
        self.__logger.info(f" LLM model to load: {colorama.Fore.GREEN}{model_file_name}{colorama.Style.RESET_ALL}")

        # cache folder where you want to store the downloaded model
        cache_folder = self.__config_data["cache_folder"]
        os.makedirs(cache_folder, exist_ok=True)
        self.__logger.info(f" LLM model will be stored here: {colorama.Fore.GREEN}{cache_folder}{colorama.Style.RESET_ALL}")

        self.__llamacpp_model_path = hf_hub_download(repo_id=hf_model_repo, filename=model_file_name, cache_dir=cache_folder, local_files_only=local_files_only)
        self.__logger.info(f" Downloaded model stored in: {colorama.Fore.GREEN}{self.__llamacpp_model_path}{colorama.Style.RESET_ALL} \n")

    """ Function for post processing of the generated prompts. The LLM output is filtered from punctuation symbols and all non alphabetic characters.
    :param prompt_list: a list with strings (generated prompts)
    :return a list with processed prompts stored as strings.
    """
    def post_process_prompts(self, prompts_list: list):
        result_prompts = []
        for el in prompts_list:
            lines = el.split(".")
            processed_lines = []
            for i in range(len(lines)):
                line = re.sub(r'[^a-zA-Z`\s-]', '', lines[i])
                line = self.make_lowercase_except_ta(line)
                split_line = re.findall(r'[A-Z][^A-Z]*', line)
                final_lines = [split_line[j] + split_line[j + 1] if j + 1 < len(split_line) and split_line[j + 1][0].islower() else split_line[j]
                               for j in range(len(split_line)) if split_line[j][0].isupper()]
                final_lines = [l + "\n" if "\n" not in l else l for l in final_lines]
                processed_lines += final_lines
            result_prompts += processed_lines
        return result_prompts

    """ Function that bring to lower case all letters except T and A; helper function for filtering. """
    @staticmethod
    def make_lowercase_except_ta(text: str):
        modified_text = ''
        for char in text:
            if char.upper() not in ['T', 'A']:
                modified_text += char.lower()
            else:
                modified_text += char
        return modified_text

    """Function for loading input prompt-instruction for the LLM. It will be used for generating prompts.
       :return loaded prompt as a string from the config file.
       """

    def _load_input_prompt(self):
        # prompt for dataset generation
        prompt = self.__config_data["prompt"]
        prompt = prompt.replace("prompts_num", str(self.__config_data["prompts_num"]))

        self.__logger.info(" Input prompt: ")

        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)

        self.__logger.info(f"{colorama.Fore.GREEN}{prompt_printing}{colorama.Style.RESET_ALL}")

        return prompt
