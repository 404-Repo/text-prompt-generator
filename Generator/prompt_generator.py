import sys
import re
import copy
import yaml
import random
from time import time

import tqdm
import torch
import groq
import transformers

from vllm import LLM, SamplingParams
from loguru import logger
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig)
from langchain_community.llms import VLLM
from huggingface_hub import login


def load_config_file():
    """ Function for loading parameters for running the LLM
    return loaded dictionary with data from the configuration file"""
    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def load_file_with_prompts(file_name: str):
    """ Function for loading the prompts dataset for processing.

    :param return: list with loaded prompts
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
    def __init__(self, config_file_data: dict, logger_: logger):
        """

        :param config_file_data:
        :param logger_:
        """
        self._config_data = config_file_data
        self._logger = logger_

        transformers.logging.set_verbosity_error()

        if self._config_data["groq_api_key"] == "":
            self._logger.warning(f"Groq Api Access Token was not specified. "
                                 f"You will not be able to use Groq API without it.")

        if self._config_data["openai_api_key"] == "":
            self._logger.warning(f"OpenAI Api Access Token was not specified. "
                                 f"You will not be able to use OpenAI API without it.")

        # login to hugging face platform using api token
        if self._config_data["hugging_face_api_key"] == "":
            self._logger.warning(f"Hugging Face Api Access Token was not specified. "
                                 f"You will not be able to download Gemma model.")
        else:
            login(token=self._config_data["hugging_face_api_key"])

        self._pipeline = None
        self._llamacpp_model_path = ""

    def groq_generator(self):
        """ Function that calls Groq api for generating requested output. All supported by Groq models are supported. """
        self._logger.info(f"\n")
        self._logger.info("*" * 40)
        self._logger.info(" *** Prompt Dataset Generator ***")
        self._logger.info("*" * 40)
        self._logger.info(f"\n")

        prompt = self._load_input_prompt()
        object_categories = self._config_data['obj_categories']
        self._logger.info(f" Object categories: {object_categories}")
        self._logger.info(" Started prompt generation.")

        t1 = time()
        client = groq.Groq(api_key=self._config_data["groq_api_key"])

        output_prompts = []
        for i in range(self._config_data["iteration_num"]):
            self._logger.info(f"\n")
            self._logger.info(f" Iteration: {i}")
            self._logger.info(f"\n")

            prompt_in = copy.copy(prompt)

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.5)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt_in.replace("member_placeholder", category)

                if self._config_data['llm_model']['seed'] < 0:
                    seed = random.randint(0, sys.maxsize)
                else:
                    seed = self._config_data['llm_model']['seed']

                output = client.chat.completions.create(messages=[
                                                                    {
                                                                        "role": "user",
                                                                        "content": prompt_in
                                                                    }
                                                                 ],
                                                        model=self._config_data["groq_llm_model"],
                                                        temperature=temperature,
                                                        seed=seed,
                                                        top_p=1,
                                                        max_tokens=self._config_data["llm_model"]["max_tokens"])

                prompt_in = prompt_in.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_prompt = output.choices[0].message.content
                output_list.append(output_prompt)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

            self._logger.info(f" Done.")
            self._logger.info(f"\n")

        t2 = time()
        duration = (t2 - t1) / 60.0
        self._logger.info(f" It took: {duration} min.")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return output_prompts

    def transformers_generator(self):
        """ Transformers version of the pipeline for generating prompt dataset """

        assert self._pipeline is not None

        prompt = self._load_input_prompt()
        object_categories = self._config_data['obj_categories']
        self._logger.info(f" Object categories: {object_categories}")

        # generate prompts using the provided object categories
        self._logger.info(" Started prompt generation.")
        t1 = time()

        output_prompts = []
        for i in range(self._config_data["iteration_num"]):
            self._logger.info(f"\n")
            self._logger.info(f" Iteration: {i}")
            self._logger.info(f"\n")

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.5)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt.replace("member_placeholder", category)
                outputs = self._pipeline(prompt_in,
                                         max_new_tokens=self._config_data['llm_model']['max_tokens'],
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
        self._logger.info(f" It took: {duration} min.")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return output_prompts

    def transformers_load_checkpoint(self, load_in_4bit: bool = True,
                                     load_in_8bit: bool = False,
                                     bnb_4bit_quant_type: str = "nf4",
                                     bnb_4bit_use_double_quant: bool = True):
        """ Function for pre-loading checkpoints for the requested models using transformers.

        :param load_in_4bit: a boolean parameter that controls whether the model will be loaded using 4 bit quantization (VRAM used ~ 9 Gb).
        :param load_in_8bit: a boolean parameter that controls whether the model will be loaded using 8 bit quantization (VRAM used ~ 18 Gb).
        :param bnb_4bit_quant_type: string parameter that defines the quantization type for 4 bit quantization
        :param bnb_4bit_use_double_quant: boolean parameter that defines whether to use or not double quantization
        """

        if load_in_4bit:
            load_in_8bit = False
        elif load_in_8bit:
            load_in_4bit = False
        else:
            load_in_4bit = True
            load_in_8bit = False

        model_name = self._config_data["transformers_llm_model"]
        bnb_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit,
                                        load_in_8bit=load_in_8bit,
                                        bnb_4bit_quant_type=bnb_4bit_quant_type,
                                        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     quantization_config=bnb_config,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     trust_remote_code=True)

        self._pipeline = transformers.pipeline("text-generation",
                                               model=model,
                                               tokenizer=tokenizer,
                                               torch_dtype=torch.bfloat16,
                                               device_map="auto")

    def vllm_generator(self):
        """  """
        generator = VLLM(model=self._config_data["vllm_llm_model"],
                         trust_remote_code=True,
                         max_new_tokens=self._config_data["llm_model"]["max_tokens"])

        prompt = self._load_input_prompt()
        object_categories = self._config_data['obj_categories']

        self._logger.info(f" Object categories: {object_categories}")

        # generate prompts using the provided object categories
        self._logger.info(" Started prompt generation.")
        t1 = time()

        output_prompts = []
        for i in range(self._config_data["iteration_num"]):
            self._logger.info(f"\n")
            self._logger.info(f" Iteration: {i}")
            self._logger.info(f"\n")

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.6)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt.replace("member_placeholder", category)

                outputs = generator.invoke(prompt_in,
                                           top_k=1,
                                           top_p=1,
                                           temperature=temperature)

                prompt = prompt.replace(category, "member_placeholder")
                output_list.append(outputs)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

        t2 = time()
        duration = (t2 - t1) / 60.0
        self._logger.info(f" It took: {duration} min.")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return output_prompts

    def post_process_prompts(self, prompts_list: list):
        """ Function for post processing of the generated prompts. The LLM output is filtered from punctuation symbols and all non alphabetic characters.

        :param prompts_list: a list with strings (generated prompts)
        :return a list with processed prompts stored as strings.
        """
        result_prompts = []
        for el in prompts_list:
            lines = el.split("\n")
            processed_lines = []
            for i in range(len(lines)):
                line = re.sub(r'[^a-zA-Z`\s-]', '', lines[i])
                line = re.sub(r'\d+', '', line)
                line = line.replace(".", "")

                # line = self.make_lowercase_except_ta(line)
                # split_line = re.findall(r'[A-Z][^A-Z]*', line)
                # final_lines = [split_line[j] + split_line[j + 1] if j + 1 < len(split_line) and split_line[j + 1][0].islower() else split_line[j]
                #                for j in range(len(split_line)) if split_line[j][0].isupper()]
                # final_lines = [l + "\n" if "\n" not in l else l for l in final_lines]
                # processed_lines += final_lines

                if len(line.split()) > 3:
                    if "\n" not in line:
                        line += "\n"
                    processed_lines += [line]
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
        prompt = self._config_data["prompt"]
        prompt = prompt.replace("prompts_num", str(self._config_data["prompts_num"]))

        self._logger.info(" Input prompt: ")

        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)

        self._logger.info(f"{prompt_printing}")

        return prompt
