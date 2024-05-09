import sys
import re
import copy
import random
from time import time
from typing import Optional

import tqdm
import groq
from vllm import LLM, SamplingParams
from loguru import logger
from huggingface_hub import login


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
        self._generator = None

        if self._config_data["groq_api_key"] == "":
            self._logger.warning(f"Groq Api Access Token was not specified. "
                                 f"You will not be able to use Groq API without it.")

        # login to hugging face platform using api token
        if self._config_data["hugging_face_api_key"] == "":
            self._logger.warning(f"Hugging Face Api Access Token was not specified. "
                                 f"You will not be able to download Gemma model.")
        else:
            login(token=self._config_data["hugging_face_api_key"])

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

        if self._config_data["iteration_num"] > 0:
            total_iters = range(self._config_data["iteration_num"])
        else:
            total_iters = iter(bool, True)

        output_prompts = []
        for i in enumerate(total_iters):
            self._logger.info(f"\n")
            self._logger.info(f" Iteration: {i}")
            self._logger.info(f"\n")

            prompt_in = copy.copy(prompt)

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.6)

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

    def vllm_generator(self):
        """  """

        prompt = self._load_input_prompt()
        object_categories = self._config_data['obj_categories']

        self._logger.info(f" Object categories: {object_categories}")

        # generate prompts using the provided object categories
        self._logger.info(" Started prompt generation.")
        t1 = time()

        if self._config_data["iteration_num"] > -1:
            total_iters = range(self._config_data["iteration_num"])
        else:
            total_iters = iter(bool, True)

        output_prompts = []
        for i, _ in enumerate(total_iters):
            self._logger.info(f"\n")
            self._logger.info(f" Iteration: {i}")
            self._logger.info(f"\n")

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.4, 0.6)

                # find 'member' in the input string and replace it with category
                prompt_in = prompt.replace("member_placeholder", category)
                sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=self._config_data["llm_model"]["max_tokens"])
                outputs = self._generator.generate([prompt_in], sampling_params)

                prompt = prompt.replace(category, "member_placeholder")
                output_list.append(outputs[0].outputs[0].text)

            processed_prompts = self.post_process_prompts(output_list)
            output_prompts += processed_prompts

        t2 = time()
        duration = (t2 - t1) / 60.0
        self._logger.info(f" It took: {duration} min.")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return output_prompts

    def preload_vllm_model(self, quantization: Optional[str]=None):
        self._generator = LLM(model=self._config_data["vllm_llm_model_prompt_checker"],
                              trust_remote_code=True,
                              quantization=quantization)

    @staticmethod
    def post_process_prompts(prompts_list: list):
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

                if len(line.split()) > 3:
                    if "\n" not in line:
                        line += "\n"
                    processed_lines += [line]
            result_prompts += processed_lines
        return result_prompts

    def _load_input_prompt(self):
        """ Function for loading input prompt-instruction for the LLM. It will be used for generating prompts.

        :return loaded prompt as a string from the config file.
        """

        # prompt for dataset generation
        prompt = self._config_data["prompt"]
        prompt = prompt.replace("prompts_num", str(self._config_data["prompts_num"]))

        self._logger.info(" Input prompt: ")

        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)

        self._logger.info(f"{prompt_printing}")

        return prompt
