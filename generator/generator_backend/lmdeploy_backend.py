import gc
import copy
import random
import os
from time import time
from typing import List

import torch
import tqdm
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, turbomind
from loguru import logger

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class LmdeployBackend(BaseGeneratorBackend):
    def __init__(self, config_data: dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["lmdeploy_api"]["max_tokens"]
        self._seed = config_data["lmdeploy_api"]['seed']
        self._temperature = config_data["lmdeploy_api"]["temperature"]

        self._generator = None

    def generate(self, instruction_prompt: str,  object_categories: List[str]):
        """
        Function that calls vLLM API for generating prompts.

        Parameters
        ----------
        instruction_prompt: a string with instruction prompt
        object_categories: a list of strings with categories

        Returns
        -------
        output_prompts: list with generated prompts
        """

        # generate prompts using the provided object categories
        prompt_in = copy.copy(instruction_prompt)
        t1 = time()

        output_prompts = []
        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(self._temperature[0], self._temperature[1])

            # find 'member' in the input string and replace it with category
            prompt_in = prompt_in.replace("[category_name]", category)
            prompt_in += " Do not use words: design, build, create, produce, develop, generate, make in prompts."

            prompt =[
                {"role": "system", "content": "You are a helpful assistant."},
                {'role': 'user', 'content': prompt_in}]

            if self._seed < 0:
                seed = random.randint(0, int(1e+5))
            else:
                seed = self._seed

            gen_config = GenerationConfig(temperature=temperature,
                                          max_new_tokens=self._max_tokens,
                                          random_seed=seed,
                                          bad_words=["design", "build", "create", "produce", "develop", "generate",
                                                     "make"])

            outputs = self._generator(prompt, gen_config=gen_config)
            print(outputs.text)

            prompt_in = prompt_in.replace(category, "[category_name]")
            output_prompts.append(outputs.text)

        t2 = time()
        duration = (t2 - t1) / 60.0
        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info(f"\n")

        return output_prompts

    def preload_model(self, model_name: str):
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face)
        """
        backend_config = TurbomindEngineConfig(cache_max_entry_count=0.8)
        self._generator = pipeline(model_name, backend_config=backend_config)

    def unload_model(self):
        """ Function for unloading the model """
        pass
