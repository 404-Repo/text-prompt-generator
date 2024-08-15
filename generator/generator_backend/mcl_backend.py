import copy
import random
from time import time
from typing import List

import tqdm
from loguru import logger
from mlc_llm import MLCEngine

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class MCLBackend(BaseGeneratorBackend):
    def __init__(self, config_data: dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["mlc_api"]["max_tokens"]
        self._seed = config_data["mlc_api"]['seed']
        self._temperature = config_data["mlc_api"]["temperature"]

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

            if self._seed < 0:
                seed = random.randint(0, int(1e+5))
            else:
                seed = self._seed

            outputs = self._generator.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_in
                        }
                    ],
                    seed=seed,
                    temperature=temperature,
                    max_tokens=self._max_tokens,
                    stream=False
                )
            print(outputs.choices[0].message.content)

            prompt_in = prompt_in.replace(category, "[category_name]")
            output_prompts.append(outputs.choices[0].message.content)

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
        model_name = "HF://" + model_name
        self._generator = MLCEngine(model=model_name)

    def unload_model(self):
        """ Function for unloading the model """
        self._generator.terminate()
