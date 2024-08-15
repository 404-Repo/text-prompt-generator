import sys
import copy
import random
from time import time
from typing import List, Dict

import tqdm
import groq
from loguru import logger

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class GroqBackend(BaseGeneratorBackend):
    def __init__(self, config_data: Dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["groq_api"]["max_tokens"]
        self._seed = config_data["groq_api"]['seed']
        self._temperature = config_data["groq_api"]["temperature"]
        self._model_name = ""

        self._generator = groq.Groq(api_key=config_data["groq_api"]["api_key"])

    def generate(self, instruction_prompt: str, object_categories: List[str]):
        """
        Function that calls Groq api for generating requested output.

        Parameters
        ----------
        instruction_prompt: a string with instruction prompt
        object_categories: a list of strings with categories

        Returns
        -------
        output_prompts: list with generated prompts
        """

        t1 = time()

        prompt_in = copy.copy(instruction_prompt)

        output_prompts = []
        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(self._temperature[0], self._temperature[1])

            # find 'member' in the input string and replace it with category
            prompt_in = prompt_in.replace("[category_name]", category)

            if self._seed < 0:
                seed = random.randint(0, sys.maxsize)
            else:
                seed = self._seed

            output = self._generator.chat.completions.create(messages=[
                {
                    "role": "user",
                    "content": prompt_in
                }
            ],
                model=self._model_name,
                temperature=temperature,
                seed=seed,
                top_p=1,
                max_tokens=self._max_tokens)

            prompt_in = prompt_in.replace(category, "[category_name]")

            # extracting the response of the llm model: generated prompts
            output_prompt = output.choices[0].message.content
            output_prompts.append(output_prompt)

        t2 = time()
        duration = (t2 - t1) / 60.0

        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info(f"\n")

        return output_prompts

    def preload_model(self, model_name: str):
        """
        Function for assigning one of the supported by Groq platform LLM models to the generator.

        Parameters
        ----------
        model_name: a string with model name in the format of the Groq platform
        """
        self._model_name = model_name

    def unload_model(self):
        """ With groq we do not need to unload anything. """
        pass
