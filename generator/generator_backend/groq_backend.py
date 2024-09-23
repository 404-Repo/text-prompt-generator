import secrets
import sys
from time import time

import groq
import tqdm
from loguru import logger

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class GroqBackend(BaseGeneratorBackend):
    def __init__(self, config_data: dict) -> None:
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["groq_api"]["max_tokens"]
        self._seed = config_data["groq_api"]["seed"]
        self._temperature = config_data["groq_api"]["temperature"]
        self._model_name = ""

        self._generator = groq.Groq(api_key=config_data["groq_api"]["api_key"])

    def generate(self, instruction_prompt: str, object_categories: list[str]) -> list[str]:
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
        start_time = time()
        output_prompts: list[str] = []

        for category in tqdm.tqdm(object_categories):
            temperature = secrets.SystemRandom().uniform(self._temperature[0], self._temperature[1])
            seed = secrets.randbelow(sys.maxsize) if self._seed < 0 else self._seed

            prompt = self._prepare_prompt(instruction_prompt, category)

            output = self._generate_completion(prompt, temperature, seed)

            if output.choices[0].message.content:
                output_prompts.append(output.choices[0].message.content)

        self._log_generation_time(start_time)
        return output_prompts

    def _prepare_prompt(self, instruction_prompt: str, category: str) -> str:
        return instruction_prompt.replace("[category_name]", category)

    def _generate_completion(self, prompt: str, temperature: float, seed: int) -> groq.types.chat.ChatCompletion:
        return self._generator.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_name,
            temperature=temperature,
            seed=seed,
            top_p=1,
            max_tokens=self._max_tokens,
        )

    def _log_generation_time(self, start_time: float) -> None:
        duration = (time() - start_time) / 60.0
        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info("\n")

    def preload_model(self, model_name: str) -> None:
        """
        Function for assigning one of the supported by Groq platform LLM models to the generator.

        Parameters
        ----------
        model_name: a string with model name in the format of the Groq platform
        """
        self._model_name = model_name

    def unload_model(self) -> None:
        """With groq we do not need to unload anything."""
        pass
