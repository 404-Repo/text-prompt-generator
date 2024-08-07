import sys
import copy
import random
from time import time

import tqdm
import groq
from loguru import logger


class GroqGenerator:
    def __init__(self, config_data: dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._instruction_prompt = config_data["prompt"]
        self._object_categories = config_data["obj_categories"]
        self._max_tokens = config_data["groq_api"]["max_tokens"]
        self._seed = config_data["groq_api"]['seed']
        self._model_name = config_data["groq_api"]["llm_model"]
        self._temperature = [0.25, 0.6]

        self._generator = groq.Groq(api_key=config_data["groq_api"]["api_key"])

    def groq_generator(self):
        """
        Function that calls Groq api for generating requested output. All supported by Groq models are supported.

        Returns
        -------
        output_prompts: list with generated prompts
        """

        t1 = time()

        prompt_in = copy.copy(self._instruction_prompt)

        output_prompts = []
        for category, _ in zip(self._object_categories, tqdm.trange(len(self._object_categories))):
            temperature = random.uniform(self._temperature[0], self._temperature[1])

            # find 'member' in the input string and replace it with category
            prompt_in = prompt_in.replace("member_placeholder", category)

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

            prompt_in = prompt_in.replace(category, "member_placeholder")

            # extracting the response of the llm model: generated prompts
            output_prompt = output.choices[0].message.content
            output_prompts.append(output_prompt)

        t2 = time()
        duration = (t2 - t1) / 60.0

        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info(f"\n")

        return output_prompts
