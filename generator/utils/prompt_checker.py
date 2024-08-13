import os
from typing import List

from loguru import logger
from huggingface_hub import login

import generator.utils.io_utils as io_utils
from generator.backend_generators.groq_generator import GroqGenerator
from generator.backend_generators.vllm_generator import VLLMGenerator


class PromptChecker:
    def __init__(self, backend: str):
        """
        Parameters
        ----------
        backend: one of the supported inference engines: VLLM or Groq
        """
        current_dir = os.getcwd()
        self._backend = backend
        generator_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                                  "configs/generator_config.yml"))

        if self._backend == "groq":
            if generator_config["groq_api"]["api_key"] != "":
                checker_config = {
                    "groq_api": {
                        "api_key": generator_config["groq_api"]["api_key"],
                        "llm_models": [""],
                        "max_tokens": 2024,
                        "temperature": [0.25, 0.4],
                        "seed": -1
                    }

                }
                self._generator = GroqGenerator(checker_config)
            else:
                logger.error("Groq API key was not specified.")

        elif self._backend == "vllm":
            checker_config = {
                "vllm_api": {
                    "llm_models": [""],
                    "max_tokens": 2024,
                    "max_model_len": 2024,
                    "temperature": [0.25, 0.4],
                    "seed": -1
                }
            }
            self._generator = VLLMGenerator(checker_config)

        else:
            raise ValueError("Unknown generator_type was specified.")

        self._pipeline_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                                       "configs/pipeline_config.yml"))
        if self._pipeline_config["hugging_face_api_key"] != "":
            login(token=self._pipeline_config["hugging_face_api_key"])
        else:
            logger.error("Hugging Face API key was not specified.")

    def check_prompts_for_completeness(self, prompts: List[str]):
        """

        Parameters
        ----------
        prompts

        Returns
        -------

        """
        instruction = "Prompts list: \n" + (", ".join(prompts) + "\n" +
                                            "Given the Prompts list, identify and complete any that are unfinished. "
                                            "Return only numbered Prompts list with corrected fully completed prompts.")
        prompts = self._generator.check_prompts_for_completeness(instruction)
        return prompts

    def load_model(self, model_name: str):
        """

        Parameters
        ----------
        model_name

        Returns
        -------

        """
        self._generator.preload_model(model_name)

    def unload_model(self):
        """Function for unloading model"""
        if self._backend == "vllm":
            self._generator.unload_model()
        else:
            logger.warning("Model unloading needed only for VLLM pipeline.")