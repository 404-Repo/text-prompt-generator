import re
import os

from loguru import logger
from huggingface_hub import login

import generator.utils.io_utils as io_utils
from generator.generator_backend.groq_backend import GroqBackend
from generator.generator_backend.vllm_backend import VLLMBackend


class PromptGenerator:
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
                self._generator = GroqBackend(generator_config)
            else:
                logger.error("Groq API key was not specified.")

        elif self._backend == "vllm":
            self._generator = VLLMBackend(generator_config)
            self._speculative_model = generator_config["vllm_api"]["speculative_model"]

        else:
            raise ValueError("Unknown generator_type was specified.")

        self._pipeline_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                                       "configs/pipeline_config.yml"))
        if self._pipeline_config["hugging_face_api_key"] != "":
            login(token=self._pipeline_config["hugging_face_api_key"])
        else:
            logger.error("Hugging Face API key was not specified.")

        self._instruction_prompt = self._get_instruction_prompt()

    def load_model(self, model_name: str):
        """

        Parameters
        ----------
        model_name: a string with model name from HF (hugging face)
        """
        if self._backend == "vllm":
            self._generator.preload_model(model_name, self._speculative_model)
        else:
            self._generator.preload_model(model_name)

    def unload_model(self):
        """Function for unloading model"""
        self._generator.unload_model()

    def generate(self):
        """
        Function wrapper for calling one of the backend generators.

        Returns
        -------
        prompts: list of strings with generated prompts
        """

        prompts = self._generator.generate(self._instruction_prompt, self._pipeline_config["obj_categories"])
        return prompts

    def _get_instruction_prompt(self):
        """
        Function for pre-processing input prompt-instruction for the LLM.
        It will be used for generating prompts.

        Returns
        -------
        instruction_prompt: prompt as a string from the config file.
        """

        # prompt for dataset generation
        instruction_prompt = self._pipeline_config["instruction_prompt"]
        instruction_prompt = instruction_prompt.replace("[prompts_number]", str(self._pipeline_config["prompts_number"]))

        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', instruction_prompt)

        logger.info(f" Input prompt: {prompt_printing}")

        return instruction_prompt
