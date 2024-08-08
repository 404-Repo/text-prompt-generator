import re
from typing import Optional

from loguru import logger
from huggingface_hub import login

import utils.io_utils as io_utils
from generator.backend_generators.groq_generator import GroqGenerator
from generator.backend_generators.vllm_generator import VLLMGenerator


class PromptGenerator:
    def __init__(self, generator_type: str):
        """

        Parameters
        ----------
        generator_type:
        """
        self._generator_type = generator_type

        if self._generator_type == "groq":
            self._generator_config = io_utils.load_config_file("../configs/groq_config.yml")

            if self._generator_config["api_key"] != "":
                self._generator = GroqGenerator(self._generator_config)
            else:
                logger.error("Groq API key was not specified.")

        elif self._generator_type == "vllm":
            self._generator_config = io_utils.load_config_file("../configs/vllm_config.yml")
            self._generator = VLLMGenerator(self._generator_config)

        else:
            raise ValueError("Unknown generator_type was specified.")

        self._pipeline_config = io_utils.load_config_file("../configs/pipeline_config.yml")
        if self._pipeline_config["hugging_face_api_key"] != "":
            login(token=self._pipeline_config["hugging_face_api_key"])
        else:
            logger.error("Hugging Face API key was not specified.")

        self._instruction_prompt = self._get_instruction_prompt()

    def load_model(self, model_name: str, quantization: Optional[str] = ""):
        """

        Parameters
        ----------
        model_name: a string with model name from HF (hugging face)
        quantization: optional parameter that defines the quantizaton of the model:
                             "awq", "gptq", "squeezellm", and "fp8" (experimental); Default value None.

        """
        if self._generator_type == "vllm":
            self._generator.preload_model(model_name, quantization)
        elif self._generator_type == "groq":
            self._generator.preload_model(model_name)
        else:
            logger.warning(f"Unknown generator type was specified: {self._generator_type}")

    def unload_model(self):
        """Function for unloading model"""
        if self._generator_type == "vllm":
            self._generator.unload_model()
        else:
            logger.warning("Model unloading needed only for VLLM pipeline.")

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
