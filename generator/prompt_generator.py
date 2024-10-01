import re
from pathlib import Path

from huggingface_hub import login
from loguru import logger

import generator.utils.io_utils as io_utils
from generator.generator_backend.groq_backend import GroqBackend
from generator.generator_backend.vllm_backend import VLLMBackend


class PromptGenerator:
    def __init__(self, backend: str) -> None:
        """
        Parameters
        ----------
        backend: one of the supported inference engines: VLLM or Groq
        """
        current_dir = Path.cwd()
        self._generator: VLLMBackend | GroqBackend
        self._backend = backend
        self._generator_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "generator_config.yml")
        self._pipeline_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "pipeline_config.yml")

        self._initialize_backend()
        self._initialize_huggingface()
        self._instruction_prompt = self._get_instruction_prompt()

    def _initialize_backend(self) -> None:
        if self._backend == "groq":
            self._initialize_groq()
        elif self._backend == "vllm":
            self._initialize_vllm()
        else:
            raise ValueError(f"Unknown generator_type was specified: {self._backend}")

    def _initialize_groq(self) -> None:
        if self._generator_config["groq_api"]["api_key"]:
            self._generator = GroqBackend(self._generator_config)
        else:
            logger.error("Groq API key was not specified.")

    def _initialize_vllm(self) -> None:
        self._generator = VLLMBackend(self._generator_config)
        self._speculative_model = self._generator_config["vllm_api"]["speculative_model"]

    def _initialize_huggingface(self) -> None:
        if self._pipeline_config["hugging_face_api_key"]:
            login(token=self._pipeline_config["hugging_face_api_key"])
        else:
            logger.error("Hugging Face API key was not specified.")

    def load_model(self, model_name: str) -> None:
        """

        Parameters
        ----------
        model_name: a string with model name from HF (hugging face)
        """
        if isinstance(self._generator, VLLMBackend):
            self._generator.preload_model(model_name, self._speculative_model)
        else:
            self._generator.preload_model(model_name)

    def unload_model(self) -> None:
        """Function for unloading model"""
        self._generator.unload_model()

    def generate(self) -> list[str]:
        """
        Function wrapper for calling one of the backend generators.

        Returns
        -------
        prompts: list of strings with generated prompts
        """

        prompts = self._generator.generate(self._instruction_prompt, self._pipeline_config["obj_categories"])
        return prompts

    def _get_instruction_prompt(self) -> str:
        """
        Function for pre-processing input prompt-instruction for the LLM.
        It will be used for generating prompts.

        Returns
        -------
        instruction_prompt: prompt as a string from the config file.
        """

        # prompt for dataset generation
        instruction_prompt = self._pipeline_config["instruction_prompt"]
        instruction_prompt = instruction_prompt.replace(
            "[prompts_number]", str(self._pipeline_config["prompts_number"])
        )

        regex = re.compile(r"(?<=[^.{}])(\.)(?![{}])".format("e.g.", "e.g."))
        prompt_printing = regex.sub(r"\1\n", instruction_prompt)

        logger.info(f" Input prompt: {prompt_printing}")

        return str(instruction_prompt)
