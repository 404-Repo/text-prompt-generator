from loguru import logger
from huggingface_hub import login

from generator.backend_generators.groq_generator import GroqGenerator
from generator.backend_generators.vllm_generator import VLLMGenerator


class PromptGenerator:
    """ Class that implements a text prompt generator for creating prompts for generating 3D models. """
    def __init__(self, config_data: dict, generator_type: str):
        """
        :param config_data: dictionary with config data
        """
        self._generator_type = generator_type

        if generator_type == "groq":
            if config_data["groq_api"]["api_key"] != "":
                self._generator = GroqGenerator(config_data)
            else:
                logger.error("Groq API key was not specified.")
        elif generator_type == "vllm":
            if config_data["hugging_face_api_key"] != "":
                login(token=config_data["hugging_face_api_key"])
                self._generator = VLLMGenerator(config_data)
            else:
                logger.error("Hugging Face API key was not specified.")

    def load_model(self, quantization: str):
        """

        Parameters
        ----------
        quantization

        Returns
        -------

        """
        if self._generator_type == "vllm":
            self._generator.preload_vllm_model(quantization)
        else:
            logger.warning("Model preloading needed only for VLLM pipeline.")

    def unload_model(self):
        """"""
        if self._generator_type == "vllm":
            self._generator.unload_vllm_model()
        else:
            logger.warning("Model unloading needed only for VLLM pipeline.")

    def generate(self):
        """"""
        prompts = self._generator.generate()
        return prompts


