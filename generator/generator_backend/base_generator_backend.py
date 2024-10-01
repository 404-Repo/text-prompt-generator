from abc import ABC, abstractmethod


class BaseGeneratorBackend(ABC):
    """
    BaseGenerator is an abstract base class for defining different prompt generator backends.
    Subclasses must implement methods for generating prompts.
    """

    @abstractmethod
    def generate(self, instruction_prompt: str, object_categories: list[str]) -> list[str]:
        """
        Function that calls generator API for generating prompts.

        Parameters
        ----------
        instruction_prompt: a string with instruction prompt
        object_categories: a list of strings with categories

        Returns
        -------
        output_prompts: list with generated prompts
        """
        pass

    @abstractmethod
    def preload_model(self, model_name: str) -> None:
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face)
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Function for unloading the model"""
        pass
