import re

from huggingface_hub import login
from loguru import logger

from generator.config import GeneratorSettings, PipelineSettings
from generator.generator_backend.vllm_backend import VLLMBackend


def _initialize_huggingface(pipeline_settings: PipelineSettings) -> None:
    if pipeline_settings.hugging_face_api_key:
        login(token=pipeline_settings.hugging_face_api_key)
    else:
        raise RuntimeError("Hugging Face API key was not specified.")


def _get_instruction_prompt(pipeline_settings: PipelineSettings) -> str:
    """
    Function for pre-processing input prompt-instruction for the LLM.
    It will be used for generating prompts.

    Returns
    -------
    instruction_prompt: prompt as a string from the config file.
    """

    # prompt for dataset generation
    instruction_prompt = pipeline_settings.instruction_prompt
    instruction_prompt = instruction_prompt.replace("[prompts_number]", str(pipeline_settings.prompts_number))

    regex = re.compile(r"(?<=[^.{}])(\.)(?![{}])".format("e.g.", "e.g."))
    prompt_printing = regex.sub(r"\1\n", instruction_prompt)

    logger.info(f" Input prompt: {prompt_printing}")

    return str(instruction_prompt)


class PromptGenerator:
    def __init__(self, generator_config: GeneratorSettings, pipeline_settings: PipelineSettings) -> None:
        self._generator = VLLMBackend(generator_config)

        _initialize_huggingface(pipeline_settings)
        self._instruction_prompt = _get_instruction_prompt(pipeline_settings)

        self._obj_categories: list[str] = pipeline_settings.obj_categories

    def load_vllm_model(self, model_name: str) -> None:
        self._generator.load_vllm_model(model_name)

    def unload_vllm_model(self) -> None:
        self._generator.unload_vllm_model()

    def generate(self) -> list[str]:
        return self._generator.generate(self._instruction_prompt, self._obj_categories)
