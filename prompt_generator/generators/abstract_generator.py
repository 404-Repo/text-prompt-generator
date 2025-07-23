from abc import ABC, abstractmethod

from prompt_generator.config import GeneratorSettings, PipelineSettings
from prompt_generator.backends import VLLMBackend
import prompt_generator.utils.template as template
from prompt_generator.utils import huggingface as hf


class AbstractGenerator(ABC):
    def __init__(self, generator_config: GeneratorSettings, pipeline_settings: PipelineSettings) -> None:
        hf.initialize(pipeline_settings.hugging_face_api_key)
        self._backend = VLLMBackend(generator_config)
        self._number_of_prompts = pipeline_settings.prompts_number
        self._instruction_template = template.load(self.DEFAULT_TEMPLATE)

        if pipeline_settings.instruction_template:
            self._instruction_template = template.load(pipeline_settings.instruction_template)

    def load_model(self, model_name: str) -> None:
        self._backend.load_model(model_name)

    def unload_model(self) -> None:
        self._backend.unload_model()

    def load_next_model(self) -> None:
        self._backend.load_next_model()

    @abstractmethod
    def generate(self) -> list[str]:
        pass
