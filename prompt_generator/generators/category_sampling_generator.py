from time import time
import tqdm

from prompt_generator.config import GeneratorSettings, PipelineSettings
from prompt_generator.generators import AbstractGenerator
from prompt_generator.utils.logging import log_duration
from prompt_generator.utils.prompt_filtering import postprocess_prompts


class CategorySamplingGenerator(AbstractGenerator):
    DEFAULT_TEMPLATE = "category_sampling_prompt"

    def __init__(self, generator_config: GeneratorSettings, pipeline_settings: PipelineSettings) -> None:
        super().__init__(generator_config, pipeline_settings)
        self._obj_categories: list[str] = pipeline_settings.obj_categories
        self._words_to_filter: set[str] = pipeline_settings.prompts_with_words_to_filter_out
        self._words_to_remove: set[str] = pipeline_settings.words_to_remove_from_prompts
        self._prepositions: set[str] = pipeline_settings.prepositions

    def generate(self) -> list[str]:
        start_time = time()
        output_prompts = []

        for category in tqdm.tqdm(self._obj_categories, desc="Generating prompts"):
            instruction_prompt = self._instruction_template.render(
                category=category, number_of_prompts=self._number_of_prompts
            )
            output_prompts.append(self._backend.generate(instruction_prompt))

        output_prompts = postprocess_prompts(
            output_prompts, self._words_to_filter, self._words_to_remove, self._prepositions
        )

        log_duration(f"Generated {len()} prompts in", start_time)

        return output_prompts
