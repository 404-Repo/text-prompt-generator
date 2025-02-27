from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl


class PromptAggregatorServiceSettings(BaseModel):
    service_url: HttpUrl | None = Field(default=None, description="service URL.")
    api_key: str | None = Field(default=None, description="`API-Key.")
    send_max_retries: int = Field(default=5, description="Maximum retries before continuing prompt generation.")
    send_retry_delay: int = Field(default=1, description="Delay (in seconds) before retrying the request.")


class ServiceSettings(BaseModel):
    generator_id: str = Field(..., description="Unique identifier for the generator")
    get_prompts_service: PromptAggregatorServiceSettings = Field(
        default_factory=lambda: PromptAggregatorServiceSettings(),
        description="service to aggregate and distribute generated prompts",
    )
    prompts_validator_service: PromptAggregatorServiceSettings = Field(
        default_factory=lambda: PromptAggregatorServiceSettings(),
        description="service to monitor speed and quality of the generated prompts",
    )
    discord_webhook_url: HttpUrl | None = Field(None, description="Discord webhook URL for server error notifications.")


class PipelineSettings(BaseModel):
    hugging_face_api_key: str | None = Field(
        None, description="Hugging Face API token for downloading LLMs that " "require extra access (string)."
    )
    instruction_prompt: str = Field(
        ...,
        description="Template for generating dataset prompts. "
        "Requires placeholders: [prompts_number] and [category_name].",
    )
    obj_categories: list[str] = Field(..., description="List of object categories used for dataset generation.")
    prompts_with_words_to_filter_out: set[str] = Field(
        ...,
        description="Words that should not appear in prompts. Prompts containing them will be filtered out.",
    )
    words_to_remove_from_prompts: set[str] = Field(
        ..., description="Words to remove from prompts before finalizing the dataset."
    )
    prepositions: set[str] = Field(..., description="Set of prepositions.")
    prompts_number: int | str = Field(
        150, description="Number of prompts to generate per category. Can be an integer or a word representation."
    )
    iterations_number: int = Field(-1, description="Number of times the model should run. -1 for infinite generation.")
    iterations_for_swapping_model: int = Field(500, description="Number of iterations before swapping the LLM model.")
    prompts_cache_file: str | None = Field(None, description="File to cache generated prompts to")


class GeneratorSettings(BaseModel):
    llm_models: list[str] = Field(default_factory=lambda: [""], description="List of vLLM models for prompt generation")
    max_tokens: int = Field(150, description="Maximum tokens for prompt generation")
    max_model_len: int = Field(1024, description="Defines the maximum length of the model output")
    temperature: tuple[float, float] = Field(
        (0.35, 0.55), description="Temperature interval [min_val, max_val] (0 to 1)"
    )
    gpu_memory_utilization: float = Field(0.9, description="GPU memory utilization for LLM")
    top_p: float = Field(0.95, description="Cumulative probability of top tokens (0,1], 1 considers all tokens")
    presence_penalty: float = Field(0.3, description="Penalty for new tokens based on appearance in generated text")
    frequency_penalty: float = Field(0.3, description="Penalty for new tokens based on frequency in generated text")
    seed: int = Field(-1, description="Random seed for sampling")
    tensor_parallel_size: int = Field(1, description="Number of GPUs used by vLLM")
    speculative_model: str | None = Field(None, description="Speculative decoding model (Optional)")
    num_speculative_tokens: int = Field(5, description="Number of speculative tokens")
    ngram_prompt_lookup_max: int = Field(4, description="Ngram-specific parameter")
    use_v2_block_manager: bool = Field(True, description="Set to True if speculative decoding is in use")
    speculative_draft_tensor_parallel_size: int = Field(1, description="Set to 1, future changes depend on development")


def load_service_settings_from_yaml(file_path: Path) -> ServiceSettings:
    with file_path.open() as f:
        config_data = yaml.safe_load(f)
    return ServiceSettings.model_validate(config_data)


def load_pipeline_settings_from_yaml(file_path: Path) -> PipelineSettings:
    with file_path.open() as f:
        config_data = yaml.safe_load(f)
    return PipelineSettings.model_validate(config_data)


def load_generator_settings_from_yaml(file_path: Path) -> GeneratorSettings:
    with file_path.open() as f:
        config_data = yaml.safe_load(f)
    return GeneratorSettings.model_validate(config_data)
