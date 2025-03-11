import gc
import json
import time
from pathlib import Path

import requests
import torch
from generator.config import (
    GeneratorSettings,
    PipelineSettings,
    PromptAggregatorServiceSettings,
    ServiceSettings,
    load_generator_settings_from_yaml,
    load_pipeline_settings_from_yaml,
    load_service_settings_from_yaml,
)
from generator.prompt_generator import PromptGenerator
from generator.utils.discord import post_exception_to_discord, send_discord_message
from generator.utils.prompts_filtering_utils import (
    correct_non_finished_prompts,
    filter_prompts_with_words,
    filter_unique_prompts,
    post_process_generated_prompts,
    remove_words_from_prompts,
)
from loguru import logger


def main() -> None:
    service_settings, pipeline_settings, generator_settings = load_settings()
    generator = PromptGenerator(generator_settings, pipeline_settings)

    if service_settings.discord_webhook_url:
        send_discord_message(
            message=f"🚨 **Prompt Generator {service_settings.generator_id}** 🚨\n\n ONLINE",
            webhook_url=service_settings.discord_webhook_url,
        )

    try:
        generate(generator, service_settings, pipeline_settings, generator_settings.llm_models)
    except Exception as e:
        if service_settings.discord_webhook_url:
            post_exception_to_discord(e, service_settings.generator_id, service_settings.discord_webhook_url)

        logger.exception(e)
        cleanup(generator)
        raise e

    cleanup(generator)
    logger.info("Done.")


def cleanup(generator: PromptGenerator) -> None:
    generator.unload_vllm_model()
    gc.collect()
    torch.cuda.empty_cache()


def load_settings() -> tuple[ServiceSettings, PipelineSettings, GeneratorSettings]:
    current_dir = Path.cwd()
    service_settings = load_service_settings_from_yaml(current_dir.resolve() / "configs" / "service_config.yml")
    generator_settings = load_generator_settings_from_yaml(current_dir.resolve() / "configs" / "generator_config.yml")
    pipeline_settings = load_pipeline_settings_from_yaml(current_dir.resolve() / "configs" / "pipeline_config.yml")
    return service_settings, pipeline_settings, generator_settings


def generate(
    generator: PromptGenerator,
    service_settings: ServiceSettings,
    pipeline_settings: PipelineSettings,
    vllm_models: list[str],
) -> None:
    next_model_idx = 0
    prompts_to_send = []
    i = 0
    while pipeline_settings.iterations_number < 0 or i < pipeline_settings.iterations_number:
        if pipeline_settings.iterations_for_swapping_model > 0:
            if i % pipeline_settings.iterations_for_swapping_model == 0:
                next_model_idx = load_next_vllm_model(generator, vllm_models, next_model_idx)
        elif i == 0 and pipeline_settings.iterations_for_swapping_model == 0:
            next_model_idx = load_next_vllm_model(generator, vllm_models, next_model_idx)

        logger.info(f"Generation Iteration: {i}\n")

        generation_start_time = time.time()
        prompts = generator.generate()
        prompts_to_send += postprocess_prompts(prompts, pipeline_settings)
        prompts_to_send = filter_unique_prompts(prompts_to_send)

        logger.info(f"Current prompts list size: {len(prompts_to_send)} / 1000+")

        iter_duration = (time.time() - generation_start_time) / 60.0
        logger.info(f"Current iteration took: {iter_duration} min.")

        # posting accumulated prompts to the remote server with prompt validator
        if len(prompts_to_send) >= 1000:
            # The only case when we want to keep accumulating prompts is
            # the `get-prompts` service configured and prompt not being able to be delivered.
            clear_prompts = not service_settings.get_prompts_service.service_url

            if pipeline_settings.prompts_cache_file:
                cache_prompts_to_file(pipeline_settings.prompts_cache_file, prompts_to_send)

            if service_settings.get_prompts_service.service_url:
                clear_prompts = send_data_with_retry(service_settings.get_prompts_service, prompts_to_send)

            if service_settings.prompts_validator_service.service_url:
                send_data_with_retry(service_settings.prompts_validator_service, prompts_to_send)

            if clear_prompts:
                prompts_to_send.clear()

        i += 1


def postprocess_prompts(prompts: list[str], pipeline_settings: PipelineSettings) -> list[str]:
    prompts_out = post_process_generated_prompts(prompts)
    prompts_out = filter_prompts_with_words(prompts_out, pipeline_settings.prompts_with_words_to_filter_out)
    prompts_out = remove_words_from_prompts(prompts_out, pipeline_settings.words_to_remove_from_prompts)
    prompts_out = correct_non_finished_prompts(prompts_out, pipeline_settings.prepositions)
    return prompts_out


def cache_prompts_to_file(filename: str, prompts: list[str]) -> None:
    with Path(filename).open("w") as f:
        f.writelines("\n".join(prompts))


def send_data_with_retry(service_settings: PromptAggregatorServiceSettings, prompts: list[str]) -> bool:
    logger.info("Sending prompts to the `get-prompts` service.")

    prompts_to_json = json.dumps({"prompts": prompts})
    max_retries = service_settings.send_max_retries
    retry_delay = service_settings.send_retry_delay

    headers = {"Content-Type": "application/json", "X-Api-Key": f"{service_settings.api_key}"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                str(service_settings.service_url), data=prompts_to_json, headers=headers, timeout=30
            )

            if response.status_code == 200:
                logger.info("Prompts sent successfully!")
                return True

            logger.warning(f"Failed to send prompts. Attempt: {attempt}. Response code: {response.status_code}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send prompts. Attempt: {attempt}. Exception: {e}.")

        if attempt < max_retries:
            logger.info(f"Retrying to send prompts in {retry_delay} seconds.")
            time.sleep(retry_delay)

    logger.warning("Max retries reached. Failed to send prompt. Continue generating prompts.")
    return False


def load_next_vllm_model(generator: PromptGenerator, model_names: list[str], model_idx: int) -> int:
    generator.unload_vllm_model()

    logger.info(f"Next model to use: [ {model_names[model_idx]} ]")
    generator.load_vllm_model(model_names[model_idx])

    gc.collect()
    torch.cuda.empty_cache()

    return (model_idx + 1) % len(model_names)


if __name__ == "__main__":
    main()
