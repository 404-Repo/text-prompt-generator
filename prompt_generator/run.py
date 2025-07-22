import gc
import json
from time import time
from pathlib import Path

import requests
import torch
from loguru import logger

from prompt_generator.config import (
    PipelineSettings,
    PromptAggregatorServiceSettings,
    ServiceSettings,
    load_settings,
)
from prompt_generator.generators import CategorySamplingGenerator, AbstractGenerator, SimpleGenerator
from prompt_generator.utils.discord import post_exception_to_discord, send_discord_message
from prompt_generator.utils.prompt_filtering import filter_unique_prompts
from prompt_generator.utils.logging import log_duration


def cleanup(generator: AbstractGenerator) -> None:
    generator.unload_model()
    gc.collect()
    torch.cuda.empty_cache()


def generate(
    generator: AbstractGenerator,
    service_settings: ServiceSettings,
    pipeline_settings: PipelineSettings,
) -> None:
    prompts_to_send = []
    i = 0
    while pipeline_settings.iterations_number < 0 or i < pipeline_settings.iterations_number:

        if pipeline_settings.iterations_for_swapping_model > 0 and i % pipeline_settings.iterations_for_swapping_model == 0:
                generator.load_next_model()
        elif i == 0 and pipeline_settings.iterations_for_swapping_model == 0:
                generator.load_next_model()

        logger.info(f"Generation Iteration: {i}\n")

        generation_start_time = time()
        prompts_to_send += generator.generate()
        prompts_to_send = filter_unique_prompts(prompts_to_send)

        logger.info(f"Current prompts list size: {len(prompts_to_send)} / 1000+")
        log_duration(f"Iteration {i}: ", generation_start_time)

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

def main() -> None:
    service_settings, pipeline_settings, generator_settings = load_settings()
    generator = SimpleGenerator(generator_settings, pipeline_settings)

    if service_settings.discord_webhook_url:
        send_discord_message(
            message=f"🚨 **Prompt Generator {service_settings.generator_id}** 🚨\n\n ONLINE",
            webhook_url=service_settings.discord_webhook_url,
        )
    try:
        generate(generator, service_settings, pipeline_settings)
    except Exception as e:
        if service_settings.discord_webhook_url:
            post_exception_to_discord(e, service_settings.generator_id, service_settings.discord_webhook_url)

        logger.exception(e)
        cleanup(generator)
        raise e

    cleanup(generator)
    logger.info("Done.")

if __name__ == "__main__":
    main()
