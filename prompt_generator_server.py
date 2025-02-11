import argparse
import gc
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import sleep, time

import generator.utils.io_utils as io_utils
import generator.utils.prompts_filtering_utils as prompt_filters
import requests
import torch
import tqdm
import uvicorn
from fastapi import FastAPI, Form
from generator.prompt_generator import PromptGenerator
from loguru import logger


def get_args() -> tuple[argparse.Namespace, list[str]]:
    """
    Function for setting up server port arg
    Returns
    -------
    args: list with input (port number)
    wxtras: other extra arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--backend", type=str, default="vllm")
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """
    Function for preloading LLM models before they will be used by pipeline

    Parameters
    ----------
    app: FastAPI server app
    args: list of input arguments
    """
    logger.info("Pre-downloading all models.")
    # Startup logic
    current_dir = Path.cwd()
    generator_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "generator_config.yml")
    llm_model = generator_config["vllm_api"]["llm_model"]
    app.state.generator = PromptGenerator(args.backend)
    app.state.generator.load_model(llm_model)

    yield

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done.")


app.router.lifespan_context = lifespan


def send_data_with_retry(config_data: dict, prompts_list: list[str], headers: dict) -> bool:
    """
    Function for sending generated prompts to the remote server

    Parameters
    ----------
    config_data: dictionary with configuration for the remote server
    prompts_list: list of strings with prompts
    headers: a dictionary with data that is essential for posting data to the remote server

    Returns
    -------
    true if data was sent successfully;
    false otherwise
    """
    logger.info("Sending the data to the server.")

    prompts_to_json = json.dumps({"prompts": prompts_list})
    max_retries = config_data["server_max_retries_number"]
    retry_delay = int(config_data["server_retry_delay"])

    for attempt in tqdm.trange(1, max_retries + 1):
        try:
            response = requests.post(
                config_data["api_prompt_server_url"], data=prompts_to_json, headers=headers, timeout=30
            )

            if response.status_code == 200:
                logger.info("Successfully sent the data!")
                return True

            logger.warning(f"Failed to send data (attempt {attempt}): {response.status_code}.")
            logger.warning("Reattempting...")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error sending data [attempt: {attempt}]: {e}")

        if attempt < max_retries:
            logger.info(f"Retrying in {retry_delay} seconds.")
            sleep(retry_delay)

    logger.warning("Max retries reached. Failed to send data. Continue generating prompts.")
    return False


@app.post("/generate_prompts/")
async def generate_prompts(save_locally_only: bool = Form(False)) -> None:
    """
    Server function for running the prompt generation
    Parameters
    ----------
    inference_api: string with the name of the inference api that is supported by the generator:
                   currently: groq, vllm
    save_locally_only: enable/disable saving locally to the hdd only without posting to remote server

    """

    logger.info("\n" + "*" * 35)
    logger.info(" *** Prompt Dataset generator ***")
    logger.info("*" * 35 + "\n")

    # loading server config
    current_dir = Path.cwd()

    server_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "server_config.yml")
    headers = {"Content-Type": "application/json", "X-Api-Key": f'{server_config["api_key_prompt_server"]}'}

    # defines whether we will have an infinite loop or not
    pipeline_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "pipeline_config.yml")

    total_iters: range | Iterator[bool]
    if pipeline_config["iterations_number"] > -1:
        total_iters = range(pipeline_config["iterations_number"])
        logger.info(f"Requested amount of iterations: {total_iters}")
    else:
        total_iters = iter(bool, True)
        logger.info("Running infinite loop. Interrupt by CTRL + C.")

    # loop for prompt generation
    prompts_to_send = []
    for i, _ in enumerate(total_iters):
        t1_local = time()

        logger.info(f"Generation Iteration: {i}\n")
        prompts = app.state.generator.generate()

        prompts_out = prompt_filters.post_process_generated_prompts(prompts)
        prompts_out = prompt_filters.filter_unique_prompts(prompts_out)
        prompts_out = prompt_filters.filter_prompts_with_words(
            prompts_out, pipeline_config["prompts_with_words_to_filter_out"]
        )
        prompts_out = prompt_filters.remove_words_from_prompts(
            prompts_out, pipeline_config["words_to_remove_from_prompts"]
        )
        prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)

        prompts_to_send += prompts_out
        prompts_to_send = prompt_filters.filter_unique_prompts(prompts_to_send)

        logger.info(f"Current prompts list size: {len(prompts_to_send)} / 1000+")

        io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out)

        t2_local = time()
        iter_duration = (t2_local - t1_local) / 60.0
        logger.info(f"Current iteration took: {iter_duration} min.")

        if len(prompts_to_send) >= 1000:
            if not save_locally_only:
                result = send_data_with_retry(server_config, prompts_to_send, headers)
                if result:
                    prompts_to_send.clear()

        if i % 500 == 0:
            prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
            prompts_filtered = prompt_filters.filter_unique_prompts(prompts)
            io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_filtered, "w")
            prompts_to_send.clear()
            logger.info(f"Current dataset size: {len(prompts_filtered)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)  # noqa:S104
    app.state.generator.unload_model()
