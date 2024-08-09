import os
import gc
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

import argparse
import requests
import json
import numpy as np
from time import (time, sleep)
from typing import Dict, List
from contextlib import asynccontextmanager

import tqdm
import torch
from loguru import logger
from fastapi import FastAPI
import uvicorn

import generator.utils.io_utils as io_utils
from generator.prompt_generator import PromptGenerator
import generator.utils.prompts_filtering_utils as prompt_filters


def get_args():
    """
    Function for setting up server port arg
    Returns
    -------
    args: list with input (port number)
    wxtras: other extra arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Function for preloading LLM models before they will be used by pipeline

    Parameters
    ----------
    app: FastAPI server app
    """
    # Startup logic
    vllm_config = io_utils.load_config_file("./configs/vllm_config.yml")
    llm_models = vllm_config["llm_models"]
    app.state.generator = PromptGenerator("vllm")

    for i in tqdm.trange(len(llm_models)):
        app.state.generator.load_model(llm_models[i])

    yield

    gc.collect()
    torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


def send_data_with_retry(config_data: Dict, prompts_list: List[str], headers: Dict):
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

    for attempt in tqdm.trange(1, config_data["server_max_retries_number"] + 1):
        try:
            response = requests.post(config_data["api_prompt_server_url"],
                                     data=prompts_to_json,
                                     headers=headers)

            if response.status_code == 200:
                logger.info("Successfully sent the data!")
                return True
            else:
                logger.warning(f"Failed to send data (attempt {attempt}): {response.status_code}.")
                logger.warning("Reattempting...")
        except requests.RequestException as e:
            logger.warning(f"Error sending data [attempt: {attempt}]: {e}")

        if attempt < config_data["server_max_retries_number"]:
            logger.info(f'Retrying in {config_data["server_retry_delay"]}seconds.')
            retry_delay = int(config_data["server_retry_delay"])
            sleep(retry_delay)

    logger.warning("Max retries reached. Failed to send data. Continue generating prompts.")
    return False


@app.post("/generate_prompts/")
async def generate_prompts():
    """ Server function for running the prompt generation """

    logger.info(f"\n")
    logger.info("*" * 35)
    logger.info(" *** Prompt Dataset generator ***")
    logger.info("*" * 35)
    logger.info(f"\n")

    # loading server config
    server_config = io_utils.load_config_file("./configs/server_config.yml")
    headers = {'Content-Type': 'application/json',
               'X-Api-Key': f'{server_config["api_key_prompt_server"]}'}

    # loading vllm config and getting list of models
    vllm_config = io_utils.load_config_file("./configs/vllm_config.yml")
    llm_models = vllm_config["llm_models"]

    # defines whether we will have an infinite loop or not
    pipeline_config = io_utils.load_config_file("./configs/pipeline_config.yml")

    if pipeline_config["iteration_number"] > -1:
        total_iters = range(pipeline_config["iterations_number"])
        logger.info(f"Requested amount of iterations: {total_iters}")
    else:
        total_iters = iter(bool, True)
        logger.info("Running infinite loop. Interrupt by CTRL + C.")

    # init prompt generator
    prompt_generator = PromptGenerator("vllm")
    prompt_generator.load_model(llm_models[0])

    # loop for prompt generation
    prompts_to_send = []
    for i, _ in enumerate(total_iters):
        t1_local = time()

        logger.info(f"Generation Iteration: {i}\n")
        prompts = prompt_generator.generate()

        prompts_out = prompt_filters.post_process_generated_prompts(prompts)
        prompts_out = prompt_filters.filter_unique_prompts(prompts_out)
        prompts_out = prompt_filters.filter_prompts_with_words(prompts_out,
                                                               pipeline_config["filter_prompts_with_words"])
        prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)

        prompts_to_send += prompts_out
        prompts_to_send = prompt_filters.filter_unique_prompts(prompts_to_send)

        logger.info(f"Current prompts list size: {len(prompts_to_send)} / 1000+")

        io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out)

        t2_local = time()
        iter_duration = (t2_local-t1_local)/60.0
        logger.info(f"Current iteration took: {iter_duration} min.")

        if len(prompts_to_send) >= 1000:
            result = send_data_with_retry(server_config, prompts_to_send, headers)
            if result and i % 500 == 0:
                prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
                prompts_filtered = prompt_filters.filter_unique_prompts(prompts)
                io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_filtered, "w")
                prompts_to_send.clear()

                if len(llm_models) > 1:
                    model_id = np.random.randint(0, len(llm_models))
                    prompt_generator.unload_model()
                    prompt_generator.load_model(llm_models[model_id])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
