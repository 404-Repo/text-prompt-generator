import os
import gc
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
from fastapi import FastAPI, Form
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
    logger.info("Pre-downloading all models.")
    # Startup logic
    current_dir = os.getcwd()
    generator_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                              "configs/generator_config.yml"))
    llm_models = generator_config["vllm_api"]["llm_models"]
    app.state.generator = PromptGenerator("vllm")

    if len(llm_models) > 0:
        for i in tqdm.trange(len(llm_models)):
            app.state.generator.load_model(llm_models[i])
            app.state.generator.unload_model()

    yield

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done.")


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
async def generate_prompts(inference_api: str = Form(),
                           save_locally_only: bool = Form()):
    """
    Server function for running the prompt generation
    Parameters
    ----------
    inference_api: string with the name of the inference api that is supported by the generator:
                   currently: groq, vllm
    save_locally_only: enable/disable saving locally to the hdd only without posting to remote server

    """

    logger.info(f"\n")
    logger.info("*" * 35)
    logger.info(" *** Prompt Dataset generator ***")
    logger.info("*" * 35)
    logger.info(f"\n")

    # loading server config
    current_dir = os.getcwd()

    server_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                           "configs/server_config.yml"))
    headers = {'Content-Type': 'application/json',
               'X-Api-Key': f'{server_config["api_key_prompt_server"]}'}

    # loading vllm config and getting list of models
    generator_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                              "configs/generator_config.yml"))

    if inference_api == "vllm":
        llm_models = generator_config["vllm_api"]["llm_models"]
    elif inference_api == "groq":
        llm_models = generator_config["groq_api"]["llm_models"]
    else:
        raise ValueError(f"Unsupported inference engine was specified: {inference_api}.")

    # defines whether we will have an infinite loop or not
    pipeline_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                             "configs/pipeline_config.yml"))

    if pipeline_config["iterations_number"] > -1:
        total_iters = range(pipeline_config["iterations_number"])
        logger.info(f"Requested amount of iterations: {total_iters}")
    else:
        total_iters = iter(bool, True)
        logger.info("Running infinite loop. Interrupt by CTRL + C.")

    # init prompt generator
    prompt_generator = PromptGenerator(inference_api)
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

            if len(llm_models) > 1:
                model_id = np.random.randint(0, len(llm_models))
                prompt_generator.unload_model()
                prompt_generator.load_model(llm_models[model_id])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
