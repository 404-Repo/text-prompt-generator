import argparse
import requests
import json
import time
from time import time

import tqdm
from loguru import logger
from fastapi import FastAPI
import uvicorn

from Generator.prompt_generator import PromptGenerator
from Generator.prompt_checker import PromptChecker
from Generator.utils import (save_prompts,
                             load_config_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


def send_data_with_retry(config_data: dict, prompts_list: list, headers: dict):
    """

    :param config_data:
    :param prompts_list:
    :param headers:
    :return:
    """
    logger.info("Sending the data to the server.")

    prompts_to_json = json.dumps(prompts_list)
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
            time.sleep(config_data["server_retry_delay"])

    logger.warning("Max retries reached. Failed to send data. Continue generating prompts.")
    return False


@app.post("/generate_prompts/")
async def generate_prompts():
    """ Server function for running the prompt generation

    :param with_extra_llm_check: boolean variable that defines whether to perform extra
           LLM based check of generated prompts or not
    """

    logger.info("Start prompt generation.")

    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data)
    prompt_generator.preload_vllm_model()
    prompt_checker = PromptChecker(config_data)

    headers = {'Authentication': f'Bearer {config_data["api_key_prompt_server"]}'}

    # defines whether we will have an infinite loop or not
    if config_data["iteration_num"] > -1:
        total_iters = range(config_data["iteration_num"])
        logger.info(f"Requested amount of iterations: {total_iters}")
    else:
        total_iters = iter(bool, True)
        logger.info("Running infinite loop. Interrupt by CTRL + C.")

    t1 = time()
    # loop for prompt generation
    prompts_to_send = []
    for i, _ in enumerate(total_iters):
        t1_local = time()

        logger.info(f"\nGeneration Iteration: {i}\n")
        prompts = prompt_generator.vllm_generator()

        prompts_out = prompt_checker.filter_unique_prompts(prompts)
        prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)

        prompts_to_send += prompts_out
        prompts_to_send = prompt_checker.filter_unique_prompts(prompts_to_send)

        logger.info(f"Current prompts list size: {len(prompts_to_send)}")

        save_prompts(config_data["prompts_output_file"], prompts_out)

        t2_local = time()
        iter_duration = (t2_local-t1_local)/60.0
        logger.info(f"Current iteration took: {iter_duration} min.")

        if len(prompts_to_send) >= 1000:
            result = send_data_with_retry(config_data, prompts_to_send, headers)
            if result:
                prompts_to_send.clear()

    t2 = time()
    total_duration = (t2-t1)/60.0
    logger.info(f"Total time: {total_duration} min.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
