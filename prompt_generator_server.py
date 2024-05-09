import argparse
from time import time

from loguru import logger
from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import tqdm

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


@app.post("/generate_prompts/")
async def generate_prompts(with_extra_llm_check: bool = Form(False)):
    """ Server function for running the prompt generation

    :param with_extra_llm_check: boolean variable that defines whether to perform extra
           LLM based check of generated prompts or not
    """

    logger.info("Start prompt generation.")

    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data)
    prompt_generator.preload_vllm_model()

    prompt_checker = PromptChecker(config_data)

    # defines whether we will have an infinite loop or not
    if config_data["iteration_num"] > -1:
        total_iters = range(config_data["iteration_num"])
        logger.info(f"Requested amount of iterations: {total_iters}")
    else:
        total_iters = iter(bool, True)
        logger.info("Running infinite loop. Interrupt by CTRL + C.")

    t1 = time()
    # loop for prompt generation
    for i, _ in enumerate(total_iters):
        t1_local = time()

        logger.info(f"\nGeneration Iteration: {i}\n")
        prompts = prompt_generator.vllm_generator()
        # if with_extra_llm_check:
        #     del prompt_generator
        #     prompt_checker = PromptChecker(config_data)
        #     prompt_checker.preload_vllm_model()

        prompts_out = prompt_checker.filter_unique_prompts(prompts)
        prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)

        # if with_extra_llm_check:
        #     prompt_checker.preload_vllm_model()
        #     for p, _ in zip(prompts_out, tqdm.trange(len(prompts))):
        #         score = prompt_checker.vllm_check_prompt(p)
        #         if float(score) >= 0.5:
        #             p = prompt_checker.vllm_correct_prompt(p)
        #             p = p.strip()
        #             p += "\n"
        #             save_prompts(config_data["prompts_output_file"], [p], "a")
        # else:
        save_prompts(config_data["prompts_output_file"], prompts_out)

        # if with_extra_llm_check:
        #     del prompt_checker
        #     prompt_generator = PromptGenerator(config_data)
        #     prompt_generator.preload_vllm_model()

        t2_local = time()
        iter_duration = (t2_local-t1_local)/60.0
        logger.info(f"Current iteration took: {iter_duration} min.")

    t2 = time()
    total_duration = (t2-t1)/60.0
    logger.info(f"Total time: {total_duration} min.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
