import argparse
from time import time

from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tqdm

from Generator.prompt_generator import PromptGenerator
from Generator.prompt_checker import PromptChecker
from Generator.utils import (save_prompts,
                             load_file_with_prompts,
                             load_config_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()


class RequestData(BaseModel):
    """ Data class

    :param with_extra_llm_check: enables/disables running the prompt generator with extra LLM check of the prompts quality
    """
    with_extra_llm_check: bool


@app.post("/generate_prompts/")
async def generate_prompts(request: RequestData):
    """ Server function for running the prompt generation

    :param request: parameter with boolean variable that defines how prompt generation will work
    """

    logger.info("Start prompt generation.")

    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data)
    prompt_generator.preload_vllm_model()

    prompt_checker = PromptChecker(config_data)

    # defines whether we will have an infinite loop or not
    if config_data["iteration_num"] > -1:
        total_iters = range(config_data["iteration_num"])
    else:
        total_iters = iter(bool, True)

    # loop for prompt generation
    for i, _ in enumerate(total_iters):
        prompts = prompt_generator.vllm_generator()
        if request.with_extra_llm_check:
            prompt_generator.unload_vllm_model()

        prompts_out = prompt_checker.filter_unique_prompts(prompts)
        prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)

        if request.with_extra_llm_check:
            prompt_checker.preload_vllm_model()
            for p, _ in zip(prompts_out, tqdm.trange(len(prompts))):
                score = prompt_checker.vllm_check_prompt(p)
                if float(score) >= 0.5:
                    p = prompt_checker.vllm_correct_prompt(p)
                    p = p.strip()
                    p += "\n"
                    save_prompts("prompt_dataset.txt", [p], "a")
        else:
            save_prompts("prompt_dataset.txt", prompts_out)

        if request.with_extra_llm_check:
            prompt_checker.unload_vllm_model()
            prompt_generator.preload_vllm_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
