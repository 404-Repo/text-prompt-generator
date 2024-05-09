import argparse
import random
import numpy as np
from time import time

from loguru import logger
from fastapi import FastAPI, Form
from pydantic import BaseModel
import uvicorn
import tqdm

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


@app.post("/get_prompts/")
async def generate_prompts(number: int = Form()):
    if number > 100:
        logger.info("To take more than 100 prompts in one request is permitted. Fallback to 100.")
        total_number = 100
    else:
        total_number = number

    config_data = load_config_file()
    prompts = load_file_with_prompts(config_data["prompts_output_file"])

    prompt_checker = PromptChecker(config_data)
    prompts = np.array(prompt_checker.filter_unique_prompts(prompts))

    sample = np.array([random.randint(0, len(prompts)) for _ in range(total_number)])
    return prompts[sample]
