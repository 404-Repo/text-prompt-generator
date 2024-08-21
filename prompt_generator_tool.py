import os
import argparse

import numpy as np
import tqdm
from loguru import logger
from time import time

import generator.utils.prompts_filtering_utils as prompt_filters
import generator.utils.io_utils as io_utils

from generator.prompt_generator import PromptGenerator


def console_args():
    """
    Function that parses the argument passed via console

    Returns
    -------
    proc_mode: tool mode, string value (prompt_generation, filter_unique_prompts, filter_prompts_with_words)
    proc_mode_option: processing option (only for prompt_generation: vllm or groq, otherwise "")
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=False, help="options: "
                                                       "'preload_llm, vllm', "
                                                       "'prompt_generation, groq', "
                                                       "'prompt_generation, vllm', "
                                                       "'filter_unique_prompts', "
                                                       "'filter_prompts_with_words'")
    args = parser.parse_args()

    if args.mode is not None:
        inputs = args.mode.split(",")
        if len(inputs) == 2:
            proc_mode = inputs[0].strip(" ")
            proc_mode_option = inputs[1].strip(" ")
        else:
            proc_mode = inputs[0].strip(" ")
            proc_mode_option = ""
    else:
        proc_mode = ""
        proc_mode_option = ""

    return proc_mode, proc_mode_option


def main():
    """ Pipeline wrapper """
    proc_mode, proc_mode_option = console_args()
    current_dir = os.getcwd()

    pipeline_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                             "configs/pipeline_config.yml"))
    generator_config = io_utils.load_config_file(os.path.join(os.path.relpath(current_dir),
                                                              "configs/generator_config.yml"))
    prompt_generator = PromptGenerator(proc_mode_option)

    if proc_mode_option == "vllm":
        llm_model = generator_config["vllm_api"]["llm_model"]
    elif proc_mode_option == "groq":
        llm_model = generator_config["groq_api"]["llm_model"]
    else:
        raise ValueError(f"Unknown backend was specified: {proc_mode_option}")

    if proc_mode == "preload_llm":
        t1 = time()

        prompt_generator.load_model(llm_model)
        prompt_generator.unload_model()

        t2 = time()
        duration = (t2 - t1) / 60
        logger.info(f" It tooK: {duration} mins.")

    elif proc_mode == "prompt_generation" and proc_mode_option != "":
        if pipeline_config["iterations_number"] > -1:
            total_iters = range(pipeline_config["iterations_number"])
        else:
            total_iters = iter(bool, True)

        prompt_generator.load_model(llm_model)

        t1 = time()
        prompts_dataset = []
        for i, _ in enumerate(total_iters):
            prompts = prompt_generator.generate()
            prompts_out = prompt_filters.post_process_generated_prompts(prompts)
            prompts_out = prompt_filters.filter_unique_prompts(prompts_out)
            prompts_out = prompt_filters.filter_prompts_with_words(prompts_out,
                                                                   pipeline_config["prompts_with_words_to_filter_out"])
            prompts_out = prompt_filters.remove_words_from_prompts(prompts_out,
                                                                   pipeline_config["words_to_remove_from_prompts"])
            prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)
            prompts_dataset += prompts_out

            if (len(prompts_dataset) > 100) or (i >= pipeline_config["iterations_number"]-1):
                logger.info(f"Saving batch of prompts: {len(prompts_dataset)}")
                io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_dataset, "a")
                prompts_dataset.clear()

        t2 = time()
        duration = (t2 - t1) / 60
        logger.info(f" It tooK: {duration} mins.")

        prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
        prompts_filtered = prompt_filters.filter_unique_prompts(prompts)
        io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_filtered, "w")

    elif proc_mode == "filter_unique_prompts":
        prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
        prompts_out = prompt_filters.filter_unique_prompts(prompts)
        prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)
        io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out, "w")

    elif proc_mode == "filter_prompts_with_words":
        prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
        prompts_out = prompt_filters.filter_prompts_with_words(prompts, pipeline_config["prompts_with_words_to_filter_out"])
        prompts_out = prompt_filters.remove_words_from_prompts(prompts_out,
                                                               pipeline_config["words_to_remove_from_prompts"])
        prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)
        io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out, "w")

    else:
        raise ValueError("Unknown mode was specified. Check supported modes using -h option.")

    prompt_generator.unload_model()


if __name__ == '__main__':
    main()
