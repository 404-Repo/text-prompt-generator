import argparse
from collections.abc import Iterator
from pathlib import Path
from time import time

import generator.utils.io_utils as io_utils
import generator.utils.prompts_filtering_utils as prompt_filters
from generator.prompt_generator import PromptGenerator
from loguru import logger


def console_args() -> tuple[str, str]:
    """
    Function that parses the argument passed via console

    Returns
    -------
    proc_mode: tool mode, string value (prompt_generation, filter_unique_prompts, filter_prompts_with_words)
    proc_mode_option: processing option (only for prompt_generation: vllm or groq, otherwise "")
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=False,
        help="options: "
        "'preload_llm, vllm', "
        "'prompt_generation, groq', "
        "'prompt_generation, vllm', "
        "'filter_unique_prompts', "
        "'filter_prompts_with_words'",
    )
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


def main() -> None:
    """Pipeline wrapper"""
    proc_mode, proc_mode_option = console_args()
    current_dir = Path.cwd()

    pipeline_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "pipeline_config.yml")
    generator_config = io_utils.load_config_file(current_dir.resolve() / "configs" / "generator_config.yml")
    prompt_generator = PromptGenerator(proc_mode_option)

    llm_model = get_llm_model(proc_mode_option, generator_config)

    try:
        if proc_mode == "preload_llm":
            preload_llm(prompt_generator, llm_model)
        elif proc_mode == "prompt_generation" and proc_mode_option:
            generate_prompts(prompt_generator, llm_model, pipeline_config)
        elif proc_mode == "filter_unique_prompts":
            filter_unique_prompts(pipeline_config)
        elif proc_mode == "filter_prompts_with_words":
            filter_prompts_with_words(pipeline_config)
        else:
            raise ValueError("Unknown mode was specified. Check supported modes using -h option.")
    finally:
        prompt_generator.unload_model()


def get_llm_model(proc_mode_option: str, generator_config: dict) -> str:
    if proc_mode_option == "vllm":
        return str(generator_config["vllm_api"]["llm_model"])
    elif proc_mode_option == "groq":
        return str(generator_config["groq_api"]["llm_model"])
    else:
        raise ValueError(f"Unknown backend was specified: {proc_mode_option}")


def preload_llm(prompt_generator: PromptGenerator, llm_model: str) -> None:
    start_time = time()
    prompt_generator.load_model(llm_model)
    prompt_generator.unload_model()
    duration = (time() - start_time) / 60
    logger.info(f"It took: {duration:.2f} mins.")


def generate_prompts(prompt_generator: PromptGenerator, llm_model: str, pipeline_config: dict) -> None:
    total_iters = get_total_iters(pipeline_config)
    prompt_generator.load_model(llm_model)

    start_time = time()
    prompts_dataset: list[str] = []
    for i, _ in enumerate(total_iters):
        prompts = prompt_generator.generate()
        prompts_out = process_prompts(prompts, pipeline_config)
        prompts_dataset.extend(prompts_out)

        if (len(prompts_dataset) > 100) or (i >= pipeline_config["iterations_number"] - 1):
            logger.info(f"Saving batch of prompts: {len(prompts_dataset)}")
            io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_dataset, "a")
            prompts_dataset.clear()

    duration = (time() - start_time) / 60
    logger.info(f"It took: {duration:.2f} mins.")

    finalize_prompts(pipeline_config)


def get_total_iters(pipeline_config: dict) -> range | Iterator[bool]:
    return (
        range(pipeline_config["iterations_number"]) if pipeline_config["iterations_number"] > -1 else iter(bool, True)
    )


def process_prompts(prompts: list[str], pipeline_config: dict) -> list[str]:
    prompts_out = prompt_filters.post_process_generated_prompts(prompts)
    prompts_out = prompt_filters.filter_unique_prompts(prompts_out)
    prompts_out = prompt_filters.filter_prompts_with_words(
        prompts_out, pipeline_config["prompts_with_words_to_filter_out"]
    )
    prompts_out = prompt_filters.remove_words_from_prompts(prompts_out, pipeline_config["words_to_remove_from_prompts"])
    return prompt_filters.correct_non_finished_prompts(prompts_out)


def finalize_prompts(pipeline_config: dict) -> None:
    prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
    prompts_filtered = prompt_filters.filter_unique_prompts(prompts)
    io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_filtered, "w")


def filter_unique_prompts(pipeline_config: dict) -> None:
    prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
    prompts_out = prompt_filters.filter_unique_prompts(prompts)
    prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)
    io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out, "w")


def filter_prompts_with_words(pipeline_config: dict) -> None:
    prompts = io_utils.load_file_with_prompts(pipeline_config["prompts_output_file"])
    prompts_out = prompt_filters.filter_prompts_with_words(prompts, pipeline_config["prompts_with_words_to_filter_out"])
    prompts_out = prompt_filters.remove_words_from_prompts(prompts_out, pipeline_config["words_to_remove_from_prompts"])
    prompts_out = prompt_filters.correct_non_finished_prompts(prompts_out)
    io_utils.save_prompts(pipeline_config["prompts_output_file"], prompts_out, "w")


if __name__ == "__main__":
    main()
