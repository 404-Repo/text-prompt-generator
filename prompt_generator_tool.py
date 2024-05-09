import tqdm
import argparse

from Generator.prompt_generator import PromptGenerator
from Generator.utils import (load_config_file,
                             load_file_with_prompts,
                             save_prompts)
from Generator.prompt_checker import PromptChecker


def postprocess_prompts(prompt_checker: PromptChecker, prompts: list):
    """ Function for post-processing input prompts: grammar check and filtering
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :return a list with processed prompts
    """
    prompts_out = prompt_checker.filter_unique_prompts(prompts)
    prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)
    return prompts_out


def check_prompts(prompt_checker: PromptChecker,
                  prompts: list,
                  model_name: str,
                  mode: str,
                  file_name: str = "correct_prompts.txt"):
    """ Function for checking the quality of the prompts using another LLM. It also corrects/reformulates prompts.
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :param model_name: the name of the LLM that will be used
    :param mode: can be 'online' or 'offline'
    :param file_name: the name of the file where the suitable prompts will be output after filtering (optional), default "correct_prompts.txt"
    """
    start_batch = 180000
    end_batch = 180100

    for p, _ in zip(prompts[start_batch:end_batch], tqdm.trange(len(prompts[start_batch:end_batch]))):
        if mode == "vllm":
            score = prompt_checker.vllm_check_prompt(p)
        elif mode == "groq":
            score = prompt_checker.groq_check_prompt(p)
        else:
            raise ValueError("Unknown mode was specified. Supported ones are: online and offline")

        if float(score) >= 0.5:
            if "gemma" not in model_name:
                if mode == "vllm":
                    p = prompt_checker.vllm_correct_prompt(p)
                else:
                    p = prompt_checker.groq_correct_prompt(p)

            p = p.strip()
            p += "\n"
            save_prompts(file_name, [p], "a")
        else:
            p = p.strip()
            p += ", [ " + score + " ]\n"
            save_prompts("wrong_prompts.txt", [p], "a")


def console_args():
    """ Function that parses the argument passed via console
    :return a list of parsed arguments with their values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=False, help="options: 'prompt_generation, groq', 'prompt_generation, vllm',"
                                                       "'grammar', 'filter_unique_prompts', 'filter_prompts', "
                                                       "'semantic_check, qroq', 'semantic_check, vllm'")
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


if __name__ == '__main__':
    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data)
    prompt_checker = PromptChecker(config_data)

    proc_mode, proc_mode_option = console_args()

    if config_data["iteration_num"] > -1:
        total_iters = range(config_data["iteration_num"])
    else:
        total_iters = iter(bool, True)

    if proc_mode == "prompt_generation":
        if proc_mode_option == "groq":
            for i, _ in enumerate(total_iters):
                prompts = prompt_generator.groq_generator()
                prompts = postprocess_prompts(prompt_checker, prompts)
                check_prompts(prompt_checker, prompts, config_data["groq_llm_model_prompt_checker"], "vllm", config_data["prompts_output_file"])

        elif proc_mode_option == "vllm":
            prompt_generator.preload_vllm_model()
            for i, _ in enumerate(total_iters):
                prompts = prompt_generator.vllm_generator()
                prompts = postprocess_prompts(prompt_checker, prompts)
                check_prompts(prompt_checker, prompts, config_data["vllm_llm_model_prompt_checker"], "vllm", config_data["prompts_output_file"])

        else:
            raise UserWarning("No option was specified in the form: --mode prompt_generation, groq. Nothing to be done.")

    elif proc_mode == "filter_unique_prompts":
        prompts = load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_unique_prompts(prompts)
        save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif proc_mode == "filter_prompts":
        prompts = load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_prompts_with_words(prompts)
        save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif proc_mode == "semantic_check":
        if proc_mode_option == "groq":
            prompts = load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["groq_llm_model_prompt_checker"], "groq")
        elif proc_mode_option == "vllm":
            prompts = load_file_with_prompts(config_data["prompts_output_file"])
            prompt_checker.preload_vllm_model()
            check_prompts(prompt_checker, prompts, config_data["vllm_llm_model_prompt_checker"], "vllm")
        else:
            raise UserWarning("No option was specified in the form: --mode prompt_generation, groq. Nothing to be done.")

    else:
        raise ValueError("Unknown mode was specified. Check supported modes using -h option.")
