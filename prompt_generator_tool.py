import tqdm
import re

import PromptGenerator
import PromptChecker
import argparse


def postprocess_prompts(prompt_checker: PromptChecker.PromptChecker, prompts: list):
    """ Function for post-processing input prompts: grammar check and filtering
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :return a list with processed prompts
    """
    prompts_out = prompt_checker.check_grammar(prompts)
    prompts_out = prompt_checker.filter_prompts(prompts_out)
    return prompts_out


def check_prompts(prompt_checker: PromptChecker.PromptChecker, prompts: list, model_name: str, mode: str):
    """ Function for checking the quality of the prompts using another LLM. It also corrects/reformulates prompts.
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :param model_name: the name of the LLM that will be used
    :param mode: can be 'online' or 'offline'
    """
    for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):
        if mode == "offline":
            score = prompt_checker.transformers_check_prompt(p)
        elif mode == "online":
            score = prompt_checker.groq_check_prompt(p)
        else:
            raise ValueError("Unknown mode was specified. Supported ones are: online and offline")

        if float(score) >= 0.5:
            if "gemma" not in model_name:
                if mode == "offline":
                    p = prompt_checker.transformers_correct_prompt(p)
                else:
                    p = prompt_checker.groq_correct_prompt(p)

            p += "\n"
            PromptGenerator.save_prompts("correct_prompts.txt", [p], "a")
        else:
            p += ", [ " + score + " ]\n"
            PromptGenerator.save_prompts("wrong_prompts.txt", [p], "a")


def recycle_prompts(prompt_checker: PromptChecker.PromptChecker, prompts: list, mode: str):
    """Function for recycling unsuitable prompts and storing the qualified ones in the file
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :param mode: can be 'online' or 'offline'
    """

    for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):
        pattern = r'\[\s*\d+(\.\d+)?\s*\]'
        p = re.sub(pattern, '', p)

        if mode == "offline":
            prompt = prompt_checker.transformers_correct_prompt(p, 1.0)
            score = prompt_checker.transformers_check_prompt(prompt)
        elif mode == "online":
            prompt = prompt_checker.groq_correct_prompt(p, 1.0)
            score = prompt_checker.groq_check_prompt(prompt)
        else:
            raise ValueError("Unknown mode was specified. Supported ones are: online and offline")

        if float(score) > 0.5:
            prompt += "\n"
            PromptGenerator.save_prompts("correct_prompts_.txt", [prompt], "a")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="options: 'online', 'offline', 'filter', 'grammar', "
                                                      "'semantic_check_online', 'semantic_check_offline', "
                                                      "'recycle_prompts_offline', 'recycle_prompts_online' ")
    args = parser.parse_args()

    config_data = PromptGenerator.load_config_file()
    prompt_generator = PromptGenerator.PromptGenerator(config_data)
    prompt_checker = PromptChecker.PromptChecker(config_data)
    prompt_checker.transformers_load_checkpoint()

    if args.mode == "online":
        prompts = prompt_generator.groq_generator()
        prompts = postprocess_prompts(prompt_checker, prompts)
        check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "offline")

    elif args.mode == "offline":
        prompts = prompt_generator.transformers_generator()
        prompts = postprocess_prompts(prompt_checker, prompts)
        check_prompts(prompt_checker, prompts, config_data["llm_model_file_name"], "offline")

    elif args.mode == "filter":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_prompts(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif args.mode == "grammar":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.check_grammar(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif args.mode == "semantic_check_offline":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        check_prompts(prompt_checker, prompts, config_data["llm_model_file_name"], "offline")

    elif args.mode == "semantic_check_online":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        check_prompts(prompt_checker, prompts, config_data["llm_model_file_name"], "online")

    elif args.mode == "recycle_prompts_offline":
        prompts = PromptGenerator.load_file_with_prompts("wrong_prompts.txt")
        recycle_prompts(prompt_checker, prompts, "offline")

    elif args.mode == "recycle_prompts_online":
        prompts = PromptGenerator.load_file_with_prompts("wrong_prompts.txt")
        recycle_prompts(prompt_checker, prompts, "online")

    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Check supported modes using -h option.")
