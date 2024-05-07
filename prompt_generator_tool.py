import tqdm
import argparse

from Generator.prompt_generator import (PromptGenerator,
                                        load_config_file,
                                        load_file_with_prompts,
                                        save_prompts)
from Generator.prompt_checker import PromptChecker


def postprocess_prompts(prompt_checker: PromptChecker, prompts: list):
    """ Function for post-processing input prompts: grammar check and filtering
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :return a list with processed prompts
    """
    prompts_out = prompt_checker.check_grammar(prompts)
    prompts_out = prompt_checker.filter_unique_prompts(prompts_out)
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
    for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):  #179000 - 184000 :
        if mode == "transformers":
            score = prompt_checker.transformers_check_prompt(p)
        elif mode == "llamacpp":
            score = prompt_checker.llamacpp_check_prompt(p)
        elif mode == "groq":
            score = prompt_checker.groq_check_prompt(p)
        else:
            raise ValueError("Unknown mode was specified. Supported ones are: online and offline")

        if float(score) >= 0.5:
            if "gemma" not in model_name:
                if mode == "transformers":
                    p = prompt_checker.transformers_correct_prompt(p)
                elif mode == "llamacpp":
                    p = prompt_checker.llamacpp_correct_prompt(p)
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
    parser.add_argument("--mode", required=False, help="options: 'prompt_generation, groq', 'prompt_generation, transformers', 'prompt_generation, llamacpp',"
                                                       "'grammar', 'filter_unique_prompts', 'filter_prompts', "
                                                       "'semantic_check, qroq', 'semantic_check, llamacpp', 'semantic_check, transformers'")
    parser.add_argument("--preload_llm", required=False, help="'prompt_generation, transformers', 'prompt_generation, llamacpp', "
                                                              "'prompt_checking, transformers', 'prompt_checking, llamacpp' - options for preloading offline models.")
    parser.add_argument("--quantize_llamacpp", required=False, help="'prompt_generation, digit' or 'prompt_checking, digit', where digit 1-3.")
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

    if args.preload_llm is not None:
        inputs = args.preload_llm.split(",")
        load_llm_mode = inputs[0].strip(" ")
        load_llm_option = inputs[1].strip(" ")
    else:
        load_llm_mode = ""
        load_llm_option = ""

    if args.quantize_llamacpp is not None:
        inputs = args.quantize_llamacpp.split(",")
        quant_llm_mode = inputs[0].strip(" ")
        quant_llm_option = int(inputs[1].strip(" "))
    else:
        quant_llm_mode = ""
        quant_llm_option = ""

    return proc_mode, proc_mode_option, load_llm_mode, load_llm_option, quant_llm_mode, quant_llm_option


if __name__ == '__main__':
    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data)
    prompt_checker = PromptChecker(config_data)

    proc_mode, proc_mode_option, load_llm_mode, load_llm_option, quant_llm_mode, quant_llm_option = console_args()

    if proc_mode == "prompt_generation":
        if proc_mode_option == "groq":
            prompts = prompt_generator.groq_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            prompt_checker.transformers_load_checkpoint()
            check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "transformers", config_data["prompts_output_file"])

        elif proc_mode_option == "openai":
            prompts = prompt_generator.openai_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            save_prompts("chatgpt.txt", prompts)

        elif proc_mode_option == "transformers":
            prompt_generator.transformers_load_checkpoint()
            prompts = prompt_generator.transformers_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            prompt_checker.transformers_load_checkpoint()
            check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "transformers", config_data["prompts_output_file"])

        elif proc_mode_option == "llamacpp":
            prompt_generator.llamacpp_load_checkpoint()
            prompts = prompt_generator.llamacpp_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            prompt_checker.llamacpp_load_checkpoint()
            prompt_checker.llamacpp_load_model()
            check_prompts(prompt_checker, prompts, config_data["llamacpp_model_file_name_prompt_checker"], "llamacpp", config_data["prompts_output_file"])
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

    elif proc_mode == "grammar":
        prompts = load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.check_grammar(prompts)
        save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif proc_mode == "semantic_check":
        if proc_mode_option == "groq":
            prompts = load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "groq")
        elif proc_mode_option == "transformers":
            prompt_checker.transformers_load_checkpoint()
            prompts = load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "transformers")
        elif proc_mode_option == "llamacpp":
            prompt_checker.llamacpp_load_checkpoint()
            prompt_checker.llamacpp_load_model()
            prompts = load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["llamacpp_model_file_name"], "llamacpp")
        else:
            raise UserWarning("No option was specified in the form: --mode prompt_generation, groq. Nothing to be done.")

    elif load_llm_mode == "prompt_generation":
        if load_llm_option == "transformers":
            prompt_generator.transformers_load_checkpoint(load_in_4bit=True,
                                                          load_in_8bit=False,
                                                          bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_use_double_quant=True)
        elif load_llm_option == "llamacpp":
            prompt_generator.llamacpp_load_checkpoint(local_files_only=False)
        else:
            raise UserWarning("No option was specified in the form: --preload_llm prompt_generation, transformers. Nothing to be done.")

    elif load_llm_mode == "prompt_checking":
        if load_llm_option == "transformers":
            prompt_checker.transformers_load_checkpoint(load_in_4bit=True, load_in_8bit=False)
        elif load_llm_option == "llamacpp":
            prompt_checker.llamacpp_load_checkpoint(local_files_only=False)
        else:
            raise UserWarning("No option was specified in the form: --preload_llm prompt_checking, transformers. Nothing to be done.")

    elif quant_llm_mode == "prompt_checking":
        prompt_checker.llamacpp_load_checkpoint()
        prompt_checker.llamacpp_quantize_model(quant_llm_option)

    elif quant_llm_mode == "prompt_generation":
        prompt_generator.llamacpp_load_checkpoint()
        prompt_generator.llamacpp_quantize_model(quant_llm_option)

    else:
        raise ValueError("Unknown mode was specified. Check supported modes using -h option.")
