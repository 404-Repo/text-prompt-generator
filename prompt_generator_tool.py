import tqdm
import argparse
import PromptGenerator
import PromptChecker


def postprocess_prompts(prompt_checker: PromptChecker.PromptChecker, prompts: list):
    """ Function for post-processing input prompts: grammar check and filtering
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :return a list with processed prompts
    """
    prompts_out = prompt_checker.check_grammar(prompts)
    prompts_out = prompt_checker.filter_unique_prompts(prompts_out)
    prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)
    return prompts_out


def check_prompts(prompt_checker: PromptChecker.PromptChecker,
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
    for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):  #86000 :
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
            PromptGenerator.save_prompts(file_name, [p], "a")
        else:
            p = p.strip()
            p += ", [ " + score + " ]\n"
            PromptGenerator.save_prompts("wrong_prompts.txt", [p], "a")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="options: 'prompt_generation, groq', 'prompt_generation, transformers', 'prompt_generation, llamacpp',"
                                                      "'grammar', 'filter_unique_prompts', 'filter_prompts', "
                                                      "'semantic_check, qroq', 'semantic_check, llamacpp', 'semantic_check, transformers'")
    parser.add_argument("--preload_llm", required=False, default=False, help="preload llama-cpp models")
    parser.add_argument("--quantize_llamacpp", required=False, help="'prompt_generation, digit' or 'prompt_checking, digit', where digit 1-3.")
    args = parser.parse_args()

    config_data = PromptGenerator.load_config_file()
    prompt_generator = PromptGenerator.PromptGenerator(config_data)
    prompt_checker = PromptChecker.PromptChecker(config_data)

    if bool(args.preload_llm):
        local_files_only = False
    else:
        local_files_only = True

    inputs = args.mode.split(",")
    if len(inputs) == 2:
        mode = inputs[0].strip(" ")
        option = inputs[1].strip(" ")
    else:
        mode = inputs[0].strip(" ")
        option = ""

    if mode == "prompt_generation":
        if option == "groq":
            prompts = prompt_generator.groq_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            # prompt_checker.transformers_load_checkpoint()
            prompt_checker.llamacpp_load_checkpoint(local_files_only=local_files_only)
            prompt_checker.llamacpp_load_model()
            check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "llamacpp", config_data["prompts_output_file"])
        elif option == "transformers":
            prompts = prompt_generator.transformers_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            prompt_checker.transformers_load_checkpoint()
            check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "transformers", config_data["prompts_output_file"])

        elif option == "llamacpp":
            prompt_generator.llamacpp_load_checkpoint(local_files_only=local_files_only)
            prompts = prompt_generator.llamacpp_generator()
            prompts = postprocess_prompts(prompt_checker, prompts)
            prompt_checker.llamacpp_load_checkpoint(local_files_only=local_files_only)
            prompt_checker.llamacpp_load_model()
            check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "llamacpp", config_data["prompts_output_file"])
        else:
            raise UserWarning("No option was specified in the form: --mode prompt_generation, groq. Nothing to be done.")

    elif mode == "filter_unique_prompts":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_unique_prompts(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif mode == "filter_prompts":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_prompts_with_words(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif mode == "grammar":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.check_grammar(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif mode == "semantic_check":
        if option == "groq":
            prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "groq")
        elif option == "transformers":
            prompt_checker.transformers_load_checkpoint()
            prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "transformers")
        elif option == "llamacpp":
            prompt_checker.llamacpp_load_checkpoint(local_files_only=local_files_only)
            prompt_checker.llamacpp_load_model()
            prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
            check_prompts(prompt_checker, prompts, config_data["llamacpp_model_file_name"], "llamacpp")
        else:
            raise UserWarning("No option was specified in the form: --mode prompt_generation, groq. Nothing to be done.")

    elif args.quantize_llamacpp.split(",")[0].strip(" ") == "prompt_checking":
        prompt_checker.llamacpp_load_checkpoint()
        prompt_checker.llamacpp_quantize_model(int(args.quantize_llamacpp.split(",")[1].strip(" ")))

    elif args.quantize_llamacpp.split(",")[0].strip(" ") == "prompt_generation":
        prompt_generator.llamacpp_load_checkpoint()
        prompt_generator.llamacpp_quantize_model(int(args.quantize_llamacpp.split(",")[1].strip(" ")))

    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Check supported modes using -h option.")
