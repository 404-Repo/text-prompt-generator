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
    for p, _ in zip(prompts[:], tqdm.trange(len(prompts[:]))):  #76000 :
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

            p = p.strip()
            p += "\n"
            PromptGenerator.save_prompts(file_name, [p], "a")
        else:
            p = p.strip()
            p += ", [ " + score + " ]\n"
            PromptGenerator.save_prompts("wrong_prompts.txt", [p], "a")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, help="options: 'online', 'offline', 'grammar', "
                                                      "'semantic_check_online', 'semantic_check_offline', "
                                                      "'filter_unique_prompts', 'filter_prompts' ")
    args = parser.parse_args()

    config_data = PromptGenerator.load_config_file()
    prompt_generator = PromptGenerator.PromptGenerator(config_data)
    prompt_checker = PromptChecker.PromptChecker(config_data)

    if args.mode == "online":
        prompts = prompt_generator.groq_generator()
        prompts = postprocess_prompts(prompt_checker, prompts)
        prompt_checker.transformers_load_checkpoint()
        check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "offline", config_data["prompts_output_file"])

    elif args.mode == "offline":
        prompts = prompt_generator.transformers_generator()
        prompts = postprocess_prompts(prompt_checker, prompts)
        prompt_checker.transformers_load_checkpoint()
        check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "offline", config_data["prompts_output_file"])

    elif args.mode == "filter_unique_prompts":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_unique_prompts(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif args.mode == "filter_prompts":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.filter_prompts_with_words(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif args.mode == "grammar":
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        prompts = prompt_checker.check_grammar(prompts)
        PromptGenerator.save_prompts(config_data["prompts_output_file"], prompts, "w")

    elif args.mode == "semantic_check_offline":
        prompt_checker.transformers_load_checkpoint()
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        check_prompts(prompt_checker, prompts, config_data["transformers_llm_model"], "offline")

    elif args.mode == "semantic_check_online":
        prompt_checker.transformers_load_checkpoint()
        prompts = PromptGenerator.load_file_with_prompts(config_data["prompts_output_file"])
        check_prompts(prompt_checker, prompts, config_data["groq_llm_model"], "online")

    else:
        raise ValueError(f"Unknown mode was specified: {args.mode}. Check supported modes using -h option.")
