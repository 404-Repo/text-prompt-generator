from Generator.prompt_generator import PromptGenerator, load_config_file, save_prompts
from Generator.prompt_checker import PromptChecker
from loguru import logger


def postprocess_prompts(prompt_checker: PromptChecker, prompts: list):
    """ Function for post-processing input prompts: grammar check and filtering
    :param prompt_checker: object that provides access to the PromptChecker methods
    :param prompts: list with input prompts
    :return a list with processed prompts
    """
    # prompts_out = prompt_checker.check_grammar(prompts)
    prompts_out = prompt_checker.filter_unique_prompts(prompts)
    prompts_out = prompt_checker.filter_prompts_with_words(prompts_out)
    return prompts_out


if __name__ == '__main__':
    config_data = load_config_file()
    prompt_generator = PromptGenerator(config_data, logger)
    prompt_checker = PromptChecker(config_data, logger)

    # path = generator.ctranslate2_prepare_model("meta-llama/Meta-Llama-3-8B-Instruct", "model", "float16")
    # prompts = prompt_generator.ctranslate2_generator()
    prompts = prompt_generator.vllm_generator()
    prompts = postprocess_prompts(prompt_checker, prompts)
    save_prompts("llama3_offline.txt", prompts)



