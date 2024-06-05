import yaml

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def load_config_file(config_path: str):
    """ Function for loading parameters for running the LLM
    :param config_path: path to the configuration file
    return loaded dictionary with data from the configuration file"""
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def load_file_with_prompts(file_name: str):
    """ Function for loading the prompts dataset for processing.

    :param return: list with loaded prompts
    """
    with open(file_name, "r") as file:
        prompts = [line.rstrip() for line in file]
    return prompts


def save_prompts(file_name: str, prompts_list: list, mode: str = "a"):
    """ Function for saving the prompts stored in the prompts list

    :param file_name: a string with the name of the file that will be loaded
    :param prompts_list: a list with strings (generated prompts)
    :param mode: mode for writing the file: 'a', 'w'
    """
    with open(file_name, mode) as file:
        for p in prompts_list:
            file.write("%s" % p)


def quantize_llm_awq(model_path_hf: str, quant_model_path: str):
    """ Function for quantizing the LLM model using AWQ library

    :param model_path_hf: path to the higging face library
    :param quant_model_path: output path of the quantozed model
    """
    quant_config = {"zero_point": True,
                    "q_group_size": 128,
                    "w_bit": 4,
                    "version": "GEMM"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path_hf)
    tokenizer = AutoTokenizer.from_pretrained(model_path_hf, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_model_path)
    tokenizer.save_pretrained(quant_model_path)
