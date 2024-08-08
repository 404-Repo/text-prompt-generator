import yaml
import os


def load_config_file(config_path: str):
    """
    Function for loading parameters for running the LLM

    Parameters
    ----------
    config_path: path to the configuration file

    Returns
    -------
    config_data: loaded dictionary with data from the configuration file
    """

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def load_file_with_prompts(file_path: str):
    """
    Function for loading the prompts dataset for processing.

    Parameters
    ----------
    file_path: path to the file

    Returns
    -------
    prompts: list with loaded prompts
    """
    with open(file_path, "r") as file:
        prompts = [line.rstrip() for line in file]
    return prompts


def save_prompts(file_name: str, prompts_list: list, mode: str = "a"):
    """
    Function for saving the prompts stored in the prompts list

    Parameters
    ----------
    file_name: a string with the name of the file that will be loaded
    prompts_list: a list with strings (generated prompts)
    mode: mode for writing the file: 'a', 'w'

    """
    with open(file_name, mode) as file:
        for p in prompts_list:
            file.write("%s" % p)
