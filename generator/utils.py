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


    def _preload_input_data(self):
        """ Function for preloading input data """
        self._instruction_prompt = self._load_input_prompt()
        self._object_categories = self._config_data['obj_categories']
        self._logger.info(f" Object categories: {self._object_categories}")


    @staticmethod
    def post_process_prompts(prompts_list: list):
        """ Function for post processing of the generated prompts. The LLM output is filtered from punctuation symbols and all non alphabetic characters.

        :param prompts_list: a list with strings (generated prompts)
        :return a list with processed prompts stored as strings.
        """
        result_prompts = []
        for el in prompts_list:
            lines = el.split("\n")
            processed_lines = []
            for i in range(len(lines)):
                line = re.sub(r'[^a-zA-Z`\s-]', '', lines[i])
                line = re.sub(r'\d+', '', line)
                line = line.replace(".", "")
                line = line.replace("- ", "")

                if len(line.split()) > 3:
                    if "\n" not in line:
                        line += "\n"
                    processed_lines += [line]
            result_prompts += processed_lines
        return result_prompts

    def _load_input_prompt(self):
        """ Function for loading input prompt-instruction for the LLM. It will be used for generating prompts.

        :return loaded prompt as a string from the config file.
        """

        # prompt for dataset generation
        prompt = self._config_data["prompt"]
        prompt = prompt.replace("prompts_num", str(self._config_data["prompts_num"]))

        self._logger.info(" Input prompt: ")

        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)

        self._logger.info(f"{prompt_printing}")

        return prompt
