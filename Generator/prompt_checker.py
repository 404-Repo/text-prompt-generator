import re
from typing import Optional

import groq
from loguru import logger
from huggingface_hub import login
from vllm import LLM, SamplingParams


class PromptChecker:
    """
    Class that provides an implementation for different filtering & correction methods for generated prompts.
    """
    def __init__(self, config_file_data: dict, logger_: logger):
        """
        :param config_file_data: a dictionary with preloaded parameters for running the pipeline.
        :param logger_:
        """

        self._logger = logger_
        self._config_data = config_file_data

        if self._config_data["groq_api_key"] == "":
            self._logger.warning(f" Groq Api Access Token was not specified. "
                                 f"You will not be able to use Groq API without it.")

        # login to hugging face platform using api token
        if self._config_data["hugging_face_api_key"] == "":
            self._logger.warning(f" Hugging Face Api Access Token was not specified. "
                                 f"You will not be able to download Gemma model.")
        else:
            login(token=self._config_data["hugging_face_api_key"])

        self._generator = None

        self._prompt_for_correction = (f"Perform semantic analysis check of the input prompt. "
                                       f"Perform contextual analysis check of the input prompt. "
                                       f"Remove all digits from the corrected prompt. "
                                       f"On the basis of those checks correct input prompt so it will pass them with the highest score. "
                                       f"Corrected prompt should contain no more than five or six words. "
                                       f"You must always output only corrected prompt and nothing else. ")

        self._prompt_for_checking = (f"Perform semantic analysis check of the input prompt. If failed, score it the lowest. "
                                     f"Perform contextual analysis check of the input prompt. If failed, score it the lowest. "
                                     f"Check if all the words in the input prompt makes sense together and describe an object. If failed, score it the lowest. "
                                     f"Check if the input prompt has a logic between the words. If failed, score it the lowest. "
                                     f"Check if the input prompt is finished and has an object or subject in it. If not, score it the lowest and ignore other checks. "
                                     f"Check if all words in the prompt can be found in a dictionary. If not, score it the lowest. "
                                     f"Use performed checks to score the input prompt between 0 (all checks are failed) and 1 (all checks passed). "
                                     f"You must keep answers short and concise. "
                                     f"You must always output only a single float digit. ")

    def groq_check_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.
        This uses online groq api. Keep in mind that with time the performance will degenerate.

        :param prompt: a string with prompt that will be checked.
        :param temperature:
        :return a float value between 0 and 1 that will be used for filtering of the prompt.
        """

        object_categories = self._config_data['obj_categories']

        prompt_in = ((f"input prompt: '{prompt}'. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. ") +
                     self._prompt_for_checking)

        result = self._groq_process_prompt(prompt_in, 100, temperature)
        score = re.findall("\d+\.\d+", result)

        return score[0]

    def groq_correct_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for correcting the input prompt in case if it does not satisfy provided conditions.
        This uses online groq api. Keep in mind that with time the performance will degenerate.

        :param prompt: a string with prompt that will be checked and potentially rewritten.
        :param temperature:
        :return a rewritten prompt as a python string.
        """

        object_categories = self._config_data['obj_categories']
        filter_words = self._config_data["filter_prompts_with_words"]

        prompt_in = ((f"input prompt: {prompt}. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. "
                      f"Avoid using words from the list: {filter_words}. ") +
                     self._prompt_for_correction)

        result = self._groq_process_prompt(prompt_in, 500, temperature)
        result = result.split("\n")
        result = result[0].replace("**Corrected Prompt:**", "").strip()
        result = result.replace("**Corrected prompt:**", "")
        result = result.replace("**Corrected prompt:", "")
        result = result.replace("**", "")
        return result

    def _groq_process_prompt(self, prompt: str, max_tokens: int, temperature: float):
        """ Function that process the input prompt-instruction

        :param prompt: a string with prompt that will be checked and potentially rewritten
        :param max_tokens: the maximum amount of tokens that will limit the output prompt
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return generated prompt
        """

        client = groq.Groq(api_key=self._config_data["groq_api_key"])
        output = client.chat.completions.create(messages=[{
                                                            "role": "user",
                                                            "content": prompt
                                                         }],
                                                model="gemma-7b-it",
                                                seed=self._config_data['llm_model']['seed'],
                                                temperature=temperature,
                                                top_p=1,
                                                max_tokens=max_tokens)
        result = output.choices[0].message.content

        return result

    def vllm_check_prompt(self, prompt: str, temperature: float = 0.5):
        object_categories = self._config_data['obj_categories']

        prompt_in = ((f"input prompt: '{prompt}'. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. ") +
                     self._prompt_for_checking)

        result = self._vllm_process_prompt(prompt_in, 100, temperature)
        score = re.findall("\d+\.\d+", result)
        return score[0]

    def vllm_correct_prompt(self, prompt: str, temperature: float = 0.5):
        object_categories = self._config_data['obj_categories']
        filter_words = self._config_data["filter_prompts_with_words"]

        prompt_in = ((f"input prompt: {prompt}. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. "
                      f"Avoid using words from the list: {filter_words}. ") +
                     self._prompt_for_correction)
        result = self._vllm_process_prompt(prompt_in, 150, temperature)
        result = result.split("\n")
        result = result[0].replace("**Corrected Prompt:**", "").strip()
        result = result.replace("**Corrected prompt:**", "")
        result = result.replace("**Corrected prompt:", "")
        result = result.replace("**", "")
        return result

    def _vllm_process_prompt(self, prompt: str, max_tokens: int, temperature: float):
        sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)
        outputs = self._generator.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def preload_vllm_model(self, quantization: Optional[str] = None):
        """ Function for preloading LLM model in GPU memory

        :param quantization: optional parameter that defines the quantizaton of the model:
                             "awq", "gptq", "squeezellm", and "fp8" (experimental); Default value None.
        """

        self._generator = LLM(model=self._config_data["vllm_llm_model_prompt_checker"],
                              trust_remote_code=True,
                              quantization=quantization)

    def filter_unique_prompts(self, prompts: list):
        """ Function for filtering all duplicates from the input prompt list

        :param prompts: a list with input prompts
        :return a list with unique prompts
        """
        self._logger.info(f"\n")
        self._logger.info("*" * 40)
        self._logger.info(" *** Prompt Dataset Cleaner: unique prompts. ***")
        self._logger.info("*" * 40)
        self._logger.info(f"\n")

        for i, p in enumerate(prompts):
            prompts[i] = ' '.join(word.lower() for word in p.split())

        self._logger.info(f" Total lines in the dataset before: {len(prompts)}")

        articles = ["a", "the", "an"]
        prompts = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
        prompts = list(set(prompts))
        prompts = [l + "\n" if "\n" not in l else l for l in prompts]

        self._logger.info(f" Total lines in the dataset after: {len(prompts)}")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return prompts

    def filter_prompts_with_words(self, prompts: list):
        """ Function that filters prompts with undesired words and prompts of certain length that might contain LLM bot output.

        :param prompts: a list with input prompts
        :return list with filtered prompts
        """

        self._logger.info(f"\n")
        self._logger.info("*" * 40)
        self._logger.info(" *** Prompt Dataset Cleaner: filter prompts with undesired words. ***")
        self._logger.info("*" * 40)
        self._logger.info(f"\n")

        self._logger.info(f" Total lines in the dataset before: {len(prompts)}")

        prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
        prompts = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(self._config_data["filter_prompts_with_words"]), prompts))
        prompts = [l + "\n" if "\n" not in l else l for l in prompts]

        self._logger.info(f" Total lines in the dataset after: {len(prompts)}")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return prompts
