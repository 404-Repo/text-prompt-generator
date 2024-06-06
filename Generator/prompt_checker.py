import re
import gc
import random
import contextlib
from typing import (Optional,
                    List,
                    Dict)

import groq
import torch
import transformers
from loguru import logger
from huggingface_hub import login
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel


class PromptChecker:
    """
    Class that provides an implementation for different filtering & correction methods for generated prompts.
    """
    def __init__(self, config_file_data: Dict):
        """
        :param config_file_data: a dictionary with preloaded parameters for running the pipeline.
        """

        self._logger = logger
        self._config_data = config_file_data

        if self._config_data["groq_api"]["api_key"] == "":
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
                                       f"Check if the input prompt has a logic between the words. "
                                       f"Remove all digits from the corrected prompt. "
                                       f"On the basis of those checks correct input prompt so it will pass them with the highest score. "
                                       f"Corrected prompt should contain no more than five or six words. "
                                       f"You must always output only corrected prompt and nothing else. ")

        self._prompt_for_checking = (f"Evaluate input prompt and give it a score between 0 (all checks are failed) and 1 (all checks passed). "
                                     f"Use the following checks as a criteria for prompt evaluation. "
                                     f"Perform semantic analysis check of the input prompt. If failed, score it the lowest. "
                                     f"Perform contextual analysis check of the input prompt. If failed, score it the lowest. "
                                     f"Check if all the words in the input prompt makes sense together and describe an object. If failed, score it the lowest. "
                                     f"Check if the input prompt has a logic between the words. If failed, score it the lowest. "
                                     f"Check if the input prompt is finished and has an object or subject in it. If failed, score it the lowest and ignore other checks. "
                                     f"Check if all words in the prompt can be found in a dictionary. If failed, score it the lowest. "
                                     f"You must keep answers short and concise. ")

    def groq_check_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.
        This uses online groq api. Keep in mind that with time the performance will degenerate.

        :param prompt: a string with prompt that will be checked.
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
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
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
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

        client = groq.Groq(api_key=self._config_data["groq_api"]["api_key"])
        output = client.chat.completions.create(messages=[{
                                                            "role": "user",
                                                            "content": prompt
                                                         }],
                                                model=self._config_data["groq_api"]["llm_model_prompt_checker"],
                                                seed=self._config_data['groq_api']['seed'],
                                                temperature=temperature,
                                                top_p=1,
                                                max_tokens=max_tokens)
        result = output.choices[0].message.content

        return result

    def vllm_check_prompt(self, prompt: str, temperature: float = 0.5):
        """  Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.
        This uses online groq api. Keep in mind that with time the performance will degenerate.

        :param prompt: a string with prompt that will be checked.
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return: the evaluation made by the LLM
        """
        object_categories = self._config_data['obj_categories']

        prompt_in = ((f"input prompt: '{prompt}'. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. "
                      f"Be a strict quality judge. Provide single float value as an output. ") +
                     self._prompt_for_checking)

        result = self._vllm_process_prompt(prompt_in, 20, temperature)

        score = re.findall("\d+\.\d+", result)
        return score[0]

    def vllm_correct_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for correcting the input prompt in case if it does not satisfy provided conditions.
        This uses online groq api. Keep in mind that with time the performance will degenerate.

        :param prompt: a string with prompt that will be checked and potentially rewritten.
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return: corrected prompt as a string
        """
        object_categories = self._config_data['obj_categories']
        filter_words = self._config_data["filter_prompts_with_words"]

        prompt_in = ((f"input prompt: {prompt}. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. "
                      f"Avoid using words from the list: {filter_words}. ") +
                     self._prompt_for_correction)
        result = self._vllm_process_prompt(prompt_in, 150, temperature)
        result = result[0].replace("**Corrected Prompt:**", "")
        result = result.replace("**Corrected prompt:**", "")
        result = result.replace("**Corrected prompt:", "")
        result = result.replace("**", "")
        return result

    def _vllm_process_prompt(self, prompt: str, max_tokens: int, temperature: float):
        """  Function for processing prompts according to the passed prompt instruction.

        :param prompt: a string with prompt that will be checked and potentially rewritten
        :param max_tokens: the maximum amount of tokens that will limit the output prompt
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return: processed prompt as a string
        """
        sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)
        outputs = self._generator.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def preload_vllm_model(self, quantization: Optional[str] = None):
        """ Function for preloading LLM model in GPU memory

        :param quantization: optional parameter that defines the quantizaton of the model:
                             "awq", "gptq", "squeezellm", and "fp8" (experimental); Default value None.
        """
        if self._config_data["vllm_api"]['seed'] < 0:
            seed = random.randint(0, int(1e+5))
        else:
            seed = self._config_data["vllm_api"]['seed']

        self._generator = LLM(model=self._config_data["vllm_api"]["llm_model_prompt_checker"],
                              trust_remote_code=True,
                              quantization=quantization,
                              seed=seed)

    def unload_vllm_model(self):
        """ Function for unloading the model """
        logger.info("Deleting model in use.")
        destroy_model_parallel()
        del self._generator.llm_engine.model_executor.driver_worker

        gc.collect()
        torch.cuda.empty_cache()

        self._generator = None

        logger.info(f"GPU allocated memory: {torch.cuda.memory_allocated() / 10000000} Gb\n")
        return torch.cuda.memory_cached(), torch.cuda.memory_allocated()

    def transformers_load_checkpoint(self, load_in_4bit: bool = True, load_in_8bit: bool = False):
        """ Function for pre-loading checkpoints for the requested models using transformers.

        :param load_in_4bit: a boolean parameter that controls whether the model will be loaded using 4 bit quantization (VRAM used ~ 9 Gb).
        :param load_in_8bit: a boolean parameter that controls whether the model will be loaded using 8 bit quantization (VRAM used ~ 18 Gb).
        """

        if load_in_4bit:
            load_in_8bit = False
        elif load_in_8bit:
            load_in_4bit = False
        else:
            load_in_4bit = True
            load_in_8bit = False

        model = self._config_data["transformers_llm_model_prompt_checker"]
        self._generator = transformers.pipeline("text-generation",
                                                model=model,
                                                model_kwargs={
                                                    "torch_dtype": torch.bfloat16,
                                                    "quantization_config": {
                                                        "load_in_4bit": load_in_4bit,
                                                        "load_in_8bit": load_in_8bit
                                                        }
                                                   }
                                                )

    def transformers_check_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.

        :param prompt: a string with prompt that will be checked.
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return a float value between 0 and 1 that will be used for filtering of the prompt.
        """
        if self._generator is None:
            raise ValueError("Transformers pipeline was not initialized by calling transformers_load_checkpoint() function. Abort!")

        object_categories = self._config_data['obj_categories']

        prompt_in = ((f"input prompt: '{prompt}'. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. ") +
                     self._prompt_for_checking)

        result = self._transformers_process_prompt(prompt_in, 100, temperature)
        score = re.findall("\d+\.\d+", result)

        return score[0]

    def transformers_correct_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for correcting the input prompt in case if it does not satisfy provided conditions.

        :param prompt: a string with prompt that will be checked and potentially rewritten.
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return a rewritten prompt as a python string.
        """

        object_categories = self._config_data['obj_categories']
        filter_words = self._config_data["filter_prompts_with_words"]

        prompt_in = ((f"input prompt: {prompt}. "
                      f"This prompt might describe an object from one of these categories: {object_categories}. "
                      f"Avoid using words from the list: {filter_words}. ") +
                     self._prompt_for_correction)

        result = self._transformers_process_prompt(prompt_in, 500, temperature)
        result = result.split("\n")
        result = result[0].replace("**Corrected Prompt:**", "").strip()
        result = result.replace("**Corrected prompt:**", "")
        result = result.replace("**Corrected prompt:", "")
        result = result.replace("**", "")

        return result

    def _transformers_process_prompt(self, prompt: str, max_tokens: int, temperature: float):
        """ Function for processing prompts according to the passed prompt instruction.

        :param prompt: a string with prompt that will be checked and potentially rewritten
        :param max_tokens: the maximum amount of tokens that will limit the output prompt
        :param temperature: value between 0 and 1 that defines how 'inventive' will be the llm
        :return generated prompt
        """

        prompt = self._generator.tokenizer.apply_chat_template(conversation=[{
                                                                    "role": "user",
                                                                    "content": prompt
                                                               }],
                                                               tokenize=False,
                                                               add_generation_prompt=False)
        outputs = self._generator(prompt,
                                  max_new_tokens=max_tokens,
                                  do_sample=True,
                                  temperature=temperature,
                                  top_k=1)

        result = outputs[0]["generated_text"][len(prompt):]
        return result

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

    def filter_prompts_with_words(self, prompts: List[str], words_to_filter: List[str]):
        """ Function that filters prompts with undesired words and prompts of certain length that might contain LLM bot output.

        :param prompts: a list with input prompts
        :param words_to_filter: a list with words that will be used for filtering input prompts
        :return list with filtered prompts
        """

        self._logger.info(f"\n")
        self._logger.info("*" * 40)
        self._logger.info(" *** Prompt Dataset Cleaner: filter prompts with undesired words. ***")
        self._logger.info("*" * 40)
        self._logger.info(f"\n")

        self._logger.info(f" Total lines in the dataset before: {len(prompts)}")

        prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
        prompts = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(words_to_filter), prompts))
        prompts = [l + "\n" if "\n" not in l else l for l in prompts]

        self._logger.info(f" Total lines in the dataset after: {len(prompts)}")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return prompts
