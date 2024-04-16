import re
import sys
import tqdm
import logging
import colorama

import torch
import transformers
import groq

from time import time
from huggingface_hub import login
from symspellpy import SymSpell
import pkg_resources


class PromptChecker:
    """
    Class that provides an implementation for different filtering & correction methods for generated prompts.
    """

    """
    :param config_file_data: a dictionary with preloaded parameters for running the pipeline.
    """
    def __init__(self, config_file_data: dict):
        colorama.init()
        transformers.logging.set_verbosity_error()
        self.__config_data = config_file_data
        self._init_logger()
        self.__logger = logging.getLogger("app2")

        if self.__config_data["groq_api_key"] == "":
            self.__logger.setLevel(logging.WARNING)
            self.__logger.warning(f"{colorama.Fore.RED} Groq Api Access Token was not specified. "
                                  f"You will not be able to use Groq API without it.{colorama.Style.RESET_ALL}")
            self.__logger.setLevel(logging.INFO)

        # login to hugging face platform using api token
        if self.__config_data["hugging_face_api_key"] == "":
            self.__logger.setLevel(logging.WARNING)
            self.__logger.warning(f"{colorama.Fore.RED} Hugging Face Api Access Token was not specified. "
                                  f"You will not be able to download Gemma model.{colorama.Style.RESET_ALL}")
            self.__logger.setLevel(logging.INFO)
        else:
            login(token=self.__config_data["hugging_face_api_key"])
        self.__pipeline = None

    """ Initializing custom logger """
    @staticmethod
    def _init_logger():
        logger = logging.getLogger("app2")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.
    This uses online groq api. Keep in mind that with time the performance will degenerate.
    :param prompt: a string with prompt that will be checked.
    :return a float value between 0 and 1 that will be used for filtering of the prompt. 
    """
    def groq_check_prompt(self, prompt: str):

        object_categories = self.__config_data['obj_categories']

        prompt_in = (f"input prompt: '{prompt}'. "
                     f"This prompt might describe an object from one of these categories: {object_categories}. "
                     f"Perform semantic analysis check of the input prompt. If failed, score it the lowest. "
                     f"Perform contextual analysis check of the input prompt. If failed, score it the lowest. "
                     f"Check if all the words in the input prompt makes sense together and describe an object. If failed, score it the lowest. "
                     f"Check if the input prompt has a logic between the words. If failed, score it the lowest. "
                     f"Check if the input prompt is finished and has an object or subject in it. If not, score it the lowest and ignore other checks. "
                     f"Check if all words in the prompt can be found in a dictionary. If not, score it the lowest. "
                     f"Use performed checks to score the input prompt between 0 (all checks are failed) and 1 (all checks passed). "
                     f"You must keep answers short and concise. "
                     f"You must always output only a single float digit. ")

        client = groq.Groq(api_key=self.__config_data["groq_api_key"])
        output = client.chat.completions.create(messages=[{
            "role": "user",
            "content": prompt_in
        }],
            model="gemma-7b-it",
            seed=self.__config_data['llm_model']['seed'],
            temperature=0.5,
            top_p=1,
            max_tokens=100)
        result = output.choices[0].message.content
        score = re.findall("\d+\.\d+", result)

        return score[0]

    """ Function for correcting the input prompt in case if it does not satisfy provided conditions.
    This uses online groq api. Keep in mind that with time the performance will degenerate.
    :param prompt: a string with prompt that will be checked and potentially rewritten.
    :return a rewritten prompt as a python string. 
    """
    def groq_correct_prompt(self, prompt: str, temperature: float = 0.5):
        object_categories = self.__config_data['obj_categories']
        filter_words = self.__config_data["filter_prompts_with_words"]

        prompt_in = (f"input prompt: {prompt}. "
                     f"This prompt might describe an object from one of these categories: {object_categories}. "
                     f"Avoid using words from the list: {filter_words}. "
                     f"Perform semantic analysis check of the input prompt. "
                     f"Perform contextual analysis check of the input prompt. "
                     f"Remove all digits from the corrected prompt. "
                     f"On the basis of those checks correct input prompt so it will pass them with the highest score. "
                     f"Corrected prompt should contain no more than five or six words. "
                     f"You must always output only corrected prompt and nothing else. ")

        client = groq.Groq(api_key=self.__config_data["groq_api_key"])
        output = client.chat.completions.create(messages=[{
                                                            "role": "user",
                                                            "content": prompt_in
                                                          }],
                                                model="gemma-7b-it",
                                                seed=self.__config_data['llm_model']['seed'],
                                                temperature=temperature,
                                                top_p=1,
                                                max_tokens=500)
        result = output.choices[0].message.content

        result = result.split("\n")
        result = result[0].replace("**Corrected Prompt:**", "").strip()

        return result

    """ Function for pre-loading checkpoints for the requested models using transformers.
    :param load_in_4bit: a boolean parameter that controls whether the model will be loaded using 4 bit quantization (VRAM used ~ 9 Gb).
    :param load_in_8bit: a boolean parameter that controls whether the model will be loaded using 8 bit quantization (VRAM used ~ 18 Gb). 
    """
    def transformers_load_checkpoint(self, load_in_4bit: bool = True, load_in_8bit: bool = False):
        if load_in_4bit:
            load_in_8bit = False
        elif load_in_8bit:
            load_in_4bit = False
        else:
            load_in_4bit = True
            load_in_8bit = False

        model = self.__config_data["transformers_llm_model_prompt_checker"]
        self.__pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "quantization_config": {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            },
        )

    """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.
    :param prompt: a string with prompt that will be checked.
    :return a float value between 0 and 1 that will be used for filtering of the prompt. 
    """
    def transformers_check_prompt(self, prompt: str):
        if self.__pipeline is None:
            raise ValueError("Transformers pipeline was not initialized by calling transformers_load_checkpoint() function. Abort!")

        object_categories = self.__config_data['obj_categories']

        prompt_in = (f"input prompt: '{prompt}'. "
                     f"This prompt might describe an object from one of these categories: {object_categories}. "
                     f"Perform semantic analysis check of the input prompt. If failed, score it the lowest. "
                     f"Perform contextual analysis check of the input prompt. If failed, score it the lowest. "
                     f"Check if all the words in the input prompt makes sense together and describe an object. If failed, score it the lowest. "
                     f"Check if the input prompt has a logic between the words. If failed, score it the lowest. "
                     f"Check if the input prompt is finished and has an object or subject in it. If not, score it the lowest and ignore other checks. "
                     f"Check if all words in the prompt can be found in a dictionary. If not, score it the lowest. "
                     f"Use performed checks to score the input prompt between 0 (all checks are failed) and 1 (all checks passed). "
                     f"You must keep answers short and concise. "
                     f"You must always output only a single float digit. ")

        prompt = self.__pipeline.tokenizer.apply_chat_template(conversation=[{
                                                                                "role": "user",
                                                                                "content": prompt_in
                                                                            }],
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        outputs = self.__pipeline(prompt,
                                  max_new_tokens=100,
                                  do_sample=True,
                                  temperature=0.5,
                                  top_k=1)

        result = outputs[0]["generated_text"][len(prompt):]
        score = re.findall("\d+\.\d+", result)

        return score[0]

    """  Function for correcting the input prompt in case if it does not satisfy provided conditions.
    :param prompt: a string with prompt that will be checked and potentially rewritten.
    :return a rewritten prompt as a python string. 
    """
    def transformers_correct_prompt(self, prompt: str, temperature: float = 0.5):
        object_categories = self.__config_data['obj_categories']
        filter_words = self.__config_data["filter_prompts_with_words"]

        prompt_in = (f"input prompt: {prompt}. "
                     f"This prompt might describe an object from one of these categories: {object_categories}. "
                     f"Avoid using words from the list: {filter_words}. "
                     f"Perform semantic analysis check of the input prompt. "
                     f"Perform contextual analysis check of the input prompt. "
                     f"Remove all digits from the corrected prompt. "
                     f"On the basis of those checks correct input prompt so it will pass them with the highest score. "
                     f"Corrected prompt should contain no more than five or six words. "
                     f"You must always output only corrected prompt and nothing else. ")

        prompt = self.__pipeline.tokenizer.apply_chat_template(conversation=[{
                                                                                "role": "user",
                                                                                "content": prompt_in
                                                                            }],
                                                               tokenize=False,
                                                               add_generation_prompt=True)
        outputs = self.__pipeline(prompt,
                                  max_new_tokens=500,
                                  do_sample=True,
                                  temperature=temperature,
                                  top_k=1)

        result = outputs[0]["generated_text"][len(prompt):]
        result = result.split("\n")
        result = result[0].replace("**Corrected Prompt:**", "").strip()
        result = result.replace("**Corrected prompt:**", "")

        return result

    """ Function for filtering the prompts: removing prompts with certain words and prompts of certain length. """
    def filter_prompts(self, prompts):
        self.__logger.info(f"\n")
        self.__logger.info("*" * 40)
        self.__logger.info(" *** Prompt Dataset Cleaner ***")
        self.__logger.info("*" * 40)
        self.__logger.info(f"\n")

        for i, p in enumerate(prompts):
            prompts[i] = ' '.join(word.lower() for word in p.split())

        self.__logger.info(f" Total lines in the dataset before: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")

        articles = ["a", "the", "an"]
        prompts = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
        prompts = list(set(prompts))
        prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
        prompts = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(self.__config_data["filter_prompts_with_words"]), prompts))
        prompts = [p for p in prompts if p not in self.__config_data["filter_colors"]]
        prompts = [l + "\n" if "\n" not in l else l for l in prompts]

        self.__logger.info(f" Total lines in the dataset after: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        return prompts

    """ Function for checking the words in prompts for spelling errors
    :param prompts: a list with strings (generated prompts)
    :return a list with processed prompts stored as strings.
    """
    def check_grammar(self, prompts: list):
        self.__logger.info(" Performing spell check of the generated prompts.")
        t1 = time()

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )

        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        corrected_prompts = []
        for i in tqdm.trange(len(prompts)):
            terms = sym_spell.lookup_compound(prompts[i], max_edit_distance=2)
            prompt = terms[0].term
            if "\n" not in prompt:
                prompt += "\n"
            corrected_prompts.append(prompt)

        t2 = time()
        duration = (t2 - t1) / 60.0
        self.__logger.info(f" It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        self.__logger.info(" Done.")
        self.__logger.info(f"\n")

        return corrected_prompts
