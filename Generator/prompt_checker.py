import re
import tqdm
import torch
import transformers
import groq
import pkg_resources

from loguru import logger
from time import time
from huggingface_hub import login
from symspellpy import SymSpell


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

        transformers.logging.set_verbosity_error()
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

        self._pipeline = None
        self._llama_model = None
        self._llamacpp_model_path = ""

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
        self._pipeline = transformers.pipeline("text-generation",
                                               model=model,
                                               model_kwargs={
                                                    "torch_dtype": torch.bfloat16,
                                                    "quantization_config": {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
                                               })

    def transformers_check_prompt(self, prompt: str, temperature: float = 0.5):
        """ Function for checking the quality of the prompt and outputs the score between 0 and 1 according to the provided checks.

       :param prompt: a string with prompt that will be checked.
       :param temperature:
       :return a float value between 0 and 1 that will be used for filtering of the prompt.
       """
        if self._pipeline is None:
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
        :param temperature:
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
        prompt = self._pipeline.tokenizer.apply_chat_template(conversation=[{
                                                                                "role": "user",
                                                                                "content": prompt
                                                                            }],
                                                              tokenize=False,
                                                              add_generation_prompt=True)
        outputs = self._pipeline(prompt,
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

    def check_grammar(self, prompts: list):
        """ Function for checking the words in prompts for spelling errors

        :param prompts: a list with strings (generated prompts)
        :return a list with processed prompts stored as strings.
        """

        self._logger.info(" Performing spell check of the generated prompts.")
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

        self._logger.info(f" It took: {duration} min.")
        self._logger.info(" Done.")
        self._logger.info(f"\n")

        return corrected_prompts
