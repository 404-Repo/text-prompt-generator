import sys
import re
import os
import copy
import yaml
import colorama
import random
import tqdm

import llama_cpp
import groq

from huggingface_hub import hf_hub_download
from symspellpy import SymSpell
import pkg_resources
from time import time


class PromptGenerator:
    """ Class that implements a text prompt generator for creating prompts for generating 3D models.
    It provides access to two different LLM APIs:
    1) Groq (online) - platform that provides access to three LLM models with quick inference
    2) Offline LLM - slower than Groq but any LLM model can be plugged in that is compatible with llama-cpp
    """
    def __init__(self):
        colorama.init()
        self.__config_data = self.load_config_file()

    """ Function that calls Groq api for generating requested output. All supported by Groq models are supported. """
    def online_generator(self):
        print("\n", "*" * 40)
        print("[INFO] *** Prompt Dataset Generator ***")
        print("*" * 40, "\n")

        prompt = self._load_input_prompt()
        object_categories = self._load_object_categories()

        print("[INFO] Started prompt generation.")
        t1 = time()

        client = groq.Groq(api_key=self.__config_data["groq_api_key"])

        for i in range(self.__config_data["iteration_num"]):
            print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
            prompt_in = copy.copy(prompt)

            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.45, 0.65)
                # print(f"[INFO] Current temperature: {colorama.Fore.GREEN}{temperature}{colorama.Style.RESET_ALL}")

                # find 'member' in the input string and replace it with category
                prompt_in = prompt_in.replace("member_placeholder", category)

                if self.__config_data['llm_model']['seed'] < 0:
                    seed = random.randint(0, sys.maxsize)
                else:
                    seed = self.__config_data['llm_model']['seed']

                output = client.chat.completions.create(messages=[
                                                                    {
                                                                        "role": "user",
                                                                        "content": prompt_in
                                                                    }
                                                                 ],
                                                        model=self.__config_data["groq_llm_model"],
                                                        temperature=temperature,
                                                        seed=seed,
                                                        top_p=1,
                                                        max_tokens=self.__config_data["llm_model"]["max_tokens"])

                prompt_in = prompt_in.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_list.append(output.choices[0].message.content)

            processed_prompts = self.post_process_prompts(output_list)
            checked_prompts = self.check_grammar(processed_prompts)
            self.save_prompts(checked_prompts)

        t2 = time()
        duration = (t2 - t1) / 60.0
        print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        print("[INFO] Done.")

    """ llama-cpp loader for LLM models. LLM models should be stored in .gguf file format. """
    def offline_generator(self):
        print("*" * 40)
        print("[INFO] *** Prompt Dataset Generator ***")
        print("*" * 40, "\n")

        model_path = self._load_offline_model()

        # init the llm model using Llama pipeline
        print("[INFO] Preparing model.")
        if self.__config_data['llm_model']['seed'] < 0:
            seed = random.randint(0, sys.maxsize)
        else:
            seed = self.__config_data['llm_model']['seed']

        llm_model = llama_cpp.Llama(model_path=model_path,
                                    seed=seed,
                                    n_ctx=self.__config_data['llm_model']['n_ctx'],
                                    last_n_tokens_size=self.__config_data['llm_model']['last_n_tokens_size'],
                                    n_threads=self.__config_data['llm_model']['n_threads'],
                                    n_gpu_layers=self.__config_data['llm_model']['n_gpu_layers'],
                                    verbose=self.__config_data['llm_model']['verbose'])
        print("[INFO] Done.\n")

        prompt = self._load_input_prompt()
        object_categories = self._load_object_categories()

        # defining the grammar for the LLM model -> forcing to output strings according to specified rules
        grammar = llama_cpp.LlamaGrammar.from_string(r'''root ::= items
                                                             items ::= item ("," ws* item)*
                                                             item ::= string
                                                             string  ::= "\"" word (ws+ word)* "\"" ws*
                                                             word ::= [a-zA-Z]+
                                                             ws ::= " "
                                                          ''', verbose=self.__config_data['llm_model']['verbose'])

        # generate prompts using the provided object categories
        print("[INFO] Started prompt generation.")
        t1 = time()
        for i in range(self.__config_data["iteration_num"]):
            print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
            output_list = []
            for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
                temperature = random.uniform(0.45, 0.65)
                # print(f"[INFO] Current temperature: {colorama.Fore.GREEN}{temperature}{colorama.Style.RESET_ALL}")

                # find 'member' in the input string and replace it with category
                prompt = prompt.replace("member_placeholder", category)
                output = llm_model.create_completion(prompt=prompt,
                                                     max_tokens=self.__config_data['llm_model']['max_tokens'],
                                                     seed=seed,
                                                     echo=False,
                                                     grammar=grammar,
                                                     temperature=temperature)
                prompt = prompt.replace(category, "member_placeholder")

                # extracting the response of the llm model: generated prompts
                output_list.append(output['choices'][0]['text'])

            processed_prompts = self.post_process_prompts(output_list)
            self.save_prompts(processed_prompts)

        t2 = time()
        duration = (t2 - t1) / 60.0
        print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        print("[INFO] Done.")

    """ Function for post processing the generated prompts. The LLM output is filtered from punctuation symbols and all non alphabetic characters.
    :param prompt_list: a list with strings (generated prompts)
    :return a list with processed prompts stored as strings.
    """
    def post_process_prompts(self, prompts_list: list):
        result_prompts = []
        for el in prompts_list:
            lines = el.split(".")
            processed_lines = []
            for i in range(len(lines)):
                line = re.sub(r'[^a-zA-Z`\s-]', '', lines[i])
                line = self._make_lowercase_except_ta(line)
                split_line = re.findall(r'[A-Z][^A-Z]*', line)
                final_lines = [split_line[j] + split_line[j + 1] if j + 1 < len(split_line) and split_line[j + 1][0].islower() else split_line[j]
                               for j in range(len(split_line)) if split_line[j][0].isupper()]
                final_lines = [l + "\n" if "\n" not in l else l for l in final_lines]
                processed_lines += final_lines
            result_prompts += processed_lines

        return result_prompts

    """ Function for filtering the prompts: removing prompts with certain words and prompts of certain length. """
    def filter_prompts(self):
        print("*" * 40)
        print("[INFO] *** Prompt Dataset Cleaner ***")
        print("*" * 40, "\n")

        with open(self.__config_data["prompts_output_file"], "r") as file:
            prompts = [line.rstrip() for line in file]
            for i, p in enumerate(prompts):
                prompts[i] = ' '.join(word.lower() for word in p.split())

        print(f"[INFO] Total lines in the dataset before: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")

        articles = ["a", "the", "an"]
        prompts = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
        prompts = list(set(prompts))
        prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
        prompts = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(self.__config_data["filter_prompts_with_words"]), prompts))
        prompts = [p for p in prompts if p not in self.__config_data["filter_colors"]]
        prompts = [l + "\n" if "\n" not in l else l for l in prompts]

        print(f"[INFO] Total lines in the dataset after: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")

        self.save_prompts(prompts, "w")

    """ Function for checking the words in prompts for spelling errors
    :param prompts: a list with strings (generated prompts)
    :return a list with processed prompts stored as strings.
    """
    @staticmethod
    def check_grammar(prompts: list):
        print("[INFO] Performing spell check of the generated prompts")
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
        print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        return corrected_prompts

    """ Function for loading parameters for running the LLM """
    @staticmethod
    def load_config_file():
        with open("launching_config.yml", "r") as file:
            config_data = yaml.safe_load(file)
        return config_data

    """ Function for loading the prompts dataset for processing.
    :return list with loaded prompts
    """
    def load_file_with_prompts(self):
        with open(self.__config_data["prompts_output_file"], "r") as file:
            prompts = [line.rstrip() for line in file]
        return prompts

    """ Function for saving the prompts stored in the prompts list
    :param prompts_list: a list with strings (generated prompts)
    :param mode: mode for writing the file: 'a', 'w'
    """
    def save_prompts(self, prompts_list: list, mode: str = "a"):
        with open(self.__config_data["prompts_output_file"], mode) as file:
            for p in prompts_list:
                file.write("%s" % p)

    """Function for loading input prompt-instruction for the LLM. It will be used for generating prompts.
    :return loaded prompt as a string from the config file.
    """
    def _load_input_prompt(self):
        # prompt for dataset generation
        prompt = self.__config_data["prompt_groq"]
        prompt = prompt.replace("prompts_num", str(self.__config_data["prompts_num"]))

        print("[INFO] Input prompt: ")
        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)
        print(f"{colorama.Fore.GREEN}{prompt_printing}{colorama.Style.RESET_ALL}")
        return prompt

    """Function for loading object categories from where the LLM will sample objects' names for generating prompts
    :return a list with stored object categories names.
    """
    def _load_object_categories(self):
        object_categories = self.__config_data['obj_categories']
        print(f"[INFO] Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")
        return object_categories

    """ Function for loading (including downloading from hugging face) the requested LLM for offline generations. """
    def _load_offline_model(self):
        # model to pick up from the hugging face (should have .gguf extension to run with llama)
        hf_model_repo = self.__config_data["hugging_face_repo"]
        print(f"[INFO] Hugging Face repository: {colorama.Fore.GREEN}{hf_model_repo}{colorama.Style.RESET_ALL}")

        # the name of the file to be downloaded
        model_file_name = self.__config_data["llm_model_file_name"]
        print(f"[INFO] LLM model to load: {colorama.Fore.GREEN}{model_file_name}{colorama.Style.RESET_ALL}")

        # cache folder where you want to store the downloaded model
        cache_folder = self.__config_data["cache_folder"]
        os.makedirs(cache_folder, exist_ok=True)
        print(f"[INFO] LLM model will be stored here: {colorama.Fore.GREEN}{cache_folder}{colorama.Style.RESET_ALL}")

        model_path = hf_hub_download(repo_id=hf_model_repo, filename=model_file_name, cache_dir=cache_folder, local_files_only=True)
        print(f"[INFO] Downloaded model stored in: {colorama.Fore.GREEN}{model_path}{colorama.Style.RESET_ALL} \n")

        return model_path

    """ Function that bring to lower case all letters except T and A; helper function for filtering. """
    @staticmethod
    def _make_lowercase_except_ta(text: str):
        modified_text = ''
        for char in text:
            if char.upper() not in ['T', 'A']:
                modified_text += char.lower()
            else:
                modified_text += char
        return modified_text
