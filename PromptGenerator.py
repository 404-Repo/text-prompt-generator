import sys
import re
import os
import copy
import yaml
import colorama
import random
import tqdm

import spacy
import contextualSpellCheck
import llama_cpp
import groq

from huggingface_hub import hf_hub_download
from time import time


'''
'''
class PromptGenerator:
    def __init__(self):
        colorama.init()
        self.__config_data = self.load_config_file()
        self.__nlp = spacy.load("en_core_web_sm")
        contextualSpellCheck.add_to_pipe(self.__nlp)

    '''
    '''
    def online_generator(self):
        print("*" * 40)
        print("[INFO] *** Prompt Dataset Generator ***")
        print("*" * 40, "\n")

        prompt = self._load_input_prompt()
        object_categories = self._load_object_categories()

        print("[INFO] Started prompt generation.")
        t1 = time()

        client = groq.Groq(api_key=self.__config_data["groq_api_key"])
        output_list = []

        for i in range(self.__config_data["iteration_num"]):
            print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
            prompt_in = copy.copy(prompt)

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
            self.save_prompts(processed_prompts)

        t2 = time()
        duration = (t2 - t1) / 60.0
        print(f"[INFO] It took: {colorama.Fore.GREEN}{duration}{colorama.Style.RESET_ALL} min.")
        print("[INFO] Done.")

    '''
    '''
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
        output_list = []
        for i in range(self.__config_data["iteration_num"]):
            print(f"\n[INFO] Iteration: {colorama.Fore.GREEN}{i}{colorama.Style.RESET_ALL} \n")
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

    '''
    '''
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

    '''
    '''
    def filter_prompts(self):
        print("*" * 40)
        print("[INFO] *** Prompt Dataset Cleaner ***")
        print("*" * 40, "\n")

        with open(self.__config_data["prompts_output_file"], "r") as file:
            prompts = [line.rstrip() for line in file]
            for i, p in enumerate(prompts):
                prompts[i] = ' '.join(word.lower() for word in p.split())
                prompt = self.__nlp(prompts[i])
                prompts[i] = prompt._.outcome_spellCheck

        print(f"[INFO] Total lines in the dataset before: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")

        articles = ["a", "the", "an"]
        prompts = [' '.join(word for word in sentence.split() if word.lower() not in articles) for sentence in prompts]
        prompts = list(set(prompts))
        prompts = list(filter(lambda sentence: 5 <= len(sentence) <= 100, prompts))
        prompts = list(filter(lambda sentence: not set(word.lower() for word in sentence.split()) & set(self.__config_data["filter_prompts_with_words"]), prompts))
        prompts = [p for p in prompts if p not in self.__config_data["filter_colors"]]

        print(f"[INFO] Total lines in the dataset after: {colorama.Fore.GREEN}{len(prompts)}{colorama.Style.RESET_ALL}")

        self.save_prompts(prompts)

    '''
    '''
    @staticmethod
    def load_config_file():
        with open("launching_config.yml", "r") as file:
            config_data = yaml.safe_load(file)
        return config_data

    '''
    
    '''
    def save_prompts(self, prompts_list: list, mode: str = "a"):
        with open(self.__config_data["prompts_output_file"], mode) as file:
            for p in prompts_list:
                file.write("%s" % p)

    '''
    '''
    def _load_input_prompt(self):
        # prompt for dataset generation
        prompt = self.__config_data["prompt_groq"]
        prompt = prompt.replace("prompts_num", str(self.__config_data["prompts_num"]))

        print("[INFO] Input prompt: ")
        regex = re.compile(r'(?<=[^.{}])(\.)(?![{}])'.format("e.g.", "e.g."))
        prompt_printing = regex.sub(r'\1\n', prompt)
        print(f"{colorama.Fore.GREEN}{prompt_printing}{colorama.Style.RESET_ALL}")
        return prompt

    '''
    '''
    def _load_object_categories(self):
        object_categories = self.__config_data['obj_categories']
        print(f"[INFO] Object categories: {colorama.Fore.GREEN}{object_categories}{colorama.Style.RESET_ALL}")
        return object_categories

    '''
    '''
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

    '''
    '''
    @staticmethod
    def _make_lowercase_except_ta(text: str):
        modified_text = ''
        for char in text:
            if char.upper() not in ['T', 'A']:
                modified_text += char.lower()
            else:
                modified_text += char
        return modified_text
