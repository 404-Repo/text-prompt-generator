import gc
import copy
import random
import os
from time import time
from typing import List

import torch
import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from loguru import logger

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class VLLMBackend(BaseGeneratorBackend):
    def __init__(self, config_data: dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["vllm_api"]["max_tokens"]
        self._seed = config_data["vllm_api"]['seed']
        self._top_p = config_data["vllm_api"]["top_p"]
        self._gpu_memory_utilization = config_data["vllm_api"]["gpu_memory_utilization"]
        self._tensor_parallel_size = config_data["vllm_api"]["tensor_parallel_size"]
        self._max_model_len = config_data["vllm_api"]["max_model_len"]
        self._temperature = config_data["vllm_api"]["temperature"]
        self._num_speculative_tokens = config_data["vllm_api"]["num_speculative_tokens"]
        self._ngram_prompt_lookup_max = config_data["vllm_api"]["ngram_prompt_lookup_max"]
        self._use_v2_block_manager = config_data["vllm_api"]["use_v2_block_manager"]
        self._speculative_draft_tensor_parallel_size = config_data["vllm_api"]["speculative_draft_tensor_parallel_size"]

        self._generator = None

    def generate(self, instruction_prompt: str,  object_categories: List[str]):
        """
        Function that calls vLLM API for generating prompts.

        Parameters
        ----------
        instruction_prompt: a string with instruction prompt;
        object_categories: a list of strings with categories;

        Returns
        -------
        output_prompts: list with generated prompts
        """

        # generate prompts using the provided object categories
        prompt_in = copy.copy(instruction_prompt)
        t1 = time()

        output_prompts = []
        for category, _ in zip(object_categories, tqdm.trange(len(object_categories))):
            temperature = random.uniform(self._temperature[0], self._temperature[1])

            # find 'member' in the input string and replace it with category
            prompt_in = prompt_in.replace("[category_name]", category)

            if self._seed < 0:
                seed = random.randint(0, int(1e+5))
            else:
                seed = self._seed

            sampling_params = SamplingParams(n=1,
                                             seed=seed,
                                             temperature=temperature,
                                             max_tokens=self._max_tokens,
                                             top_p=self._top_p)
            outputs = self._generator.generate([prompt_in], sampling_params, use_tqdm=False)

            prompt_in = prompt_in.replace(category, "[category_name]")
            output_prompts.append(outputs[0].outputs[0].text)

        t2 = time()
        duration = (t2 - t1) / 60.0
        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info(f"\n")

        return output_prompts

    def preload_model(self, model_name: str, speculative_model: str = ""):
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face);
        speculative_model: the name of the speculative model or [ngrams];
        """

        if "gemma" in model_name:
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
        else:
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

        if speculative_model == "":
            self._generator = LLM(model=model_name,
                                  trust_remote_code=True,
                                  tensor_parallel_size=1,
                                  max_model_len=self._max_model_len,
                                  gpu_memory_utilization=self._gpu_memory_utilization,
                                  seed=random.randint(0, int(1e+5)))
        elif speculative_model == "[ngram]":
            self._generator = LLM(model=model_name,
                                  trust_remote_code=True,
                                  tensor_parallel_size=self._tensor_parallel_size,
                                  seed=random.randint(0, int(1e+5)),
                                  gpu_memory_utilization=self._gpu_memory_utilization,
                                  max_model_len=self._max_model_len,
                                  speculative_model=speculative_model,
                                  num_speculative_tokens=self._num_speculative_tokens,
                                  ngram_prompt_lookup_max=self._ngram_prompt_lookup_max,
                                  use_v2_block_manager=self._use_v2_block_manager)
        else:
            self._generator = LLM(model=model_name,
                                  trust_remote_code=True,
                                  tensor_parallel_size=self._tensor_parallel_size,
                                  seed=random.randint(0, int(1e+5)),
                                  gpu_memory_utilization=self._gpu_memory_utilization,
                                  max_model_len=self._max_model_len,
                                  speculative_model=speculative_model,
                                  speculative_draft_tensor_parallel_size=self._speculative_draft_tensor_parallel_size,
                                  use_v2_block_manager=self._use_v2_block_manager)

    def unload_model(self):
        """ Function for unloading the model """
        logger.info("Unloading model from GPU VRAM.")

        _, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory_before = gpu_memory_total - torch.cuda.memory_allocated()

        logger.info(f"GPU available memory [before]: {gpu_available_memory_before / 1024 ** 3} Gb")
        destroy_model_parallel()
        del self._generator.llm_engine
        del self._generator

        gc.collect()
        torch.cuda.empty_cache()

        self._generator = None

        _, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory_after = gpu_memory_total - torch.cuda.memory_allocated()
        logger.info(f"GPU available memory [after]: {gpu_available_memory_after / 1024 ** 3} Gb\n")
