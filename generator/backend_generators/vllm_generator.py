import gc
import copy
import random
import contextlib
from time import time
from typing import Optional, List

import torch
import tqdm
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from loguru import logger


class VLLMGenerator:
    def __init__(self, config_data: dict):
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        self._max_tokens = config_data["max_tokens"]
        self._seed = config_data['seed']
        self._max_model_len = config_data["max_model_len"]
        self._temperature = config_data["temperature"]

        self._generator = None

    def generate(self, instruction_prompt: str,  object_categories: List[str]):
        """
        Function that calls vLLM API for generating prompts.

        Parameters
        ----------
        instruction_prompt: a string with instruction prompt
        object_categories: a list of strings with categories

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

            sampling_params = SamplingParams(n=1, seed=seed, temperature=temperature, max_tokens=self._max_tokens)
            outputs = self._generator.generate([prompt_in], sampling_params, use_tqdm=False)

            prompt_in = prompt_in.replace(category, "[category_name]")
            output_prompts.append(outputs[0].outputs[0].text)

        t2 = time()
        duration = (t2 - t1) / 60.0
        logger.info(f" It took: {duration} min.")
        logger.info(" Done.")
        logger.info(f"\n")

        return output_prompts

    def preload_model(self, model_name: str,  quantization: Optional[str] = None):
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face)
        quantization: optional parameter that defines the quantizaton of the model:
                      "awq", "gptq", "squeezellm", and "fp8" (experimental); Default value None.
        """

        self._generator = LLM(model=model_name,
                              trust_remote_code=True,
                              quantization=quantization,
                              max_model_len=self._max_model_len)

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
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()

        self._generator = None

        _, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory_after = gpu_memory_total - torch.cuda.memory_allocated()
        logger.info(f"GPU available memory [after]: {gpu_available_memory_after / 1024 ** 3} Gb\n")

    @staticmethod
    def quantize_model_awq(model_path_hf: str, quant_model_path: str):
        """
        Function for quantizing the LLM model using AWQ library

        Parameters
        ----------
        model_path_hf: path to the higging face library
        quant_model_path: output path of the quantozed model

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
