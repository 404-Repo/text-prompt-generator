import copy
import gc
import os
import secrets
from time import time

import torch
import tqdm
from loguru import logger
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from generator.generator_backend.base_generator_backend import BaseGeneratorBackend


class VLLMBackend(BaseGeneratorBackend):
    def __init__(self, config_data: dict) -> None:
        """
        Parameters
        ----------
        config_data: dictionary with generator configuration
        """
        # vLLM model generator parameters
        self._max_model_len = config_data["vllm_api"]["max_model_len"]
        self._max_tokens = config_data["vllm_api"]["max_tokens"]
        self._temperature = config_data["vllm_api"]["temperature"]
        self._seed = config_data["vllm_api"]["seed"]
        self._top_p = config_data["vllm_api"]["top_p"]
        self._presence_penalty = config_data["vllm_api"]["presence_penalty"]
        self._frequency_penalty = config_data["vllm_api"]["frequency_penalty"]

        # gpu parameters
        self._gpu_memory_utilization = config_data["vllm_api"]["gpu_memory_utilization"]
        self._tensor_parallel_size = config_data["vllm_api"]["tensor_parallel_size"]

        # speculative model parameters
        self._num_speculative_tokens = config_data["vllm_api"]["num_speculative_tokens"]
        self._ngram_prompt_lookup_max = config_data["vllm_api"]["ngram_prompt_lookup_max"]
        self._use_v2_block_manager = config_data["vllm_api"]["use_v2_block_manager"]
        self._speculative_draft_tensor_parallel_size = config_data["vllm_api"]["speculative_draft_tensor_parallel_size"]

        self._generator: LLM | None = None

    def generate(self, instruction_prompt: str, object_categories: list[str]) -> list[str]:
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

        prompt_in = copy.copy(instruction_prompt)
        start_time = time()

        output_prompts = []
        for category in tqdm.tqdm(object_categories, desc="Generating prompts"):
            temperature = secrets.SystemRandom().uniform(self._temperature[0], self._temperature[1])
            prompt_in = prompt_in.replace("[category_name]", category)
            seed = secrets.randbelow(int(1e5)) if self._seed < 0 else self._seed

            sampling_params = self._create_sampling_params(temperature, seed)

            if self._generator is None:
                raise ValueError("vLLM model not initialized.")

            outputs = self._generator.generate([prompt_in], sampling_params, use_tqdm=False)

            prompt_in = prompt_in.replace(category, "[category_name]")
            output_prompts.append(outputs[0].outputs[0].text)

        self._log_generation_time(start_time)

        return output_prompts

    def _create_sampling_params(self, temperature: float, seed: int) -> SamplingParams:
        return SamplingParams(
            n=1,
            presence_penalty=self._presence_penalty,
            frequency_penalty=self._frequency_penalty,
            seed=seed,
            temperature=temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )

    def _log_generation_time(self, start_time: float) -> None:
        duration = (time() - start_time) / 60.0
        logger.info(f" It took: {duration:.2f} min.")
        logger.info(" Done.")
        logger.info("\n")

    def preload_model(self, model_name: str, speculative_model: str = "") -> None:
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face);
        speculative_model: the name of the speculative model or [ngrams];
        """

        if "gemma" in model_name:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

        if speculative_model == "":
            self._generator = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=self._tensor_parallel_size,
                max_model_len=self._max_model_len,
                gpu_memory_utilization=self._gpu_memory_utilization,
                seed=secrets.randbelow(int(1e5)),
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
            )
        elif speculative_model == "[ngram]":
            self._generator = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=self._tensor_parallel_size,
                seed=secrets.randbelow(int(1e5)),
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                speculative_model=speculative_model,
                num_speculative_tokens=self._num_speculative_tokens,
                ngram_prompt_lookup_max=self._ngram_prompt_lookup_max,
                use_v2_block_manager=self._use_v2_block_manager,
            )
        else:
            self._generator = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=self._tensor_parallel_size,
                seed=secrets.randbelow(int(1e5)),
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                speculative_model=speculative_model,
                speculative_draft_tensor_parallel_size=self._speculative_draft_tensor_parallel_size,
                use_v2_block_manager=self._use_v2_block_manager,
            )

    def unload_model(self) -> None:
        """Function for unloading the model"""
        logger.info("Unloading model from GPU VRAM.")

        _, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory_before = gpu_memory_total - torch.cuda.memory_allocated()

        logger.info(f"GPU available memory [before]: {gpu_available_memory_before / 1024 ** 3} Gb")
        destroy_model_parallel()
        if self._generator is None:
            raise ValueError("vLLM model not initialized.")
        del self._generator.llm_engine
        del self._generator

        gc.collect()
        torch.cuda.empty_cache()

        self._generator = None

        _, gpu_memory_total = torch.cuda.mem_get_info()
        gpu_available_memory_after = gpu_memory_total - torch.cuda.memory_allocated()
        logger.info(f"GPU available memory [after]: {gpu_available_memory_after / 1024 ** 3} Gb\n")
