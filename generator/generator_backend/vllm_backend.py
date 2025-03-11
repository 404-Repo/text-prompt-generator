import copy
import gc
import os
import secrets
from time import time

import torch
import tqdm
from loguru import logger
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from generator.config import GeneratorSettings
from generator.utils import get_available_gpu_memory


def _log_generation_time(start_time: float) -> None:
    duration = (time() - start_time) / 60.0
    logger.info(f" It took: {duration:.2f} min.")
    logger.info(" Done.")
    logger.info("\n")


class VLLMBackend:
    def __init__(self, vllm_settings: GeneratorSettings) -> None:
        # vLLM model generator parameters
        self._max_model_len = vllm_settings.max_model_len
        self._max_tokens = vllm_settings.max_tokens
        self._temperature = vllm_settings.temperature
        self._seed = vllm_settings.seed
        self._top_p = vllm_settings.top_p
        self._presence_penalty = vllm_settings.presence_penalty
        self._frequency_penalty = vllm_settings.frequency_penalty
        self._model_name = ""

        # gpu parameters
        self._gpu_memory_utilization = vllm_settings.gpu_memory_utilization
        self._tensor_parallel_size = vllm_settings.tensor_parallel_size

        # speculative model parameters
        self._num_speculative_tokens = vllm_settings.num_speculative_tokens
        self._ngram_prompt_lookup_max = vllm_settings.ngram_prompt_lookup_max
        self._use_v2_block_manager = vllm_settings.use_v2_block_manager
        self._speculative_draft_tensor_parallel_size = vllm_settings.speculative_draft_tensor_parallel_size

        self._speculative_model = vllm_settings.speculative_model

        self._generator: LLM | None = None

    @staticmethod
    def _apply_conversation_template(prompt: str) -> list[ChatCompletionMessageParam]:
        return [
            ChatCompletionSystemMessageParam(
                role="system", content="You are a helpful assistant precisely following the instruction."
            ),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

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

            # chat_template = self._generator.get_tokenizer().chat_template
            tokeniser = self._generator.get_tokenizer()

            if hasattr(tokeniser, "chat_template"):
                conversation = self._apply_conversation_template(prompt_in)
                outputs = self._generator.chat(messages=conversation, sampling_params=sampling_params, use_tqdm=False)
                output_prompts.append(outputs[0].outputs[0].text)
            else:
                outputs = self._generator.generate([prompt_in], sampling_params=sampling_params, use_tqdm=False)
                output_prompts.append(outputs[0].outputs[0].text)

            prompt_in = prompt_in.replace(category, "[category_name]")

        _log_generation_time(start_time)

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

    def load_vllm_model(self, model_name: str) -> None:
        """
        Function for preloading LLM model in GPU memory

        Parameters
        ----------
        model_name: the name of the model from the HF (hugging face);
        """

        if "gemma" in model_name:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

        self._model_name = model_name

        if self._speculative_model == "":
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
        elif self._speculative_model == "[ngram]":
            self._generator = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=self._tensor_parallel_size,
                seed=secrets.randbelow(int(1e5)),
                gpu_memory_utilization=self._gpu_memory_utilization,
                max_model_len=self._max_model_len,
                speculative_model=self._speculative_model,
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
                speculative_model=self._speculative_model,
                speculative_draft_tensor_parallel_size=self._speculative_draft_tensor_parallel_size,
                use_v2_block_manager=self._use_v2_block_manager,
            )

    def unload_vllm_model(self) -> None:
        logger.info(f"Unloading VLLM model. VRAM available: {get_available_gpu_memory() / 1024 ** 3} Gb")
        destroy_model_parallel()
        self._generator = None
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"VLLM model unloaded. VRAM available: {get_available_gpu_memory() / 1024 ** 3} Gb")
