import os
import torch

from generator.backend_generators.vllm_generator import VLLMGenerator
from generator.utils.io_utils import load_config_file

current_dir = os.getcwd()
generator_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/generator_config.yml"))
generator = VLLMGenerator(generator_config)


def test_load_model():
    generator.preload_model(generator_config["vllm"]["llm_models"][0])
    pipeline_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/pipeline_config.yml"))
    instruction_prompt = pipeline_config["instruction_prompt"]
    instruction_prompt = instruction_prompt.replace("[prompts_number]", str(1))
    prompts = generator.generate(instruction_prompt, ["toys"])

    assert len(prompts) > 0


def test_unload_model():
    _, gpu_memory_total_before = torch.cuda.mem_get_info()
    gpu_available_memory_before = gpu_memory_total_before - torch.cuda.memory_allocated()

    generator.unload_model()

    _, gpu_memory_total_after = torch.cuda.mem_get_info()
    gpu_available_memory_after = gpu_memory_total_after - torch.cuda.memory_allocated()

    assert gpu_available_memory_before != gpu_available_memory_after
    assert (gpu_memory_total_after - gpu_available_memory_after)/(1024**3) < 10**3
