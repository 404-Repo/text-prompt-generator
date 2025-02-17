from pathlib import Path

import torch
from generator.generator_backend.vllm_backend import VLLMBackend
from generator.config import load_generator_settings_from_yaml, load_pipeline_settings_from_yaml


current_dir = Path.cwd().parent
generator_settings = load_generator_settings_from_yaml(current_dir.resolve() / "configs" / "generator_config.yml")
pipeline_settings = load_pipeline_settings_from_yaml(current_dir.resolve() / "configs" / "pipeline_config.yml")
generator = VLLMBackend(generator_settings)


def test_load_model():
    generator.load_vllm_model(generator_settings.llm_models[0])
    instruction_prompt = pipeline_settings.instruction_prompt
    instruction_prompt = instruction_prompt.replace("[prompts_number]", str(1))
    prompts = generator.generate(instruction_prompt, ["toys"])

    assert len(prompts) > 0


def test_unload_model():
    _, gpu_memory_total_before = torch.cuda.mem_get_info()
    gpu_available_memory_before = gpu_memory_total_before - torch.cuda.memory_allocated()

    generator.unload_vllm_model()

    _, gpu_memory_total_after = torch.cuda.mem_get_info()
    gpu_available_memory_after = gpu_memory_total_after - torch.cuda.memory_allocated()

    assert gpu_available_memory_before != gpu_available_memory_after
    assert (gpu_memory_total_after - gpu_available_memory_after) / (1024**3) < 10**3
