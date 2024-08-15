import os

from loguru import logger

from generator.generator_backend.groq_backend import GroqBackend
from generator.utils.io_utils import load_config_file


def test_load_model():
    current_dir = os.getcwd()
    generator_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/generator_config.yml"))
    if generator_config["groq_api"]["api_key"] != "":
        generator = GroqBackend(generator_config)
        generator.preload_model(generator_config["groq_api"]["llm_models"][0])
        pipeline_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/pipeline_config.yml"))
        instruction_prompt = pipeline_config["instruction_prompt"]
        instruction_prompt = instruction_prompt.replace("[prompts_number]", str(1))
        prompts = generator.generate(instruction_prompt, ["toys"])

        logger.info(f"Generated prompts: {prompts}")

        assert len(prompts) > 0
