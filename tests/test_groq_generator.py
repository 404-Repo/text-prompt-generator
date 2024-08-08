import os

from generator.backend_generators.groq_generator import GroqGenerator
from generator.utils.io_utils import load_config_file


def test_load_model():
    current_dir = os.getcwd()
    groq_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/groq_config.yml"))
    if groq_config["api_key"] != "":
        generator = GroqGenerator(groq_config)
        generator.preload_model(groq_config["llm_models"][0])
        pipeline_config = load_config_file(os.path.join(os.path.relpath(current_dir), "configs/pipeline_config.yml"))
        instruction_prompt = pipeline_config["instruction_prompt"]
        instruction_prompt = instruction_prompt.replace("[prompts_number]", str(1))
        prompts = generator.generate(instruction_prompt, ["toys"])

        assert len(prompts) > 0
