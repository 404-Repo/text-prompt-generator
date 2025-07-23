from huggingface_hub import login
from loguru import logger


def initialize(api_key: str) -> None:
    if api_key:
        login(token=api_key)
    else:
        logger.warning("Hugging Face API key was not specified.")
