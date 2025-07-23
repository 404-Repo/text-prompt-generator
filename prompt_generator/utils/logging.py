from time import time
from loguru import logger


def log_duration(task: str, start_time: float) -> None:
    elapsed = time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    logger.info(f"{task} {minutes}:{seconds:02} minutes")
