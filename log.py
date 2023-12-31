"""Set global logger settings
"""

import os
import logging
from ile_utils.config import Config
from pathlib import Path

config = Config.get_config()


def get_logger(file_path):
    """Get a named logger with the global logging settings.

    Args:
        name (str): file path of the logger (__file__)

    Returns:
        logger with global settings
    """
    FORMAT = config.get(
        "logging",
        "format",
        fallback="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
    )
    formatter = logging.Formatter(fmt=FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_dir = config.get("logging", "log_dir", fallback="logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_name = Path(file_path).name
    file_handler = logging.FileHandler(f"{log_dir}/{file_name}.log")
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(file_name)
    LEVEL = config.getint("logging", "level", fallback=10)
    if LEVEL % 10 != 0:
        raise ValueError(
            f"Logging level option '{LEVEL}' set to non multiple of 10. See"
            " logging.Levels docs."
        )
    logger.setLevel(LEVEL)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
