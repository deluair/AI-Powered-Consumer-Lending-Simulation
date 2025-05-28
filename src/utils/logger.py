import logging
import sys
from ..config import LOG_LEVEL, LOG_FILE

def get_logger(name: str) -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if logger already configured
    if logger.hasHandlers():
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

# Example usage:
# from src.utils.logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message.")