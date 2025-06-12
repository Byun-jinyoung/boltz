import os, sys
"""Logging configuration utilities for the Boltz package."""

import logging
from typing import ClassVar, Dict

from colorama import Fore, Style


class MyCustomFormatter(logging.Formatter):
    """Formatter that adds colours and emojis to log level names."""

    # log level emoji
    EMOJIS: ClassVar[Dict[str, str]] = {
        "DEBUG": "🔍 ",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
        "CRITICAL": "🔥 ",
    }
    # log level color
    COLORS: ClassVar[Dict[int, str]] = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        """Format the specified record without mutating it for other handlers."""
        orig_levelname = record.levelname

        # Add emoji and apply colour
        emoji = self.EMOJIS.get(orig_levelname, "")
        colour = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{colour}{emoji}{orig_levelname}{Style.RESET_ALL}"

        message = super().format(record)

        # Restore original levelname so other handlers see an unmodified record
        record.levelname = orig_levelname

        return message

def get_logger():
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # file handler
    file_handler = logging.FileHandler("my.log")
    file_handler.setLevel(logging.DEBUG)

    # configure formatter
    formatter = MyCustomFormatter(
        "[%(asctime)s | %(levelname)s ] %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

MyLogger = get_logger()
