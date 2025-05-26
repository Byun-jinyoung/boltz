import os, sys
import logging
from colorama import Fore, Style

class MyCustomFormatter(logging.Formatter):
    # log level emoji
    EMOJIS = {
        "DEBUG": "🔍 ",
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ ",
        "CRITICAL": "🔥 ",
    }
    # log level color
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        # Add emoji
        emoji = self.EMOJIS.get(record.levelname, "")
        record.levelname = f"{emoji}{record.levelname}"        
        
        # 로그 레벨에 따라 색상을 적용
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        
        return super().format(record)

def get_logger():
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # file handler
    file_handler = logging.FileHandler('my.log')
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
