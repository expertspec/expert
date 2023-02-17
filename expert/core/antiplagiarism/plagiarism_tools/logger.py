import logging


class CustomFormatter(logging.Formatter):
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"
    bold_red = "\033[38m"
    reset = "\033[31m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        
        return formatter.format(record)


logger = logging.getLogger("Antiplagiat")
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)