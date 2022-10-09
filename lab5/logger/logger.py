import logging
from logging.config import fileConfig

fileConfig("logger/logger.ini")
base_logger = logging.getLogger("base")
