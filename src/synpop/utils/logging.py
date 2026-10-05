import logging
from logging.handlers import QueueHandler


def setup_worker_logger(queue):
    """
    Configures logging for multiprocessing workers by routing logs
    to a shared queue instead of writing directly to a file.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(QueueHandler(queue))
