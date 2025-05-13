import logging
import socket

def get_logger(name: str) -> logging.Logger:
    # set logging at INFO level
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(f"[{socket.gethostname()}] {name}")
