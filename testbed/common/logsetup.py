"""Shared structured-text logging setup for every testbed microservice.

Deliberately text-line logs (timestamp level service: message) rather than
JSON — this is meant to exercise the same Drain3-parsing pipeline built for
HDFS/BGL/Thunderbird, not a separate structured-log ingestion path.
"""
import logging
import os
import sys


def setup_logging(service_name: str):
    log_dir = "/var/log/testbed"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt=f"%(asctime)s %(levelname)s {service_name}: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(f"{log_dir}/{service_name}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
