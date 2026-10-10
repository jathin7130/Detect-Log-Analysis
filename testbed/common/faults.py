"""Shared fault-injection state machine for every testbed microservice.

A fault is "armed" via POST /_fault {type, duration_s} and stays active for
that many seconds from when it's armed, then self-clears — the scenario
driver doesn't need a separate "clear" call, just a known injection
timestamp and duration, which is what the ground-truth record needs anyway.
"""
import os
import random
import threading
import time

_lock = threading.Lock()
_active_fault = {"type": None, "until": 0.0}


def arm_fault(fault_type: str, duration_s: float):
    with _lock:
        _active_fault["type"] = fault_type
        _active_fault["until"] = time.time() + duration_s


def current_fault():
    with _lock:
        if _active_fault["type"] and time.time() < _active_fault["until"]:
            return _active_fault["type"]
        return None


def maybe_restart_process(logger):
    """pod_restart fault: log a fatal line and exit; docker-compose's
    restart policy brings the container back up, which is the point."""
    if current_fault() == "pod_restart":
        logger.error("unhandled exception in request handler, process terminating")
        os._exit(1)


def maybe_delay_or_fail(logger, downstream_name: str = None):
    """Applies the currently-armed fault to a request, if any.
    Returns True if the request should fail outright (caller raises/503s).
    """
    fault = current_fault()
    if fault is None:
        return False

    if fault == "dependency_timeout" and downstream_name:
        logger.warning(f"downstream call to {downstream_name} exceeded timeout threshold, retrying")
        time.sleep(random.uniform(2.0, 4.0))
        return True

    if fault == "db_failure":
        logger.error("database connection failed: could not connect to postgres backend")
        return True

    if fault == "resource_exhaustion":
        logger.warning(f"high memory usage detected: {random.randint(88, 99)}% of limit")
        time.sleep(random.uniform(0.5, 1.5))
        return random.random() < 0.6  # degraded, not always a hard failure

    if fault == "pod_restart":
        maybe_restart_process(logger)

    return False
