"""Local, atomic training snapshots and a scoped Windows idle-sleep request."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import random
import warnings

import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(value, path):
    """A failed write leaves the previous snapshot readable."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(cpu_tree(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def random_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None}


def restore_random_state(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def keep_system_awake():
    """Prevent idle sleep while running; never change persistent power settings.

    Windows still honors explicit sleep and lid actions. The screen may turn off.
    """
    previous = None
    if os.name == "nt":
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        previous = function(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not previous:
            warnings.warn("Could not inhibit Windows idle sleep; keep the computer awake manually")
    try:
        yield
    finally:
        if previous:
            function(previous)
