"""
Usage:
@deterministic
def evaluate(args):
    # code     
"""

import torch
import numpy as np
import random
from contextlib import contextmanager

SEED = 0

@contextmanager
def deterministic_mode(seed=SEED):
    """
    A context manager that temporarily sets the random seeds and PyTorch deterministic settings.
    """
    # Save the current random states and settings
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()
    else:
        cuda_rng_state = None
    np_rng_state = np.random.get_state()
    random_state = random.getstate()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        # Set the deterministic settings
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield  # Control is transferred to the code inside the with-block
    finally:
        # Restore the original random states and settings
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(random_state)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def deterministic(func):
    def wrapper(*args, **kwargs):
        with deterministic_mode(seed=SEED):
            return func(*args, **kwargs)

    return wrapper
