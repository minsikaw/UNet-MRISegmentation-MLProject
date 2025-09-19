import random
import numpy as np
import torch


def set_seed(seed: int):
    """Sets the random seed for reproducibility across NumPy and PyTorch.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)