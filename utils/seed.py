# utils/seed.py

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Fully deterministic setup.
    Must be called before ANY model/dataloader creation.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    print(f"[Seed] Global seed set to {seed}")