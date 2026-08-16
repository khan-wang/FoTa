# uformer_pytorch/utils/random.py

import torch
import numpy as np
import random
import os


def set_all_seeds(seed: int, deterministic: bool = False):
    """
    Sets random seeds for reproducibility across all relevant libraries.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, enables deterministic algorithms in PyTorch,
                              which can impact performance. Recommended for final
                              evaluations, but often disabled for training speed.
    """
    # --- Set basic seeds ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Set CUDA-specific seeds ---
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # --- Configure PyTorch deterministic behavior ---
    if deterministic:
        print(f"INFO: Setting deterministic mode with seed {seed}. Performance may be slightly lower.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        # For CUBLAS reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        # Standard mode: non-deterministic but faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """
    Initializes a DataLoader worker with a unique seed.
    Also restricts thread usage to prevent contention.
    (This function is from the original train.py)
    """
    # The initial seed is set by the main process for the DataLoader.
    # Each worker gets a unique seed derived from this initial seed.
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # --- Threading Optimization ---
    # Avoids contention between multiple DataLoader workers
    try:
        import cv2
        cv2.setNumThreads(0)
    except ImportError:
        pass  # cv2 is not a hard dependency for this function

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)