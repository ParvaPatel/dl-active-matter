"""Training utilities: seeding, checkpointing, logging."""

import os
import random
import torch
import numpy as np


def set_seed(seed=42):
    """Fix all random seeds for reproducibility.

    Note: cudnn.benchmark=True makes results non-bit-reproducible across runs,
    but gives 10-20% speedup for fixed-size inputs (every sample is same shape).
    This trade-off is standard in SSL research.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def log_gpu_memory(tag=""):
    """Log current GPU memory usage. Call after first batch to check headroom."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    utilization = max_allocated / total * 100
    print(
        f"GPU Memory {tag}: "
        f"Allocated={allocated:.1f}GB | Reserved={reserved:.1f}GB | "
        f"Peak={max_allocated:.1f}GB / {total:.0f}GB ({utilization:.0f}%)",
        flush=True,
    )
    return {"allocated_gb": allocated, "peak_gb": max_allocated, "total_gb": total, "utilization_pct": utilization}


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pt"):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """Load checkpoint and return the epoch to resume from."""
    if not os.path.exists(checkpoint_path):
        return 0

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0)

