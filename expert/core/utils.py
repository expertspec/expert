import torch
import os


def get_torch_home() -> str:
    """Get Torch Hub cache directory used for storing downloaded models and weights."""
    
    torch_home = os.path.expanduser(
        os.getenv(
            "TORCH_HOME",
            os.path.join(os.getenv("DG_CACHE_HOME", "~/.cache"), "torch")
        )
    )
    
    return torch_home