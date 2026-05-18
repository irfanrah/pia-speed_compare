import platform
from typing import Union

import torch


def load_model_backend(device: str, type="torch") -> Union[torch.device, str]:
    """Return a torch.device for the requested backend.

    This utility maps common device names like "cpu", "cuda", and
    platform-specific options such as Apple's "mps" to the appropriate
    ``torch.device`` instance. If the requested device is unavailable,
    a CPU device is returned instead.
    """
    device = device.lower()

    if device == "cpu":
        return torch.device("cpu") if type == "torch" else "cpu"

    # macOS: prefer MPS when available even if "cuda" is requested
    if platform.system() == "Darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps") if type == "torch" else "mps"
        else :
            return torch.device("cpu") if type == "torch" else "cpu"
    # CUDA support
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda") if type == "torch" else "cuda"

    return torch.device("cpu") if type == "torch" else "cpu"
