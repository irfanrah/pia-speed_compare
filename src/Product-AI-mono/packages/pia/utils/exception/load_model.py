from typing import Union

import torch
from pia.ai.base import PiaModelBase

__all__ = ["load_state_dict_with_mismatch"]


def load_state_dict_with_mismatch(
    model: PiaModelBase,
    loaded_state_info: Union[str, torch.nn.Module.state_dict],
) -> None:
    """Load a state dict into ``model`` while ignoring mismatched shapes.

    The function operates in-place and does not return ``model``.
    ``loaded_state_info`` can be either a state dict or a path to one.
    """
    if isinstance(loaded_state_info, str):
        loaded_state_dict = torch.load(loaded_state_info, map_location="cpu")
    else:
        loaded_state_dict = loaded_state_info

    model_keys = set(model.state_dict().keys())
    load_keys = set(loaded_state_dict.keys())

    toload = {}
    for key in model_keys & load_keys:
        if model.state_dict()[key].shape == loaded_state_dict[key].shape:
            toload[key] = loaded_state_dict[key]
    model.load_state_dict(toload, strict=False)
