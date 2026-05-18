import torch
import numpy as np
import json


def load_text_feature(json_path, device="cuda", target_class=None):
    """
    loads prompts and features from json_path, optionally filtering by target_class list

    Notes:
        model: PE-Core-L14-336

        class:
            0: normal
            1: falldown
            2: fire
            3: smoke
            4: smoking
            5: esfalldown
            6: elvfalldown

        usage:
                        # all classes
            class_list, prompt_list, text_features = load_text_feature('./text_features.json')
            text_features = text_features.cuda()

            # select classes
            class_list, prompt_list, text_features = load_text_feature(
                                                    './text_features.json',
                                                    target_class=[0, 2])
            text_features = text_features.cuda()

    Args:
        json_path (str):
            path to json file containing items with keys 'class', 'prompt', 'feature'
        target_class (list[int], optional):
            list of class ids to keep. if None or empty, all are returned

    Returns:
        class_list (np.ndarray): array of kept class ids
        prompt_list (np.ndarray): array of kept prompts
        text_features (torch.Tensor): tensor of kept features
    """
    # load all entries
    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # if target_class is provided and non-empty, filter
    if target_class:
        items = [it for it in items if it['class'] in target_class]

    # extract arrays
    ID_list = np.array([it['ID'] for it in items])
    class_list = np.array([it['class'] for it in items])
    prompt_list = np.array([it['prompt'] for it in items])
    features_list = [it['feature'] for it in items]

    # convert to tensor
    text_features = torch.tensor(features_list, dtype=torch.float32).to(device)

    return ID_list, class_list, prompt_list, text_features
