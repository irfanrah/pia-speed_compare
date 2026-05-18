import json
from pathlib import Path
from typing import Any, Union

import yaml


def save_json(obj: Any, fp: Union[str, Path], create_directory: bool = False):
    """save json file

    Args:
        obj (Any): Any object that can be converted to json format.
        fp (Union[str, Path]): file path.
        create_directory (bool, optional): this determines whether create parent directory or not. Default to False.
    """

    fp = Path(fp)
    if create_directory:
        make_directory(fp.parent)

    with open(fp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def load_json(fp: Union[str, Path]) -> dict:
    """load json file

    Args:
        fp (Union[str, Path]): file path.

    Returns:
        dict: dictionary
    """

    fp = Path(fp)

    if not fp.exists():
        raise FileNotFoundError(f"{fp} does not exists")

    with open(fp) as f:
        d = json.load(f)

    return d


def save_yaml(obj: Any, fp: Union[str, Path], create_directory: bool = False):
    """save yaml file

    Args:
        obj (Any): Any object that can be converted to yaml format.
        fp (Union[str, Path]): file path.
        create_directory (bool, optional): this determines whether create parent directory or not. Default to False.
    """

    fp = Path(fp)
    if create_directory:
        make_directory(fp.parent)

    with open(fp, "w") as f:
        yaml.safe_dump(obj, f, indent=4, sort_keys=False)


def load_yaml(fp: Union[str, Path]) -> dict:
    """load yaml file

    Args:
        fp (Union[str, Path]): file path.

    Returns:
        dict: dictionary
    """

    fp = Path(fp)

    if not fp.exists():
        raise FileNotFoundError(f"{fp} does not exists")

    with open(fp) as f:
        d = yaml.safe_load(f)

    return d


def make_directory(src: Union[str, Path]):
    """Create Directory

    Args:
        src (str): directory
    """
    Path(src).mkdir(mode=0o766, parents=True, exist_ok=True)


def read_jsonl(file_path):
    with open(file_path, encoding="utf-8") as file:
        json_list = [json.loads(line.strip()) for line in file]
    return json_list
