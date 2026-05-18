import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def yaml_load(yaml_path):    
    with open(yaml_path, "r") as f:
        cfg =  yaml.safe_load(f)
    args = dict_to_namespace(cfg)
    return args
