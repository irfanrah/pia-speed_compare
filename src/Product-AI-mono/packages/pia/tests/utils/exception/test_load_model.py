import torch

from pia.utils.exception.load_model import load_state_dict_with_mismatch


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)


def test_load_state_dict_with_mismatch_dict():
    model = DummyModel()
    original = {k: v.clone() for k, v in model.state_dict().items()}

    loaded = {k: v.clone() + 1 for k, v in original.items()}
    loaded["linear2.weight"] = torch.randn(3, 3)

    load_state_dict_with_mismatch(model, loaded)

    assert torch.equal(model.linear1.weight, original["linear1.weight"] + 1)
    assert torch.equal(model.linear1.bias, original["linear1.bias"] + 1)
    assert torch.equal(model.linear2.bias, original["linear2.bias"] + 1)
    assert torch.equal(model.linear2.weight, original["linear2.weight"])


def test_load_state_dict_with_mismatch_path(tmp_path):
    model = DummyModel()
    original = {k: v.clone() for k, v in model.state_dict().items()}

    loaded = {k: v.clone() + 2 for k, v in original.items()}
    loaded["linear1.weight"] = torch.randn(5, 5)

    path = tmp_path / "state.pt"
    torch.save(loaded, path)

    load_state_dict_with_mismatch(model, str(path))

    assert torch.equal(model.linear1.weight, original["linear1.weight"])
    assert torch.equal(model.linear1.bias, original["linear1.bias"] + 2)
    assert torch.equal(model.linear2.weight, original["linear2.weight"] + 2)
    assert torch.equal(model.linear2.bias, original["linear2.bias"] + 2)
