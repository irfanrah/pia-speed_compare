import platform
from unittest import mock

import torch
from pia.ai.device import load_model_backend


def test_load_model_backend_cpu():
    assert load_model_backend("cpu") == torch.device("cpu")


def test_load_model_backend_cuda_available():
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert load_model_backend("cuda") == torch.device("cuda")


def test_load_model_backend_cuda_unavailable():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert load_model_backend("cuda") == torch.device("cpu")


def test_load_model_backend_mps_available(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with mock.patch("torch.backends.mps.is_available", return_value=True):
        assert load_model_backend("cuda") == torch.device("mps")


def test_load_model_backend_mps_unavailable(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    with mock.patch("torch.backends.mps.is_available", return_value=False):
        assert load_model_backend("cuda") == torch.device("cpu")
