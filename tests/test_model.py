import pytest
import torch
from torch import nn

import heston_fno.models.fno as fno_module


class _DummyFNO(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_model_rejects_input_with_wrong_spatial_rank(monkeypatch) -> None:
    monkeypatch.setattr(fno_module, "_load_fno_class", lambda: _DummyFNO)
    model = fno_module.HestonFNO(n_modes=(16, 16), hidden_channels=8)

    with pytest.raises(ValueError, match="n_modes=\\(16, 16\\)"):
        model(torch.randn(2, 7, 32))


def test_model_accepts_matching_two_dimensional_input(monkeypatch) -> None:
    monkeypatch.setattr(fno_module, "_load_fno_class", lambda: _DummyFNO)
    model = fno_module.HestonFNO(n_modes=(16, 16), hidden_channels=8)

    x = torch.randn(2, 7, 32, 24)
    assert model(x) is x


def test_build_model_accepts_a_single_integer_mode(monkeypatch) -> None:
    monkeypatch.setattr(fno_module, "_load_fno_class", lambda: _DummyFNO)
    model = fno_module.build_model_from_config({"model": {"n_modes": 16}})

    assert model.n_modes == (16,)
