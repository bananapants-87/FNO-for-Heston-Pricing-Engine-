"""Fourier neural operator model definitions."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _load_fno_class():
	try:
		from neuralop.models import FNO
	except ImportError as exc:  # pragma: no cover - dependency specific
		raise ImportError(
			"The neuraloperator package is required. Install it with `pip install neuraloperator`."
		) from exc
	return FNO


class HestonFNO(nn.Module):
	"""Thin wrapper around neuralop.models.FNO for the Heston surrogate."""

	def __init__(
		self,
		n_modes: Sequence[int] | int,
		hidden_channels: int,
		in_channels: int = 7,
		out_channels: int = 1,
		**kwargs,
	) -> None:
		super().__init__()
		FNO = _load_fno_class()
		self.model = FNO(
			n_modes=n_modes,
			hidden_channels=hidden_channels,
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)


def build_model_from_config(config: dict) -> HestonFNO:
	model_cfg = config.get("model", {})
	return HestonFNO(
		n_modes=tuple(model_cfg.get("n_modes", (16, 16))),
		hidden_channels=int(model_cfg.get("hidden_channels", 64)),
		in_channels=int(model_cfg.get("in_channels", 7)),
		out_channels=int(model_cfg.get("out_channels", 1)),
	)
