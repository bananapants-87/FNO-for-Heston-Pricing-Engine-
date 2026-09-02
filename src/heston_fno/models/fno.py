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
		self.n_modes = _normalise_n_modes(n_modes)
		self.in_channels = in_channels
		FNO = _load_fno_class()
		self.model = FNO(
			n_modes=self.n_modes,
			hidden_channels=hidden_channels,
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		expected_rank = len(self.n_modes) + 2
		if x.ndim != expected_rank:
			raise ValueError(
				"HestonFNO input rank does not match n_modes: "
				f"n_modes={self.n_modes} requires a tensor shaped "
				f"[batch, channels, *{len(self.n_modes)} spatial dimensions] "
				f"(rank {expected_rank}), but received shape {tuple(x.shape)}."
			)
		if x.shape[1] != self.in_channels:
			raise ValueError(
				f"HestonFNO expects {self.in_channels} input channels, "
				f"but received shape {tuple(x.shape)}."
			)
		return self.model(x)


def _normalise_n_modes(n_modes: Sequence[int] | int) -> tuple[int, ...]:
	"""Return a validated mode count for each FNO spatial dimension."""
	if isinstance(n_modes, int):
		n_modes = (n_modes,)
	else:
		n_modes = tuple(n_modes)
	if not n_modes or any(not isinstance(mode, int) or mode <= 0 for mode in n_modes):
		raise ValueError("n_modes must contain one or more positive integers")
	return n_modes


def build_model_from_config(config: dict) -> HestonFNO:
	model_cfg = config.get("model", {})
	return HestonFNO(
		n_modes=model_cfg.get("n_modes", (16, 16)),
		hidden_channels=int(model_cfg.get("hidden_channels", 64)),
		in_channels=int(model_cfg.get("in_channels", 7)),
		out_channels=int(model_cfg.get("out_channels", 1)),
	)
