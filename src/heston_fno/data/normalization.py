"""Normalization helpers for Heston FNO data."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StandardScaler:
	"""Channel-wise standardization for tensors with a channel dimension."""

	mean: torch.Tensor | None = None
	std: torch.Tensor | None = None

	def fit(self, tensor: torch.Tensor, channel_dim: int = 1) -> "StandardScaler":
		if tensor.ndim < 2:
			raise ValueError("Expected a tensor with at least one channel dimension")

		reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != channel_dim)
		mean = tensor.mean(dim=reduce_dims, keepdim=True)
		std = tensor.std(dim=reduce_dims, unbiased=False, keepdim=True)
		std = torch.where(std == 0, torch.ones_like(std), std)

		self.mean = mean
		self.std = std
		return self

	def transform(self, tensor: torch.Tensor) -> torch.Tensor:
		if self.mean is None or self.std is None:
			raise RuntimeError("StandardScaler must be fit before calling transform")
		mean, std = self._statistics_for(tensor)
		return (tensor - mean) / std

	def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
		if self.mean is None or self.std is None:
			raise RuntimeError("StandardScaler must be fit before calling inverse_transform")
		mean, std = self._statistics_for(tensor)
		return tensor * std + mean

	def _statistics_for(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""Align batch-derived statistics with either a sample or a batch."""
		assert self.mean is not None and self.std is not None
		mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
		std = self.std.to(device=tensor.device, dtype=tensor.dtype)
		if mean.ndim == tensor.ndim + 1 and mean.shape[0] == 1:
			mean = mean.squeeze(0)
			std = std.squeeze(0)
		return mean, std

	def state_dict(self) -> dict[str, torch.Tensor]:
		if self.mean is None or self.std is None:
			raise RuntimeError("StandardScaler must be fit before calling state_dict")
		return {"mean": self.mean, "std": self.std}

	def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
		self.mean = state["mean"]
		self.std = state["std"]
