"""Dataset utilities for the Heston FNO surrogate project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


def build_input_tensor(
	params: torch.Tensor,
	s_grid: torch.Tensor,
	v_grid: torch.Tensor,
) -> torch.Tensor:
	"""Build a 7-channel input tensor from Heston parameters and grids."""

	params = params.float().reshape(-1)
	s_grid = s_grid.float().reshape(-1)
	v_grid = v_grid.float().reshape(-1)

	s_mesh, v_mesh = torch.meshgrid(s_grid, v_grid, indexing="ij")
	param_channels = params.view(-1, 1, 1).expand(-1, s_grid.numel(), v_grid.numel())
	coord_channels = torch.stack([s_mesh, v_mesh], dim=0)
	return torch.cat([param_channels, coord_channels], dim=0)


class HestonDataset(Dataset):
	"""Dataset mapping Heston parameters to option-price surfaces."""

	def __init__(
		self,
		inputs_path: str | Path,
		targets_path: str | Path,
		s_grid_path: str | Path,
		v_grid_path: str | Path,
		transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
	) -> None:
		self.params = torch.load(Path(inputs_path), map_location="cpu").float()
		self.targets = torch.load(Path(targets_path), map_location="cpu").float()
		self.s_grid = torch.load(Path(s_grid_path), map_location="cpu").float().reshape(-1)
		self.v_grid = torch.load(Path(v_grid_path), map_location="cpu").float().reshape(-1)
		self.transform = transform

		if self.params.ndim != 2:
			raise ValueError("inputs.pt must contain a [N, 5] tensor of Heston parameters")
		if self.targets.ndim != 3:
			raise ValueError("targets.pt must contain a [N, H, W] tensor of price surfaces")

		self.num_samples = self.params.shape[0]
		expected_shape = (self.num_samples, self.s_grid.numel(), self.v_grid.numel())
		if tuple(self.targets.shape) != expected_shape:
			raise ValueError(f"targets shape {tuple(self.targets.shape)} does not match {expected_shape}")

		s_mesh, v_mesh = torch.meshgrid(self.s_grid, self.v_grid, indexing="ij")
		self._coord_channels = torch.stack([s_mesh, v_mesh], dim=0)

	@property
	def in_channels(self) -> int:
		return self.params.shape[1] + 2

	@property
	def out_channels(self) -> int:
		return 1

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		params = self.params[idx].float()
		target = self.targets[idx].float().unsqueeze(0)

		param_channels = params.view(-1, 1, 1).expand(-1, self.s_grid.numel(), self.v_grid.numel())
		x = torch.cat([param_channels, self._coord_channels], dim=0)

		if self.transform is not None:
			x, target = self.transform(x, target)
		return x, target
