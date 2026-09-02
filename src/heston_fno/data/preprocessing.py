"""Preprocessing helpers for Heston FNO data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import HestonDataset
from .normalization import StandardScaler


class TransformedSubset(Dataset):
	"""Subset wrapper that applies a transform to each sample."""

	def __init__(self, dataset: Dataset, indices: Sequence[int], transform=None) -> None:
		self.dataset = dataset
		self.indices = list(indices)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, idx: int):
		sample = self.dataset[self.indices[idx]]
		if self.transform is not None:
			return self.transform(*sample)
		return sample


def _normalise_split(split: Sequence[float]) -> tuple[float, float, float]:
	if len(split) != 3:
		raise ValueError("split must contain train, validation, and test fractions")
	total = float(sum(split))
	if total <= 0:
		raise ValueError("split fractions must be positive")
	train, val, test = (float(value) / total for value in split)
	return train, val, test


def split_indices(num_samples: int, split: Sequence[float] = (0.8, 0.1, 0.1), seed: int | None = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	train_frac, val_frac, _ = _normalise_split(split)
	generator = torch.Generator()
	if seed is not None:
		generator.manual_seed(seed)

	permutation = torch.randperm(num_samples, generator=generator)
	train_end = int(train_frac * num_samples)
	val_end = train_end + int(val_frac * num_samples)
	return permutation[:train_end], permutation[train_end:val_end], permutation[val_end:]


def _accumulate_channel_stats(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
	total_sum = None
	total_sq_sum = None
	total_count = 0

	for batch_x, _ in loader:
		batch_sum = batch_x.sum(dim=(0, 2, 3), keepdim=True)
		batch_sq_sum = (batch_x * batch_x).sum(dim=(0, 2, 3), keepdim=True)
		batch_count = batch_x.shape[0] * batch_x.shape[2] * batch_x.shape[3]

		total_sum = batch_sum if total_sum is None else total_sum + batch_sum
		total_sq_sum = batch_sq_sum if total_sq_sum is None else total_sq_sum + batch_sq_sum
		total_count += batch_count

	if total_sum is None or total_sq_sum is None or total_count == 0:
		raise RuntimeError("Unable to compute statistics from an empty dataset")

	mean = total_sum / total_count
	variance = total_sq_sum / total_count - mean * mean
	std = torch.sqrt(torch.clamp(variance, min=0.0))
	std = torch.where(std == 0, torch.ones_like(std), std)
	return mean, std


def fit_scalers(
	dataset: Dataset,
	indices: Sequence[int],
	batch_size: int = 128,
	num_workers: int = 0,
	pin_memory: bool = False,
) -> tuple[StandardScaler, StandardScaler]:
	subset = Subset(dataset, list(indices))
	loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

	input_mean, input_std = _accumulate_channel_stats(loader)

	target_total_sum = None
	target_total_sq_sum = None
	target_total_count = 0
	for _, batch_y in loader:
		batch_sum = batch_y.sum(dim=(0, 2, 3), keepdim=True)
		batch_sq_sum = (batch_y * batch_y).sum(dim=(0, 2, 3), keepdim=True)
		batch_count = batch_y.shape[0] * batch_y.shape[2] * batch_y.shape[3]
		target_total_sum = batch_sum if target_total_sum is None else target_total_sum + batch_sum
		target_total_sq_sum = batch_sq_sum if target_total_sq_sum is None else target_total_sq_sum + batch_sq_sum
		target_total_count += batch_count

	if target_total_sum is None or target_total_sq_sum is None or target_total_count == 0:
		raise RuntimeError("Unable to compute target statistics from an empty dataset")

	target_mean = target_total_sum / target_total_count
	target_variance = target_total_sq_sum / target_total_count - target_mean * target_mean
	target_std = torch.sqrt(torch.clamp(target_variance, min=0.0))
	target_std = torch.where(target_std == 0, torch.ones_like(target_std), target_std)

	input_scaler = StandardScaler(mean=input_mean, std=input_std)
	target_scaler = StandardScaler(mean=target_mean, std=target_std)
	return input_scaler, target_scaler


def prepare_datasets(
	data_dir: str | Path,
	split: Sequence[float] = (0.8, 0.1, 0.1),
	batch_size: int = 128,
	seed: int | None = 42,
	num_workers: int = 0,
	pin_memory: bool = False,
) -> tuple[Dataset, Dataset, Dataset, StandardScaler, StandardScaler]:
	data_dir = Path(data_dir)
	full_dataset = HestonDataset(
		inputs_path=data_dir / "inputs.pt",
		targets_path=data_dir / "targets.pt",
		s_grid_path=data_dir / "s_grid.pt",
		v_grid_path=data_dir / "v_grid.pt",
	)

	train_idx, val_idx, test_idx = split_indices(len(full_dataset), split=split, seed=seed)
	input_scaler, target_scaler = fit_scalers(
		full_dataset,
		indices=train_idx.tolist(),
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	def transform_fn(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		return input_scaler.transform(x), target_scaler.transform(y)

	train_set = TransformedSubset(full_dataset, train_idx.tolist(), transform=transform_fn)
	val_set = TransformedSubset(full_dataset, val_idx.tolist(), transform=transform_fn)
	test_set = TransformedSubset(full_dataset, test_idx.tolist(), transform=transform_fn)
	return train_set, val_set, test_set, input_scaler, target_scaler
