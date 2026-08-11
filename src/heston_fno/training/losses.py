"""Loss functions for training Heston FNO models."""

from __future__ import annotations

import torch
from torch import nn


def relative_l1_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	return torch.mean(torch.abs(prediction - target) / (torch.abs(target) + eps))


def relative_mse_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	return torch.mean(((prediction - target) ** 2) / (target**2 + eps))


def rmse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	return torch.sqrt(nn.functional.mse_loss(prediction, target))
