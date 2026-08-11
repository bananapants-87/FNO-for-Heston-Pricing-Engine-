"""Error metrics for model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Return the element-wise absolute error."""

	return torch.abs(prediction - target)


def mean_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Return the mean absolute error."""

	return absolute_error(prediction, target).mean()


def relative_error(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Return the element-wise relative error."""

	return torch.abs(prediction - target) / (torch.abs(target) + eps)


def mean_relative_error(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Return the mean relative error."""

	return relative_error(prediction, target, eps=eps).mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Return the root mean squared error."""

	return torch.sqrt(torch.mean((prediction - target) ** 2))


def max_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	"""Return the maximum absolute error."""

	return absolute_error(prediction, target).max()


@dataclass(frozen=True)
class ErrorSummary:
	"""Compact collection of common error metrics."""

	mae: torch.Tensor
	mre: torch.Tensor
	rmse: torch.Tensor
	max_ae: torch.Tensor

	def to_dict(self) -> dict[str, float]:
		"""Convert the summary to plain Python floats."""

		return {
			"mean_absolute_error": float(self.mae.item()),
			"mean_relative_error": float(self.mre.item()),
			"rmse": float(self.rmse.item()),
			"max_absolute_error": float(self.max_ae.item()),
		}


def summarize_errors(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> ErrorSummary:
	"""Bundle the most useful scalar metrics in one object."""

	return ErrorSummary(
		mae=mean_absolute_error(prediction, target),
		mre=mean_relative_error(prediction, target, eps=eps),
		rmse=rmse(prediction, target),
		max_ae=max_absolute_error(prediction, target),
	)


summarise_errors = summarize_errors
