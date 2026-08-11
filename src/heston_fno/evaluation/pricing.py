"""Pricing evaluation helpers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def _maybe_inverse_transform(tensor: torch.Tensor, scaler) -> torch.Tensor:
	if scaler is None:
		return tensor
	return scaler.inverse_transform(tensor)


def pricing_error(
	model,
	dataset,
	device: torch.device | str | None = None,
	target_scaler=None,
	batch_size: int = 16,
) -> tuple[float, float]:
	"""Return RMSE and relative error on a dataset."""

	model.eval()
	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	total_squared_error = 0.0
	total_relative_error = 0.0
	total_elements = 0

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			prediction = model(x)

			prediction = _maybe_inverse_transform(prediction, target_scaler)
			y = _maybe_inverse_transform(y, target_scaler)

			total_squared_error += torch.sum((prediction - y) ** 2).item()
			total_relative_error += torch.sum(torch.abs(prediction - y) / (torch.abs(y) + 1e-6)).item()
			total_elements += y.numel()

	rmse = (total_squared_error / max(total_elements, 1)) ** 0.5
	mean_relative_error = total_relative_error / max(total_elements, 1)
	return float(rmse), float(mean_relative_error)


def predict_surface(model, x: torch.Tensor, target_scaler=None, device: torch.device | str | None = None) -> torch.Tensor:
	"""Predict a single normalized or physical price surface."""

	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.eval()
	with torch.no_grad():
		prediction = model(x.to(device))
		prediction = _maybe_inverse_transform(prediction, target_scaler)
	return prediction.detach().cpu()
