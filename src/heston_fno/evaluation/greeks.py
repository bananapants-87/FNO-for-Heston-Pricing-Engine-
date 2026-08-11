"""Greeks evaluation helpers."""

from __future__ import annotations

import torch


def compute_delta(
	model,
	x: torch.Tensor,
	input_scaler=None,
	target_scaler=None,
	s_channel: int = 5,
	device: torch.device | str | None = None,
) -> torch.Tensor:
	"""Compute a Delta-like sensitivity by differentiating the output wrt the S channel."""

	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	x = x.clone().detach().to(device)
	x.requires_grad_(True)

	model.eval()
	prediction = model(x)
	if target_scaler is not None:
		prediction = target_scaler.inverse_transform(prediction)

	gradient = torch.autograd.grad(prediction.sum(), x, retain_graph=False, create_graph=False)[0]
	delta = gradient[:, s_channel : s_channel + 1, :, :]

	if input_scaler is not None and getattr(input_scaler, "std", None) is not None:
		delta = delta / input_scaler.std[:, s_channel : s_channel + 1, :, :]

	return delta.detach().cpu()


def compute_vega(
	model,
	x: torch.Tensor,
	input_scaler=None,
	target_scaler=None,
	v0_channel: int = 4,
	device: torch.device | str | None = None,
) -> torch.Tensor:
	"""Compute a Vega-like sensitivity by differentiating wrt the v0 channel."""

	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	x = x.clone().detach().to(device)
	x.requires_grad_(True)

	model.eval()
	prediction = model(x)
	if target_scaler is not None:
		prediction = target_scaler.inverse_transform(prediction)

	gradient = torch.autograd.grad(prediction.sum(), x, retain_graph=False, create_graph=False)[0]
	vega = gradient[:, v0_channel : v0_channel + 1, :, :]

	if input_scaler is not None and getattr(input_scaler, "std", None) is not None:
		vega = vega / input_scaler.std[:, v0_channel : v0_channel + 1, :, :]

	return vega.detach().cpu()
