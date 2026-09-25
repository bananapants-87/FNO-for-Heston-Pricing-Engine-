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


def compute_reference_delta(
	price_surface: torch.Tensor,
	s_grid: torch.Tensor,
) -> torch.Tensor:
	"""Compute reference Delta (dV/dS) from a price surface using numerical finite differences."""

	surf = price_surface.detach().cpu()
	s = s_grid.detach().cpu().reshape(-1)

	# Handle [B, C, H, W], [C, H, W], or [H, W]
	orig_shape = surf.shape
	if surf.ndim == 2:
		grad = torch.gradient(surf, spacing=(s,), dim=0)[0]
	elif surf.ndim == 3:
		# [C, H, W]
		grad = torch.stack([torch.gradient(channel, spacing=(s,), dim=0)[0] for channel in surf], dim=0)
	elif surf.ndim == 4:
		# [B, C, H, W]
		grad = torch.stack(
			[torch.stack([torch.gradient(c, spacing=(s,), dim=0)[0] for c in b], dim=0) for b in surf],
			dim=0,
		)
	else:
		raise ValueError(f"Unsupported surface shape: {orig_shape}")

	return grad.reshape(orig_shape)


def compute_reference_vega(
	price_surface: torch.Tensor,
	v_grid: torch.Tensor,
) -> torch.Tensor:
	"""Compute reference Vega/variance sensitivity (dV/dv) from a price surface using numerical finite differences."""

	surf = price_surface.detach().cpu()
	v = v_grid.detach().cpu().reshape(-1)

	orig_shape = surf.shape
	if surf.ndim == 2:
		grad = torch.gradient(surf, spacing=(v,), dim=1)[0]
	elif surf.ndim == 3:
		# [C, H, W]
		grad = torch.stack([torch.gradient(channel, spacing=(v,), dim=1)[0] for channel in surf], dim=0)
	elif surf.ndim == 4:
		# [B, C, H, W]
		grad = torch.stack(
			[torch.stack([torch.gradient(c, spacing=(v,), dim=1)[0] for c in b], dim=0) for b in surf],
			dim=0,
		)
	else:
		raise ValueError(f"Unsupported surface shape: {orig_shape}")

	return grad.reshape(orig_shape)


def compare_greeks(
	fno_greek: torch.Tensor,
	ref_greek: torch.Tensor,
	eps: float = 1e-7,
) -> dict[str, float]:
	"""Compute error metrics comparing FNO-derived Greek with reference Greek."""

	fno_greek = fno_greek.detach().cpu().reshape(-1)
	ref_greek = ref_greek.detach().cpu().reshape(-1)

	mae = torch.mean(torch.abs(fno_greek - ref_greek)).item()
	rmse = torch.sqrt(torch.mean((fno_greek - ref_greek) ** 2)).item()
	max_ae = torch.max(torch.abs(fno_greek - ref_greek)).item()
	rel_l2 = (torch.linalg.norm(fno_greek - ref_greek, ord=2) / (torch.linalg.norm(ref_greek, ord=2) + eps)).item()

	return {
		"mae": float(mae),
		"rmse": float(rmse),
		"max_absolute_error": float(max_ae),
		"relative_l2_error": float(rel_l2),
	}

