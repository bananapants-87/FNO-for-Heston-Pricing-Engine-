"""Generalization analysis helpers."""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from ..data.dataset import build_input_tensor


def compare_carr_madan(
	model,
	params: torch.Tensor | Sequence[float],
	s_grid: torch.Tensor,
	v_grid: torch.Tensor,
	reference_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
	input_scaler=None,
	target_scaler=None,
	device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
	"""Compare FNO predictions against a reference price surface."""

	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	params = torch.as_tensor(params, dtype=torch.float32)
	x = build_input_tensor(params, s_grid, v_grid).unsqueeze(0)

	if input_scaler is not None:
		x = input_scaler.transform(x)

	model.eval()
	with torch.no_grad():
		prediction = model(x.to(device))
		if target_scaler is not None:
			prediction = target_scaler.inverse_transform(prediction)

	reference = reference_fn(params, s_grid, v_grid)
	reference = torch.as_tensor(reference, dtype=prediction.dtype, device=prediction.device)
	if reference.ndim == 2:
		reference = reference.unsqueeze(0).unsqueeze(0)
	elif reference.ndim == 3:
		reference = reference.unsqueeze(0)

	abs_error = torch.mean(torch.abs(prediction - reference))
	rel_error = torch.mean(torch.abs(prediction - reference) / (torch.abs(reference) + 1e-6))
	return {
		"prediction": prediction.detach().cpu(),
		"reference": reference.detach().cpu(),
		"absolute_error": abs_error.detach().cpu(),
		"relative_error": rel_error.detach().cpu(),
	}
