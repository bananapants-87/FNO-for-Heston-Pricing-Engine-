"""Speedup benchmarking helpers."""

from __future__ import annotations

import time
from typing import Callable

import torch


def time_fno_inference(
	model,
	batch: torch.Tensor,
	device: torch.device | str | None = None,
	warmup: int = 5,
	repeats: int = 20,
) -> float:
	"""Measure average wall-clock inference time for an FNO batch."""

	device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	model.eval()
	batch = batch.to(device)

	with torch.no_grad():
		for _ in range(warmup):
			_ = model(batch)

		if device.type == "cuda":
			torch.cuda.synchronize()

		start = time.perf_counter()
		for _ in range(repeats):
			_ = model(batch)
		if device.type == "cuda":
			torch.cuda.synchronize()
		elapsed = time.perf_counter() - start

	return elapsed / max(repeats, 1)


def benchmark_speedup(
	model,
	adi_fn: Callable[[torch.Tensor], torch.Tensor | float],
	batch: torch.Tensor,
	device: torch.device | str | None = None,
	repeats: int = 20,
) -> dict[str, float]:
	"""Compare FNO inference time against an ADI reference callable."""

	fno_time = time_fno_inference(model, batch, device=device, repeats=repeats)

	start = time.perf_counter()
	for _ in range(repeats):
		_ = adi_fn(batch.cpu())
	adi_time = (time.perf_counter() - start) / max(repeats, 1)

	return {
		"fno_time": float(fno_time),
		"adi_time": float(adi_time),
		"speedup": float(adi_time / fno_time) if fno_time > 0 else float("inf"),
	}
