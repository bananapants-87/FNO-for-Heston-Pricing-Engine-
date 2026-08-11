"""FNO benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .speedup import time_fno_inference


@dataclass(frozen=True)
class BenchmarkResult:
	"""Timing summary for an FNO model."""

	average_time: float
	throughput: float

	def to_dict(self) -> dict[str, float]:
		return {
			"average_time": float(self.average_time),
			"throughput": float(self.throughput),
		}


def benchmark_fno(
	model,
	batch: torch.Tensor,
	device: torch.device | str | None = None,
	warmup: int = 5,
	repeats: int = 20,
) -> BenchmarkResult:
	"""Return timing and throughput information for an FNO model."""

	average_time = time_fno_inference(model, batch, device=device, warmup=warmup, repeats=repeats)
	batch_size = int(batch.shape[0]) if batch.ndim > 0 else 1
	throughput = float(batch_size / average_time) if average_time > 0 else float("inf")
	return BenchmarkResult(average_time=average_time, throughput=throughput)
