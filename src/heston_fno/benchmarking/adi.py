"""ADI benchmark helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
	"""Timing summary for a reference ADI solver."""

	average_time: float
	throughput: float

	def to_dict(self) -> dict[str, float]:
		return {
			"average_time": float(self.average_time),
			"throughput": float(self.throughput),
		}


def time_adi_inference(
	adi_fn: Callable[[torch.Tensor], torch.Tensor | float],
	batch: torch.Tensor,
	warmup: int = 5,
	repeats: int = 20,
) -> float:
	"""Measure average wall-clock runtime for an ADI reference callable."""

	batch = batch.detach().cpu()
	for _ in range(warmup):
		_ = adi_fn(batch)

	start = time.perf_counter()
	for _ in range(repeats):
		_ = adi_fn(batch)
	elapsed = time.perf_counter() - start
	return elapsed / max(repeats, 1)


def benchmark_adi(
	adi_fn: Callable[[torch.Tensor], torch.Tensor | float],
	batch: torch.Tensor,
	warmup: int = 5,
	repeats: int = 20,
) -> BenchmarkResult:
	"""Return timing and throughput information for an ADI baseline."""

	average_time = time_adi_inference(adi_fn, batch, warmup=warmup, repeats=repeats)
	batch_size = int(batch.shape[0]) if batch.ndim > 0 else 1
	throughput = float(batch_size / average_time) if average_time > 0 else float("inf")
	return BenchmarkResult(average_time=average_time, throughput=throughput)
