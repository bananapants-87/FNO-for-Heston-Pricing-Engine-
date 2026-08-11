import torch

from heston_fno.benchmarking import benchmark_adi, benchmark_fno


def test_benchmark_helpers_return_positive_timing() -> None:
	batch = torch.ones(4, 2, 3)

	def adi_fn(x: torch.Tensor) -> torch.Tensor:
		return x.sum(dim=-1)

	class DummyModel:
		def eval(self) -> None:
			pass

		def __call__(self, x: torch.Tensor) -> torch.Tensor:
			return x.unsqueeze(1)

	adi_result = benchmark_adi(adi_fn, batch, warmup=1, repeats=2)
	fno_result = benchmark_fno(DummyModel(), batch, repeats=2)

	assert adi_result.average_time >= 0.0
	assert adi_result.throughput >= 0.0
	assert fno_result.average_time >= 0.0
	assert fno_result.throughput >= 0.0
