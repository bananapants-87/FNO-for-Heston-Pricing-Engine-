import torch

from heston_fno.benchmarking import benchmark_adi, benchmark_fno
from heston_fno.benchmarking.adi import solve_heston_adi


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


def test_adi_solver_returns_full_surface_shape() -> None:
	params = torch.tensor([1.5, 0.04, 0.3, -0.5, 0.02], dtype=torch.float32)
	s_grid = torch.linspace(50.0, 150.0, 64, dtype=torch.float32)
	v_grid = torch.linspace(0.01, 0.25, 32, dtype=torch.float32)

	surface = solve_heston_adi(
		params=params,
		s_grid=s_grid,
		v_grid=v_grid,
		strike=100.0,
		r=0.0,
		maturity=1.0,
		n_steps=2,
	)

	assert tuple(surface.shape) == (64, 32)
